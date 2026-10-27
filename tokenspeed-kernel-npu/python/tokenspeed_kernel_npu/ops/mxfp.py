# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Expand checkpoint MXFP bytes without creating FP8/FP4 device tensors."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _decode_e4m3(code):
    code = code.to(tl.int32)
    exponent, mantissa = (code >> 3) & 15, code & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, mantissa.to(tl.float32) * (2.0**-9), normal)
    value = tl.where((code & 127) == 127, float("nan"), value)
    return tl.where((code & 128) != 0, -value, value)


@triton.jit
def _decode_e8m0(scale):
    scale = scale.to(tl.int32)
    value = tl.where(scale == 0, 1 << 22, scale << 23).to(tl.float32, bitcast=True)
    return tl.where(scale == 255, float("nan"), value)


@triton.jit
def _decode_e2m1(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(
            magnitude == 4,
            2.0,
            tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _dequantize_mxfp_weight(
    W,
    S,
    Y,
    N,
    K,
    W0,
    W1,
    S0,
    S1,
    BITS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TILE: tl.constexpr,
):
    # A bounded grid also handles whole expert shards, which exceed CANN's
    # launch-grid limit when every tile is a separate program.
    for start in range(
        tl.program_id(0).to(tl.int64) * TILE,
        N.to(tl.int64) * K,
        tl.num_programs(0) * TILE,
    ):
        offsets = start + tl.arange(0, TILE)
        row, col = offsets // K, offsets % K
        code = tl.load(W + row * W0 + (col // (8 // BITS)) * W1, row < N, other=0).to(
            tl.int32
        )
        scale = tl.load(S + (row // BN) * S0 + (col // BK) * S1, row < N, other=127).to(
            tl.int32
        )
        if BITS == 4:
            code = (code >> ((col % 2) * 4)) & 15
            value = _decode_e2m1(code)
        else:
            value = _decode_e4m3(code)
        value = value * _decode_e8m0(scale)
        tl.store(Y + offsets, value, row < N)


def dequantize_mxfp_weight(weight, scales, bits, block_size):
    rows, packed_width = weight.shape
    width = packed_width * (8 // bits)
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=weight.device)
    if out.numel():
        with torch.npu.device(weight.device):
            _dequantize_mxfp_weight[(min(1024, triton.cdiv(out.numel(), 1024)),)](
                weight,
                scales,
                out,
                rows,
                width,
                *weight.stride(),
                *scales.stride(),
                bits,
                *block_size,
                1024,
            )
    return out


def process_mxfp4_bf16_moe_weights(*, plan, w):
    """Expand only the local expert shard, then use the existing BF16 GMM layout."""
    from tokenspeed_kernel_npu.ops.moe import ascend_bf16_process_moe_weights

    for name in ("w13_weight", "w2_weight"):
        weight = w.get_parameter(name)
        if weight.dtype == torch.uint8:
            scales = w.get_parameter(name + "_scale")
            shape = weight.shape
            decoded = dequantize_mxfp_weight(
                weight.flatten(0, 1),
                scales.flatten(0, 1),
                4,
                (1, 32),
            )
            weight.data = decoded.reshape(shape[0], shape[1], shape[2] * 2)
            w.register_parameter(name + "_scale", None)
        elif weight.dtype != torch.bfloat16:
            raise TypeError("Expected packed MXFP4 or prepared BF16 experts")
    ascend_bf16_process_moe_weights(plan=plan, w=w)


def preserve_mxfp4_moe_weights(*, plan, w):
    """Transpose checkpoint bytes for contiguous on-demand BF16 expansion."""
    for name, n, k in (
        ("w13_weight", 2 * w.intermediate_size, w.hidden_size),
        ("w2_weight", w.hidden_size, w.intermediate_size),
    ):
        weight = w.get_parameter(name)
        scale = w.get_parameter(name + "_scale")
        if (
            weight.dtype != torch.uint8
            or scale.dtype != torch.uint8
            or weight.shape != (w.num_local_experts, n, k // 2)
            or scale.shape != (w.num_local_experts, n, k // 32)
            or not weight.is_contiguous()
            or not scale.is_contiguous()
        ):
            raise ValueError("Expected contiguous canonical MXFP4 expert weights")
        # Match GMM's [input, output] traversal without expanding the values.
        weight.data = weight.transpose(1, 2).contiguous()
        scale.data = scale.transpose(1, 2).contiguous()


@triton.jit
def _expand_active_experts(
    W, S, Counts, ExpertIds, Y, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    group = tl.program_id(0).to(tl.int64)
    expert = tl.load(ExpertIds + group).to(tl.int64)
    if tl.load(Counts + group) > 0:
        n_tiles = triton.cdiv(N, BLOCK)
        for tile in range(tl.program_id(1), K * n_tiles, tl.num_programs(1)):
            col = tile // n_tiles
            row = tile % n_tiles * BLOCK + tl.arange(0, BLOCK)
            code = tl.load(
                W + expert * (K // 2) * N + (col // 2) * N + row,
                row < N,
                other=0,
            ).to(tl.int32)
            code = (code >> ((col % 2) * 4)) & 15
            scale = tl.load(
                S + expert * (K // 32) * N + (col // 32) * N + row,
                row < N,
                other=127,
            )
            value = _decode_e2m1(code) * _decode_e8m0(scale)
            # All vector accesses are contiguous. Scalar tile division also
            # avoids Ascend's inexact i32 division of offsets above 2**24.
            tl.store(Y + group * K * N + col * N + row, value, row < N)


def expand_active_mxfp4_experts(weight, scales, counts, expert_ids):
    """Expand transposed packed [expert, input/2, output] weights for GMM."""
    _, packed_k, n = weight.shape
    experts = expert_ids.numel()
    k = packed_k * 2
    out = torch.empty((experts, k, n), dtype=torch.bfloat16, device=weight.device)
    _expand_active_experts[(experts, min(256, k * triton.cdiv(n, 2048)))](
        weight,
        scales,
        counts,
        expert_ids,
        out,
        n,
        k,
        2048,
        num_warps=1,
        multibuffer=False,
    )
    return out


@triton.jit
def _mxfp8_embedding(
    W, S, IDs, Y, W0, S0, START, END, ROWS, D: tl.constexpr, TILE: tl.constexpr
):
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        index = tl.load(IDs + row).to(tl.int64)
        col = tl.arange(0, TILE)
        valid = (index >= START) & (index < END)
        local = tl.where(valid, index - START, 0)
        code = tl.load(W + local * W0 + col, valid & (col < D), other=0)
        scale = tl.load(S + local * S0 + col // 32, valid & (col < D), other=127)
        value = _decode_e4m3(code) * _decode_e8m0(scale)
        tl.store(Y + row * D + col, tl.where(valid, value, 0.0), col < D)


def mxfp8_embedding(weight, scales, indices, row_start, row_end):
    width = weight.shape[1]
    out = torch.empty(
        (*indices.shape, width), dtype=torch.bfloat16, device=weight.device
    )
    if out.numel():
        with torch.npu.device(weight.device):
            _mxfp8_embedding[(min(1024, indices.numel()),)](
                weight,
                scales,
                indices,
                out,
                weight.stride(0),
                scales.stride(0),
                row_start,
                row_end,
                indices.numel(),
                width,
                triton.next_power_of_2(width),
            )
    return out
