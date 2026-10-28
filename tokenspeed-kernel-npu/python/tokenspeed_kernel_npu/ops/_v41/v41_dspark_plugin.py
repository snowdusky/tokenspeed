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

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _integers,
    _output,
    _rotate_interleaved,
    _same_device,
    _ue8m0_exponent,
    triton_dsv41_dspark_anchors,
    triton_dsv41_dspark_block,
)
from tokenspeed_kernel.ops.layernorm.triton import reference_rmsnorm_row
from tokenspeed_kernel_npu.ops.dsv41_cache import _e4m3_encode
from tokenspeed_kernel_npu.ops.mxfp import _decode_e4m3


@triton.jit
def _dspark_rows_kernel(
    X,
    W,
    POS,
    CS,
    SLOTS,
    WINDOW,
    OUT,
    X0,
    POS0,
    CS0,
    S0,
    W0,
    W1,
    O0,
    PAGES,
    P: tl.constexpr,
    MAX_POS: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_OUT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 512)
    x = tl.load(X + token * X0 + d).to(tl.float32)
    if HAS_NORM:
        weight = tl.load(W + d).to(tl.float32)
        # The reference rounds the normalized row to BF16 before rotating it.
        x = reference_rmsnorm_row(x, weight, EPS, 512).to(tl.bfloat16).to(tl.float32)
    if HAS_ROPE:
        position = tl.maximum(tl.load(POS + token * POS0).to(tl.int64), 0)
        valid = position < MAX_POS
        tl.device_assert(valid, "DSpark RoPE position is outside the cosine/sine table")
        partner = tl.gather(x, d ^ 1, axis=0)
        tail = d >= 512 - R
        pair = (d - (512 - R)) // 2
        cosine = tl.load(CS + position * CS0 + pair, tail & valid, 1.0)
        sine = tl.load(CS + position * CS0 + R // 2 + pair, tail & valid, 0.0)
        rotated = _rotate_interleaved(x, partner, cosine, sine, (d & 1) != 0)
        x = tl.where(tail, rotated.to(tl.bfloat16), x.to(tl.bfloat16)).to(tl.float32)
    # All-channel 1x32 E4M3/E8M0 round trip, the reference SWA row codec.
    values = tl.reshape(x, (16, 32))
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
    exponent = _ue8m0_exponent(amax * (1.0 / 448.0))
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = _e4m3_encode(tl.div_rn(values, scale[:, None]))
    restored = tl.reshape(_decode_e4m3(quantized) * scale[:, None], (512,)).to(
        tl.bfloat16
    )
    if HAS_OUT:
        tl.store(OUT + token * O0 + d, restored)
    if HAS_WINDOW:
        slot = tl.load(SLOTS + token * S0).to(tl.int64)
        live = (slot >= 0) & (slot < PAGES * P)
        page = tl.maximum(slot, 0) // P
        row = tl.maximum(slot, 0) % P
        tl.store(WINDOW + page * W0 + row * W1 + d, restored, live)


def triton_dsv41_dspark_rows(
    values, norm_weight, norm_eps, positions, cos_sin_cache, window, slots, out
):
    if values.ndim != 2 or values.shape[1] != 512 or values.stride(1) != 1:
        raise ValueError(
            "DSpark rows must be BF16 [tokens,512] with unit column stride"
        )
    tokens = values.shape[0]
    has_norm = norm_weight is not None
    if has_norm != (norm_eps is not None):
        raise ValueError("DSpark kv_norm weight and epsilon come together")
    if has_norm and (norm_weight.shape != (512,) or not norm_weight.is_contiguous()):
        raise ValueError("DSpark kv_norm weight must be a contiguous [512] vector")
    has_rope = cos_sin_cache is not None
    if has_rope:
        if positions is None:
            raise ValueError("DSpark RoPE needs the rows' positions")
        _integers(positions, (tokens,), "positions")
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.dtype != torch.float32
            or cos_sin_cache.shape[1] % 2
            or not 2 <= cos_sin_cache.shape[1] <= 512
            or cos_sin_cache.stride(1) != 1
        ):
            raise ValueError(
                "DSpark RoPE table must be FP32 [positions,even rotary_dim]"
            )
    elif positions is not None:
        raise ValueError("DSpark rows without a RoPE table take no positions")
    has_window = window is not None
    if has_window:
        if slots is None:
            raise ValueError("DSpark window writes need the rows' slots")
        _integers(slots, (tokens,), "slots")
        if (
            window.ndim != 3
            or window.dtype != torch.bfloat16
            or window.shape[2] != 512
            or window.stride(2) != 1
        ):
            raise ValueError("DSpark window must be a BF16 [pages,rows,512] field")
    elif slots is not None:
        raise ValueError("DSpark rows without a window take no slots")
    if out is not None:
        _output(out, (tokens, 512), torch.bfloat16, values.device)
    if not (has_window or out is not None):
        raise ValueError("DSpark rows need a window and/or an out destination")
    _same_device(values, (norm_weight, positions, cos_sin_cache, window, slots, out))
    if tokens == 0:
        return
    _dspark_rows_kernel[(tokens,)](
        values,
        norm_weight if has_norm else values,
        positions if has_rope else values,
        cos_sin_cache if has_rope else values,
        slots if has_window else values,
        window if has_window else values,
        out if out is not None else values,
        values.stride(0),
        positions.stride(0) if has_rope else 0,
        cos_sin_cache.stride(0) if has_rope else 0,
        slots.stride(0) if has_window else 0,
        window.stride(0) if has_window else 0,
        window.stride(1) if has_window else 0,
        out.stride(0) if out is not None else 0,
        window.shape[0] if has_window else 0,
        window.shape[1] if has_window else 1,
        cos_sin_cache.shape[0] if has_rope else 1,
        cos_sin_cache.shape[1] if has_rope else 2,
        float(norm_eps) if has_norm else 0.0,
        has_norm,
        has_rope,
        has_window,
        out is not None,
        num_warps=4,
        enable_fp_fusion=False,
        debug=False,
        multibuffer=False,
    )


def register():
    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures

    for name, fn, dtypes in (
        ("dspark_rows", triton_dsv41_dspark_rows, {torch.bfloat16}),
        ("dspark_anchors", triton_dsv41_dspark_anchors, {torch.int32, torch.int64}),
        ("dspark_block", triton_dsv41_dspark_block, {torch.int32, torch.int64}),
    ):
        register_kernel(
            "attention",
            "dsv41_" + name,
            name="ascend_native_v41_" + name,
            solution="ascend_native",
            capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
            signatures=format_signatures("x", "dense", dtypes),
            priority=Priority.SPECIALIZED,
        )(fn)
