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

"""DSpark window rows with the reference FP8 codec on 910 devices."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_cache import _e4m3_encode
from tokenspeed_kernel_npu.ops.mxfp import _decode_e4m3


@triton.jit
def _row(
    X,
    NORM,
    POS,
    CS,
    token,
    X0,
    PS,
    CS0,
    EPS: tl.constexpr,
    ROTARY: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    d = tl.arange(0, 512)
    x = tl.load(X + token * X0 + d).to(tl.float32)
    if HAS_NORM:
        weight = tl.load(NORM + d).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 512) + EPS)
        x = (weight * (x * inv)).to(tl.bfloat16).to(tl.float32)
    if HAS_ROPE:
        position = tl.maximum(tl.load(POS + token * PS).to(tl.int64), 0)
        partner = tl.gather(x, d ^ 1, axis=0)
        tail = d >= 512 - ROTARY
        pair = (d - (512 - ROTARY)) // 2
        cosine = tl.load(CS + position * CS0 + pair, tail, 1.0)
        sine = tl.load(CS + position * CS0 + ROTARY // 2 + pair, tail, 0.0)
        first = x * cosine
        second = partner * sine
        rotated = tl.where((d & 1) != 0, first + second, first - second)
        x = tl.where(tail, rotated, x).to(tl.bfloat16).to(tl.float32)
    values = x.reshape(16, 32)
    unrounded = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4) * (1.0 / 448.0)
    bits = unrounded.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    # Software encode/decode avoids unsupported 910 FP8 tensor conversions.
    codes = _e4m3_encode(tl.div_rn(values, scale[:, None]))
    return (_decode_e4m3(codes) * scale[:, None]).reshape(512).to(tl.bfloat16)


@triton.jit(
    do_not_specialize=[
        "X0",
        "PS",
        "CS0",
        "SS",
        "WP",
        "WR",
        "OS",
        "TOKENS",
        "PAGES",
        "PAGE_ROWS",
    ]
)
def _rows(
    X,
    NORM,
    POS,
    CS,
    SLOTS,
    WINDOW,
    OUT,
    X0,
    PS,
    CS0,
    SS,
    WP,
    WR,
    OS,
    TOKENS,
    PAGES,
    PAGE_ROWS,
    EPS: tl.constexpr,
    ROTARY: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_OUT: tl.constexpr,
):
    d = tl.arange(0, 512)
    for token in range(tl.program_id(0), TOKENS, tl.num_programs(0)):
        value = _row(
            X, NORM, POS, CS, token, X0, PS, CS0, EPS, ROTARY, HAS_NORM, HAS_ROPE
        )
        if HAS_OUT:
            tl.store(OUT + token * OS + d, value)
        if HAS_WINDOW:
            slot = tl.load(SLOTS + token * SS).to(tl.int64)
            valid = (slot >= 0) & (slot < PAGES * PAGE_ROWS)
            page = tl.maximum(slot, 0) // PAGE_ROWS
            row = tl.maximum(slot, 0) % PAGE_ROWS
            tl.store(WINDOW + page * WP + row * WR + d, value, valid)


@triton.jit(
    do_not_specialize=[
        "X0",
        "PS",
        "CS0",
        "WP",
        "WR",
        "P0",
        "P1",
        "R0",
        "R1",
        "BATCH",
        "HISTORY",
        "BLOCK",
    ]
)
def _workspace(
    X,
    NORM,
    POS,
    CS,
    WINDOW,
    PAGE,
    ROW,
    OUT,
    X0,
    PS,
    CS0,
    WP,
    WR,
    P0,
    P1,
    R0,
    R1,
    BATCH,
    HISTORY,
    BLOCK,
    EPS: tl.constexpr,
    ROTARY: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    d = tl.arange(0, 512)
    width = HISTORY + BLOCK
    for i in range(tl.program_id(0), BATCH * width, tl.num_programs(0)):
        request = i // width
        column = i % width
        if column < HISTORY:
            page = tl.load(PAGE + request * P0 + column * P1).to(tl.int64)
            row = tl.load(ROW + request * R0 + column * R1).to(tl.int64)
            value = tl.load(WINDOW + page * WP + row * WR + d)
        else:
            token = request * BLOCK + column - HISTORY
            value = _row(
                X, NORM, POS, CS, token, X0, PS, CS0, EPS, ROTARY, HAS_NORM, HAS_ROPE
            )
        tl.store(OUT + i * 512 + d, value)


def _validate(values, norm_weight, norm_eps, positions, cos_sin_cache, window):
    if (
        values.ndim != 2
        or values.shape[1] != 512
        or values.stride(1) != 1
        or values.dtype != torch.bfloat16
    ):
        raise ValueError(
            "DSpark values must be BF16 [tokens,512] with unit column stride"
        )
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("DSpark normalization weight and epsilon come together")
    if norm_weight is not None and (
        norm_weight.shape != (512,)
        or norm_weight.dtype != torch.bfloat16
        or not norm_weight.is_contiguous()
        or norm_eps <= 0
    ):
        raise ValueError(
            "DSpark normalization requires contiguous BF16 [512] weights and positive epsilon"
        )
    if (positions is None) != (cos_sin_cache is None):
        raise ValueError("DSpark positions and RoPE table come together")
    if positions is not None:
        _integers(positions, (values.shape[0],), values.device)
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.dtype != torch.float32
            or cos_sin_cache.stride(1) != 1
            or cos_sin_cache.shape[1] % 2
            or not 2 <= cos_sin_cache.shape[1] <= 512
        ):
            raise ValueError(
                "DSpark RoPE table must be FP32 [positions,even rotary_dim]"
            )
    if window is not None and (
        window.ndim != 3
        or window.shape[2] != 512
        or window.dtype != torch.bfloat16
        or window.stride(2) != 1
    ):
        raise ValueError("DSpark window must be BF16 [pages,page_rows,512]")
    for tensor in (norm_weight, positions, cos_sin_cache, window):
        if tensor is not None and tensor.device != values.device:
            raise ValueError("DSpark inputs must share one device")


def _integers(tensor, shape, device):
    if (
        tensor.shape != shape
        or tensor.dtype not in (torch.int32, torch.int64)
        or tensor.device != device
    ):
        raise ValueError(
            "DSpark addressing tensor has incorrect shape, dtype or device"
        )


def rows(values, norm_weight, norm_eps, positions, cos_sin_cache, window, slots, out):
    """Apply norm, RoPE and group32 QDQ, writing requested destinations once."""
    _validate(values, norm_weight, norm_eps, positions, cos_sin_cache, window)
    tokens = values.shape[0]
    if (window is None) != (slots is None):
        raise ValueError("DSpark window and slots come together")
    if slots is not None:
        _integers(slots, (tokens,), values.device)
    if out is not None and (
        out.shape != values.shape
        or out.dtype != values.dtype
        or out.device != values.device
        or out.stride(1) != 1
    ):
        raise ValueError("DSpark output must match values with unit column stride")
    if window is None and out is None:
        raise ValueError("DSpark rows require a window or output destination")
    if tokens:
        _rows[(min(tokens, 48),)](
            values,
            norm_weight,
            positions,
            cos_sin_cache,
            slots,
            window,
            out,
            values.stride(0),
            positions.stride(0) if positions is not None else 0,
            cos_sin_cache.stride(0) if cos_sin_cache is not None else 0,
            slots.stride(0) if slots is not None else 0,
            window.stride(0) if window is not None else 0,
            window.stride(1) if window is not None else 0,
            out.stride(0) if out is not None else 0,
            tokens,
            window.shape[0] if window is not None else 0,
            window.shape[1] if window is not None else 1,
            float(norm_eps) if norm_eps is not None else 0.0,
            cos_sin_cache.shape[1] if cos_sin_cache is not None else 0,
            norm_weight is not None,
            positions is not None,
            window is not None,
            out is not None,
            enable_fp_fusion=False,
        )


def workspace(
    values,
    norm_weight,
    norm_eps,
    positions,
    cos_sin_cache,
    window,
    page,
    row,
    block_size,
):
    """Gather history and encode proposal rows into the attention workspace."""
    _validate(values, norm_weight, norm_eps, positions, cos_sin_cache, window)
    if window is None or page.ndim != 2 or block_size < 1:
        raise ValueError(
            "DSpark workspace requires a window, rank-2 pages and a positive block size"
        )
    batch, history = page.shape
    if values.shape[0] != batch * block_size:
        raise ValueError("DSpark values must cover batch * block_size rows")
    _integers(page, (batch, history), values.device)
    _integers(row, (batch, history), values.device)
    out = values.new_empty((batch, history + block_size, 512))
    if batch:
        _workspace[(min(batch * (history + block_size), 48),)](
            values,
            norm_weight,
            positions,
            cos_sin_cache,
            window,
            page,
            row,
            out,
            values.stride(0),
            positions.stride(0) if positions is not None else 0,
            cos_sin_cache.stride(0) if cos_sin_cache is not None else 0,
            window.stride(0),
            window.stride(1),
            page.stride(0),
            page.stride(1),
            row.stride(0),
            row.stride(1),
            batch,
            history,
            block_size,
            float(norm_eps) if norm_eps is not None else 0.0,
            cos_sin_cache.shape[1] if cos_sin_cache is not None else 0,
            norm_weight is not None,
            positions is not None,
            enable_fp_fusion=False,
        )
    return out
