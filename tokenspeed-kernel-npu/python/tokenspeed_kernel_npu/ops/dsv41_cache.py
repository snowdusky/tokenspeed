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

"""Original V4.1 page-planar cache bytes on Ascend, with software FP8 codecs."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.mxfp import _decode_e4m3, _decode_e8m0

_LAYOUTS = {
    "swa": (512, 32, 512, 528),
    "global": (512, 16, 256, 288),
    "index": (128, 32, 64, 68),
}


@triton.jit
def _e4m3_encode(x):
    a = tl.minimum(tl.abs(x), 448.0)
    exponent = tl.maximum(((a.to(tl.int32, bitcast=True) >> 23) & 255) - 127, -6)
    factor = ((130 - exponent) << 23).to(tl.float32, bitcast=True)
    scaled = a * factor
    low = tl.floor(scaled)
    fraction = scaled - low
    rounded = low.to(tl.int32) + (
        (fraction > 0.5) | ((fraction == 0.5) & ((low.to(tl.int32) & 1) != 0))
    ).to(tl.int32)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 128
    return (((exponent + 6) * 8 + rounded) | sign).to(tl.uint8)


@triton.jit
def _e2m1_encode(x):
    a = tl.minimum(tl.abs(x), 6.0)
    # Nearest, ties to even *code*, including signed zero. V4's codec is not RNE.
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return (code | ((x.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)


@triton.jit
def _e2m1_decode(code):
    a = code & 7
    value = tl.where(
        a < 4,
        a.to(tl.float32) * 0.5,
        tl.where(a == 4, 2.0, tl.where(a == 5, 3.0, tl.where(a == 6, 4.0, 6.0))),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _offset(byte, CR, CB, WIDTH: tl.constexpr):
    return (byte // WIDTH) * CR + (byte % WIDTH) * CB


@triton.jit
def _pack(
    X,
    C,
    Slots,
    X0,
    X1,
    CP,
    CR,
    CB,
    SS,
    CAPACITY,
    ROWS,
    PAGE_ROWS: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    WIDTH: tl.constexpr,
    FORMAT: tl.constexpr,
    SCATTER: tl.constexpr,
):
    tile = tl.program_id(1)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        slot = tl.load(Slots + row * SS).to(tl.int64) if SCATTER else row.to(tl.int64)
        if slot >= 0 and slot < CAPACITY:
            d = tile * 128 + tl.arange(0, 128)
            x = (
                tl.load(X + row * X0 + d * X1)
                .to(tl.float32)
                .reshape(128 // GROUP, GROUP)
            )
            amax = tl.max(tl.abs(x), axis=1)
            if FORMAT == "global":
                scale_byte = _e4m3_encode(tl.div_rn(tl.maximum(amax, 6 * 2.0**-9), 6.0))
                scale = _decode_e4m3(scale_byte)
            else:
                if FORMAT == "index":
                    scaled = tl.maximum(amax, 6 * 2.0**-126) * (1.0 / 6.0)
                else:
                    scaled = tl.maximum(amax, 1.0e-4) * (1.0 / 448.0)
                bits = scaled.to(tl.int32, bitcast=True)
                exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
                scale = (exponent << 23).to(tl.float32, bitcast=True)
                scale_byte = exponent.to(tl.uint8)
            y = tl.div_rn(x, scale[:, None]).reshape(128)
            base = C + (slot // PAGE_ROWS) * CP
            row_offset = (slot % PAGE_ROWS) * VALUES
            if FORMAT == "swa":
                tl.store(base + _offset(row_offset + d, CR, CB, WIDTH), _e4m3_encode(y))
            else:
                lo, hi = tl.split(_e2m1_encode(y).reshape(64, 2))
                tl.store(
                    base
                    + _offset(row_offset + tile * 64 + tl.arange(0, 64), CR, CB, WIDTH),
                    lo | (hi << 4),
                )
            scale_offset = PAGE_ROWS * VALUES + (slot % PAGE_ROWS) * (D // GROUP)
            s = tile * (128 // GROUP) + tl.arange(0, 128 // GROUP)
            tl.store(base + _offset(scale_offset + s, CR, CB, WIDTH), scale_byte)


@triton.jit
def _load_rows(
    C,
    slot,
    d,
    valid,
    CP,
    CR,
    CB,
    PAGE_ROWS: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    WIDTH: tl.constexpr,
    FORMAT: tl.constexpr,
):
    slot = tl.where(valid, slot, 0).to(tl.int64)
    base = C + (slot // PAGE_ROWS) * CP
    data_offset = (slot % PAGE_ROWS) * VALUES + (d if FORMAT == "swa" else d // 2)
    scale_offset = PAGE_ROWS * VALUES + (slot % PAGE_ROWS) * (D // GROUP) + d // GROUP
    code = tl.load(base + _offset(data_offset, CR, CB, WIDTH), valid, other=0).to(
        tl.int32
    )
    scale = tl.load(base + _offset(scale_offset, CR, CB, WIDTH), valid, other=127)
    if FORMAT == "swa":
        value = _decode_e4m3(code) * _decode_e8m0(scale)
    else:
        value = _e2m1_decode((code >> ((d % 2) * 4)) & 15)
        if FORMAT == "global":
            value *= _decode_e4m3(scale)
        else:
            value *= _decode_e8m0(scale)
    return tl.where(valid, value, 0.0)


@triton.jit
def _gather(
    C,
    Slots,
    O,
    CP,
    CR,
    CB,
    CAPACITY,
    ROWS,
    PAGE_ROWS: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    WIDTH: tl.constexpr,
    FORMAT: tl.constexpr,
    GATHER: tl.constexpr,
):
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        slot = tl.load(Slots + row).to(tl.int64) if GATHER else row.to(tl.int64)
        d = tl.arange(0, D)
        value = _load_rows(
            C,
            slot,
            d,
            (slot >= 0) & (slot < CAPACITY),
            CP,
            CR,
            CB,
            PAGE_ROWS,
            D,
            GROUP,
            VALUES,
            WIDTH,
            FORMAT,
        )
        tl.store(O + row * D + d, value)


def _layout(cache_format):
    if cache_format not in _LAYOUTS:
        raise ValueError(f"Ascend cache requires a V4.1 format, got {cache_format}")
    return _LAYOUTS[cache_format]


def cache_scatter(rows, cache, slots, cache_format):
    d, group, values, width = _layout(cache_format)
    if (
        rows.ndim != 2
        or rows.shape[1] != d
        or rows.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Incorrect cache input shape or dtype")
    if cache.ndim != 3 or cache.shape[2] != width or cache.dtype != torch.uint8:
        raise ValueError("Incorrect page-planar cache shape or dtype")
    if cache.device != rows.device or (
        slots is not None
        and (
            slots.device != rows.device
            or slots.shape != rows.shape[:1]
            or slots.dtype not in (torch.int32, torch.int64)
        )
    ):
        raise ValueError("Cache slots must be colocated integer row indices")
    if (
        rows.dtype == torch.bfloat16
        and rows.stride(1) == 1
        and cache.stride(1) == width
        and cache.stride(2) == 1
        and (slots is None or slots.is_contiguous())
    ):
        if cache_format == "swa":
            from tokenspeed_kernel_npu.ops._v41.native_swa_pack import (
                cache_scatter as native_scatter,
            )
        elif cache_format == "global":
            from tokenspeed_kernel_npu.ops._v41.native_global_pack import (
                cache_scatter as native_scatter,
            )
        else:
            from tokenspeed_kernel_npu.ops._v41.native_index_pack import (
                cache_scatter as native_scatter,
            )

        with torch.npu.device(rows.device):
            native_scatter(rows, cache, slots, cache_format)
        return
    if rows.shape[0]:
        # Ascend reduction lowering can overflow UB for broadcast (zero-stride)
        # rows. Normalize input layout; ordinary projected rows need no copy.
        rows = rows.contiguous()
        with torch.npu.device(rows.device):
            _pack[(min(1024, rows.shape[0]), d // 128)](
                rows,
                cache,
                slots,
                *rows.stride(),
                *cache.stride(),
                0 if slots is None else slots.stride(0),
                cache.shape[0] * cache.shape[1],
                rows.shape[0],
                cache.shape[1],
                d,
                group,
                values,
                width,
                cache_format,
                slots is not None,
                enable_fp_fusion=False,
                multibuffer=False,
            )


def cache_gather(cache, slots, cache_format, out):
    d, group, values, width = _layout(cache_format)
    if cache.ndim != 3 or cache.shape[2] != width or cache.dtype != torch.uint8:
        raise ValueError("Incorrect page-planar cache shape or dtype")
    shape = (cache.shape[0], d) if slots is None else (*slots.shape, d)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=cache.device)
    if (
        out.shape != shape
        or out.device != cache.device
        or out.dtype not in (torch.bfloat16, torch.float32)
        or not out.is_contiguous()
    ):
        raise ValueError("Cache output must be colocated contiguous BF16/FP32 rows")
    if slots is not None:
        if slots.device != cache.device or slots.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Cache slots must be colocated integer indices")
        slots = slots.contiguous()
    if out.numel():
        with torch.npu.device(cache.device):
            _gather[(min(1024, out.numel() // d),)](
                cache,
                slots,
                out,
                *cache.stride(),
                cache.shape[0] * cache.shape[1],
                out.numel() // d,
                cache.shape[1],
                d,
                group,
                values,
                width,
                cache_format,
                slots is not None,
            )
    return out


def cache_pack(rows, cache_format, out):
    width = _layout(cache_format)[3]
    if out is None:
        out = torch.empty((rows.shape[0], width), dtype=torch.uint8, device=rows.device)
    if out.shape != (rows.shape[0], width):
        raise ValueError("Incorrect packed row shape")
    cache_scatter(rows, out.unsqueeze(1), None, cache_format)
    return out


def cache_unpack(packed, cache_format, out):
    if packed.ndim != 2:
        raise ValueError("Packed cache must be a row matrix")
    return cache_gather(packed.unsqueeze(1), None, cache_format, out)
