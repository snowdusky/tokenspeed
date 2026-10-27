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

import ctypes
from functools import lru_cache
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops._v41.attention_merge import grouped_merge
from tokenspeed_kernel_npu.ops._v41.attention_merge_rope import (
    merge_rope,
    merge_rope_scalar,
)


@lru_cache(maxsize=16)
def make_tables(device):
    codes = torch.arange(256, dtype=torch.int32)
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def e2(x):
        return magnitudes[x & 7] * torch.where((x & 8) == 0, 1.0, -1.0)

    pairs = (
        torch.stack([e2(codes & 15), e2(codes >> 4)], -1)
        .half()
        .view(torch.int32)
        .flatten()
    )
    e4 = codes.byte().view(torch.float8_e4m3fn).float().view(torch.int32)
    e8 = torch.pow(2.0, codes.float() - 127).view(torch.int32)
    return torch.cat([pairs, e4, e8]).int().to(device)


@triton.jit
def merge(
    P,
    S,
    Sink,
    O,
    L,
    heads,
    parts: tl.constexpr,
    PS: tl.constexpr,
    HEAD_CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    p = tl.arange(0, PS)
    d = tl.arange(0, 512)
    sums = tl.load(
        S + (row * parts + p) * (2 * HEAD_CAPACITY) + head, p < parts, other=0
    )
    maxima = tl.load(
        S + (row * parts + p) * (2 * HEAD_CAPACITY) + HEAD_CAPACITY + head,
        p < parts,
        other=-float("inf"),
    )
    maxima = tl.where(sums > 0, maxima, -float("inf"))
    sink = tl.load(Sink + head)
    m = tl.maximum(tl.max(maxima, 0), sink)
    m = tl.where(m == -float("inf"), 0.0, m)
    w = tl.where(sums > 0, tl.exp(maxima - m), 0.0)
    den = tl.sum(sums * w, 0) + tl.exp(sink - m)
    value = tl.load(
        P + ((row * parts + p[:, None]) * HEAD_CAPACITY + head) * 512 + d[None, :],
        p[:, None] < parts,
        other=0.0,
    )
    y = tl.sum(value * w[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.store(O + (row * heads + head) * 512 + d, y)
    tl.store(L + row * heads + head, tl.where(den > 0, tl.log(den) + m, -float("inf")))


class Native:
    """Standalone 910B decode component, with caller-owned graph-stable inputs.

    Cache rows are token-interleaved: BF16[512], or SWA bytes[528] and
    global bytes[288]. Each query selects 128 SWA + 512 global slots.
    Unsupported layouts and shapes are rejected before launch.
    """

    def __init__(
        self, q, sw, gl, ss, gs, sl, ll, sink, packed, tile, *, paged, scale, row_owned
    ):
        self.row_owned = row_owned
        if q.ndim != 3 or q.shape[2] != 512 or not 1 <= q.shape[1] <= 64:
            raise ValueError("query must be [rows, heads <= 64, 512]")
        if tile not in (64, 128) or q.dtype != torch.bfloat16:
            raise ValueError("BF16 queries and tiles 64/128 required")
        rows, heads = q.shape[:2]
        for slots in (ss, gs):
            if slots is not None and (
                slots.ndim != 2
                or slots.shape[0] != rows
                or slots.dtype not in (torch.int32, torch.int64)
                or any(s < 0 for s in slots.stride())
            ):
                raise ValueError("invalid live selection table")
        for lengths in (sl, ll):
            if lengths is not None and (
                lengths.shape != (rows,)
                or lengths.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("invalid live lengths")
        if sink.shape != (heads,) or sink.dtype != torch.float32:
            raise ValueError("FP32 head sinks required")
        for cache, width in (
            (sw, 528 if packed else 512),
            (gl, 288 if packed else 512),
        ):
            if (
                cache.ndim != (3 if paged else 2)
                or cache.shape[-1] != width
                or (
                    paged
                    and (
                        cache.shape[1] != 64
                        or cache.stride(1) != width
                        or cache.stride(2) != 1
                    )
                )
                or cache.dtype != (torch.uint8 if packed else torch.bfloat16)
            ):
                raise ValueError(
                    "cache does not match the explicitly selected storage format"
                )
        for x in (q, sw, gl, ss, gs, sl, ll, sink):
            if x is None:
                continue
            if (
                x.device != q.device
                or x.device.type != "npu"
                or (x is q or x is sink)
                and not x.is_contiguous()
            ):
                raise ValueError("inputs must be contiguous and reside on the same NPU")
        if q.device.index != torch.npu.current_device():
            raise ValueError(
                "select the query device before constructing the component"
            )
        self.head_capacity = 16 if heads <= 16 else 64
        library = f"libattention_codec_bits_h{self.head_capacity}.so"
        self.lib = ctypes.CDLL(str(Path(__file__).with_name(library)))
        self.f = self.lib.launch_attention_codec_bits
        self.f.restype = None
        self.f.argtypes = (
            [ctypes.c_void_p] * 11
            + [ctypes.c_int] * 6
            + [ctypes.c_int64] * 2
            + [ctypes.c_int] * 2
            + [ctypes.c_float]
            + [ctypes.c_int] * 4
            + [ctypes.c_int64] * 6
            + [ctypes.c_int, ctypes.c_void_p]
        )
        self.q = q
        self.sw = sw
        self.gl = gl
        self.ss, self.gs, self.sl, self.ll = ss, gs, sl, ll
        self.sink = sink
        self.packed = packed
        self.paged = paged
        self.scale = scale
        self.swa_selected = max(tile, triton.cdiv(ss.shape[1], tile) * tile)
        self.global_selected = (
            0 if gs is None else triton.cdiv(gs.shape[1], tile) * tile
        )
        self.selected = self.swa_selected + self.global_selected
        self.tile = tile
        self.tables = make_tables(q.device)
        self.rows, self.heads = q.shape[:2]
        self.parts = self.selected // tile
        core_bytes = tile * 512 * 2 + self.head_capacity * tile * 6 + tile * 4
        self.ws = torch.empty(24 * 2 * core_bytes, dtype=torch.uint8, device=q.device)
        self.partial = torch.empty(
            self.rows * self.parts * self.head_capacity * 512,
            dtype=torch.float32,
            device=q.device,
        )
        self.stats = torch.empty(
            self.rows * self.parts * 2 * self.head_capacity,
            dtype=torch.float32,
            device=q.device,
        )
        self.out = torch.empty_like(q)
        self.lse = torch.empty(q.shape[:2], device=q.device)

    def run(self, *, rotary=None):
        if rotary is not None:
            positions, cache = rotary
            if (
                positions.shape != (self.rows,)
                or not positions.is_contiguous()
                or positions.dtype not in (torch.int32, torch.int64)
                or cache.ndim != 2
                or cache.shape[1] != 64
                or cache.dtype != torch.float32
                or not cache.is_contiguous()
                or positions.device != self.q.device
                or cache.device != self.q.device
            ):
                raise ValueError(
                    "rotary requires live integer positions and FP32 [max_pos,64] table"
                )
        tensors = (
            self.q,
            self.sw,
            self.gl,
            self.ss,
            self.gs,
            self.sl,
            self.ll,
            self.tables,
            self.ws,
            self.partial,
            self.stats,
        )

        def flags(ids, lengths):
            return (
                int(ids is not None and ids.dtype == torch.int64)
                + 2 * int(lengths is not None and lengths.dtype == torch.int64)
                + 4 * int(lengths is not None)
            )

        self.f(
            *(x.data_ptr() if x is not None else 0 for x in tensors),
            self.rows,
            self.heads,
            self.selected,
            self.swa_selected,
            self.sw.shape[0] * (64 if self.paged else 1),
            self.gl.shape[0] * (64 if self.paged else 1),
            self.sw.stride(0) * self.sw.element_size(),
            self.gl.stride(0) * self.gl.element_size(),
            self.packed,
            self.tile,
            self.scale,
            self.paged,
            self.ss.shape[1],
            0 if self.gs is None else self.gs.shape[1],
            flags(self.ss, self.sl) + 8 * flags(self.gs, self.ll),
            *self.ss.stride(),
            *(self.gs.stride() if self.gs is not None else (0, 0)),
            0 if self.sl is None else self.sl.stride(0),
            0 if self.ll is None else self.ll.stride(0),
            self.row_owned,
            torch.npu.current_stream().npu_stream,
        )
        if rotary is not None:
            positions, cache = rotary
            if (self.parts == 5 and self.rows >= 4) or self.rows >= 16:
                group_heads = 4 if self.parts == 5 else 2
                merge_rope[(self.rows, triton.cdiv(self.heads, group_heads))](
                    self.partial,
                    self.stats,
                    self.sink,
                    self.out,
                    self.lse,
                    positions,
                    cache,
                    self.heads,
                    self.parts,
                    triton.next_power_of_2(self.parts),
                    group_heads,
                    HEAD_CAPACITY=self.head_capacity,
                    enable_fp_fusion=False,
                )
            else:
                merge_rope_scalar[(self.rows, self.heads)](
                    self.partial,
                    self.stats,
                    self.sink,
                    self.out,
                    self.lse,
                    positions,
                    cache,
                    self.heads,
                    self.parts,
                    triton.next_power_of_2(self.parts),
                    HEAD_CAPACITY=self.head_capacity,
                    enable_fp_fusion=False,
                )
            return self.out
        if (self.parts == 5 and self.rows >= 4) or self.rows >= 16:
            group_heads = 4 if self.parts == 5 else 2
            grouped_merge[(self.rows, triton.cdiv(self.heads, group_heads))](
                self.partial,
                self.stats,
                self.sink,
                self.out,
                self.lse,
                self.heads,
                self.parts,
                triton.next_power_of_2(self.parts),
                group_heads,
                HEAD_CAPACITY=self.head_capacity,
            )
        else:
            merge[(self.rows, self.heads)](
                self.partial,
                self.stats,
                self.sink,
                self.out,
                self.lse,
                self.heads,
                self.parts,
                triton.next_power_of_2(self.parts),
                HEAD_CAPACITY=self.head_capacity,
            )
        return self.out
