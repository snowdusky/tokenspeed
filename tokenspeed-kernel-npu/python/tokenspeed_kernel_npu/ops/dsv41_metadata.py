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

"""Bound every indirect read before masking on Triton-Ascend.

Invalid lanes use a valid address as well as a mask, so no indirect load
depends on a masked out-of-allocation pointer. Logical slot validity and
output dtypes remain unchanged.
"""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _slot(Table, pos, req, TR, TC, TS0, TS1, PAGES: tl.constexpr):
    col = pos // 2
    valid = (pos >= 0) & (req >= 0) & (req < TR) & (col >= 0) & (col < TC)
    if (TR > 0) & (TC > 0):
        address = tl.where(valid, req * TS0 + col * TS1, 0)
        page = tl.load(Table + address).to(tl.int64)
    else:
        page = tl.full(pos.shape, -1, tl.int64)
    valid &= (page >= 1) & (page < PAGES)
    return tl.where(valid, page * 2 + pos % 2, -1)


@triton.jit(do_not_specialize=["N", "TR", "TC", "TS0", "TS1"])
def _metadata(
    P,
    R,
    Table,
    Active,
    Pair,
    PairR,
    Previous,
    Read,
    Write,
    N,
    P0: tl.constexpr,
    R0: tl.constexpr,
    TR,
    TC,
    TS0,
    TS1,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(P + row * P0, row < N, other=-1).to(tl.int64)
    req = tl.load(R + row * R0, row < N, other=-1).to(tl.int64)
    active = (pos >= 0) & (pos % 2 == 1)
    pair = tl.where(active, pos - 1, -1)
    pair_req = tl.where(active, req, -1)
    # Clamp both sides: masked tail lanes and row zero must not form
    # out-of-allocation pointers even if a compiler speculates the read.
    previous_row = tl.minimum(tl.maximum(row - 1, 0), N - 1)
    prior_pos = tl.load(P + previous_row * P0)
    prior_req = tl.load(R + previous_row * R0)
    previous = tl.where(
        active & (req >= 0) & (row > 0) & (prior_pos == pair) & (prior_req == req),
        row - 1,
        -1,
    )
    read = _slot(Table, pair, pair_req, TR, TC, TS0, TS1, PAGES)
    write = _slot(Table, pos, req, TR, TC, TS0, TS1, PAGES)
    tl.store(Active + row, active, row < N)
    tl.store(Pair + row, pair, row < N)
    tl.store(PairR + row, pair_req, row < N)
    tl.store(Previous + row, previous, row < N)
    tl.store(Read + row, read, row < N)
    tl.store(Write + row, write, row < N)


def compressor_metadata(
    positions,
    requests,
    table,
    pages,
    active,
    pair_positions,
    pair_requests,
    previous,
    read_slots,
    write_slots,
):
    n = positions.numel()
    out = active, pair_positions, pair_requests, previous, read_slots, write_slots
    if positions.shape != (n,) or requests.shape != (n,) or table.ndim != 2:
        raise ValueError("Expected [N] positions/requests and a 2-D page table")
    if any(
        x.dtype not in (torch.int32, torch.int64)
        for x in (positions, requests, table, *out[1:])
    ):
        raise ValueError("Metadata and slot outputs require int32 or int64")
    if active.dtype != torch.bool or any(
        x.shape != (n,) or not x.is_contiguous() for x in out
    ):
        raise ValueError("Expected contiguous [N] outputs and bool active")
    if positions.device.type != "npu" or any(
        x.device != positions.device for x in (requests, table, *out)
    ):
        raise ValueError("All tensors must share an NPU")
    if n:
        _metadata[((n + 255) // 256,)](
            positions,
            requests,
            table,
            *out,
            n,
            positions.stride(0),
            requests.stride(0),
            *table.shape,
            *table.stride(),
            pages,
            256,
            num_warps=1,
            multibuffer=False,
        )
