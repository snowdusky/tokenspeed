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

"""Cube index scoring and exact selection without unsupported uint64 sorts."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_cache import cache_gather, cache_pack, cache_unpack


@triton.jit
def _select_all(
    Table,
    Visible,
    Out,
    Lengths,
    T0,
    T1,
    L0,
    TC,
    PAGES,
    WIDTH,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    ids = tl.arange(0, BLOCK)
    position = ids * STRIDE
    visible = tl.load(Visible + row * L0)
    page = tl.load(
        Table + row * T0 + (position // 64) * T1,
        (ids < WIDTH) & (position // 64 < TC),
        other=-1,
    )
    valid = (ids < WIDTH) & (position < visible) & (page > 0) & (page < PAGES)
    prefix = tl.cumsum(valid.to(tl.int32))
    count = tl.sum(valid.to(tl.int32), 0)
    # IDs are already ordered. Compact valid IDs instead of sorting a large
    # sentinel-filled vector, which exceeds Ascend UB at candidate width 2048.
    tl.store(Out + row * WIDTH + prefix - 1, ids, valid)
    tl.store(Out + row * WIDTH + ids, -1, (ids >= count) & (ids < WIDTH))
    tl.store(Lengths + row, count)


def index_select_all(page_table, visible_lens, pages, candidate_block_size, out):
    for destination, lengths, stride in (
        (out[0], out[1], 1),
        (out[2], out[3], candidate_block_size),
    ):
        if destination.shape[1] == 0:
            lengths.zero_()
        elif page_table.shape[0]:
            _select_all[(page_table.shape[0],)](
                page_table,
                visible_lens,
                destination,
                lengths,
                *page_table.stride(),
                visible_lens.stride(0),
                page_table.shape[1],
                pages,
                destination.shape[1],
                stride,
                triton.next_power_of_2(destination.shape[1]),
                num_warps=1,
                multibuffer=False,
            )
    return out


def _scores(q, weights, cache, table, visible, candidates, width, shards):
    columns = torch.arange(width, device=q.device)
    if candidates is None:
        logical = columns[None, :].expand(q.shape[0], -1)
    else:
        blocks = candidates[:, columns // 8]
        logical = (blocks.long() * 8 + columns % 8).masked_fill(blocks < 0, -1)
    pages = table.gather(1, (logical // 64).clamp(0, table.shape[1] - 1))
    valid = (
        (logical >= 0)
        & (logical < visible.clamp(0, table.shape[1] * 64)[:, None])
        & (pages >= 0)
        & (pages < cache.shape[0])
    )
    slots = (pages.long() * 64 + logical % 64).masked_fill(~valid, -1)
    keys = cache_gather(cache, slots, "index", None)
    # CANN BMM keeps the portable scorer's BF16 dot boundary; each shard's
    # weighted product and head sum still round to the weights' dtype.
    products = torch.bmm(q, keys.transpose(1, 2)).relu_() * weights[:, :, None]
    scores = torch.zeros(slots.shape, dtype=torch.float32, device=q.device)
    for partial in products.chunk(shards, dim=1):
        scores.add_(partial.sum(dim=1).float())
    return scores.to(weights.dtype).float().masked_fill(~valid, -torch.inf)


def _select(scores, k, output, lengths, candidates):
    count = min(k, scores.shape[1])
    if count == 0:
        output.fill_(-1)
        lengths.zero_()
        return
    values, ids = scores.topk(count, dim=1, sorted=False)
    ids = ids.to(torch.int32)
    if candidates is not None:
        blocks = candidates.gather(1, (ids // 8).long()).int()
        ids = blocks * 8 + ids % 8
    live = values > -torch.inf
    lengths.copy_(live.sum(1))
    ids = ids.masked_fill(~live, torch.iinfo(torch.int32).max).sort(dim=1).values
    ids.masked_fill_(ids == torch.iinfo(torch.int32).max, -1)
    output[:, :count].copy_(ids)
    output[:, count:].fill_(-1)


def index_topk(
    q,
    weights,
    cache,
    table,
    visible,
    candidates,
    topk,
    candidate_topk,
    query_chunk_size,
    shards,
    out,
):
    row_out, row_lens, block_out, block_lens = out
    width = table.shape[1] * 64 if candidates is None else candidates.shape[1] * 8
    if not q.shape[0] or not width:
        row_out.fill_(-1)
        row_lens.zero_()
        block_out.fill_(-1)
        block_lens.zero_()
        return out
    # Bound gathered BF16 keys and score/product scratch, independently of
    # request capacity. One query may exceed this budget for very long history.
    bytes_per_query = width * (128 * 2 + q.shape[1] * (2 + weights.element_size()) + 24)
    chunk = min(query_chunk_size, max(1, (32 * 1024 * 1024) // bytes_per_query))
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        query = q[start:end]
        query = cache_unpack(
            cache_pack(query.reshape(-1, 128), "index", None), "index", None
        ).reshape_as(query)
        w, t, v = weights[start:end], table[start:end], visible[start:end]
        c = None if candidates is None else candidates[start:end]
        scores = _scores(query, w, cache, t, v, c, width, shards)
        _select(scores, topk, row_out[start:end], row_lens[start:end], c)
        if candidate_topk:
            blocks = scores.unflatten(1, (-1, 8)).amax(2)
            latest = (v.clamp(0, width) - 1) // 8
            ids = torch.arange(blocks.shape[1], device=q.device)
            blocks = blocks.masked_fill(
                (ids[None, :] == latest[:, None]) & (blocks > -torch.inf), torch.inf
            )
            _select(
                blocks,
                candidate_topk,
                block_out[start:end],
                block_lens[start:end],
                None,
            )
    return out
