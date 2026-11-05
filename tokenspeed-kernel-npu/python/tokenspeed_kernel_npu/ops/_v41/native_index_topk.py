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

"""Packed Index scoring and selection under the common operator contract."""

from tokenspeed_kernel_npu.ops._v41.native_index_query import quantize_dequantize
from tokenspeed_kernel_npu.ops._v41.native_index_reduction import scores
from tokenspeed_kernel_npu.ops._v41.native_index_select import (
    block_scores,
)
from tokenspeed_kernel_npu.ops._v41.native_index_select import select as _select


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
    if table.shape[1] * 64 > 2**24:
        raise ValueError("Index positions exceed exact FP32 ID ordering range")
    row_out, row_lens, block_out, block_lens = out
    if not candidate_topk:
        block_lens.zero_()
    width = table.shape[1] * 64 if candidates is None else candidates.shape[1] * 8
    if not q.shape[0] or not width:
        row_out.fill_(-1)
        row_lens.zero_()
        block_out.fill_(-1)
        block_lens.zero_()
        return out
    # Chunking bounds score output memory only, preserving arithmetic.
    chunk = min(query_chunk_size, max(1, (64 * 1024 * 1024) // (width * 4)))
    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        query = q[start:stop]
        query = quantize_dequantize(query.flatten(0, 1)).reshape_as(query)
        v = visible[start:stop]
        c = None if candidates is None else candidates[start:stop]
        score = scores(
            query, weights[start:stop], cache, table[start:stop], v, c, width, shards
        )
        _select(score, topk, row_out[start:stop], row_lens[start:stop], c)
        if candidate_topk:
            blocks = block_scores(score, v)
            _select(
                blocks,
                candidate_topk,
                block_out[start:stop],
                block_lens[start:stop],
                None,
            )
    return out
