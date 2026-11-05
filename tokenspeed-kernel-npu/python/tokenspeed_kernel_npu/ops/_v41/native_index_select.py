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

"""Exact Index ID ordering using the device's FP32 sort implementation.

The caller limits logical positions to at most 2**24, so integer IDs are
represented exactly. Score ordering and TopK tie choices are unchanged.
"""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["K", "V0", "I0", "C0", "C1", "L0"])
def _prepare(
    V,
    I,
    C,
    Labels,
    Lengths,
    K,
    V0,
    I0,
    C0,
    C1,
    L0,
    SPARSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(V + row * V0 + col, col < K, other=-float("inf"))
    index = tl.load(I + row * I0 + col, col < K, other=0).to(tl.int32)
    valid = (col < K) & (value > -float("inf"))
    if SPARSE:
        block = tl.load(C + row * C0 + (index // 8) * C1, valid, other=0).to(tl.int32)
        index = block * 8 + index % 8
    tl.store(
        Labels + row * K + col,
        tl.where(valid, index.to(tl.float32), float("inf")),
        col < K,
    )
    tl.store(Lengths + row * L0, tl.sum(valid.to(tl.int32), 0))


def select(scores, k, output, lengths, candidates):
    count = min(k, scores.shape[1])
    rows = scores.shape[0]
    if count == 0:
        output.fill_(-1)
        lengths.zero_()
        return
    if not rows:
        return
    values, indices = scores.topk(count, dim=1, sorted=False)
    if candidates is None:
        labels = torch.empty((rows, count), device=scores.device, dtype=torch.float32)
        _prepare[(rows,)](
            values,
            indices,
            candidates if candidates is not None else indices,
            labels,
            lengths,
            count,
            values.stride(0),
            indices.stride(0),
            candidates.stride(0) if candidates is not None else 0,
            candidates.stride(1) if candidates is not None else 0,
            lengths.stride(0),
            candidates is not None,
            triton.next_power_of_2(count),
            num_warps=1,
            multibuffer=False,
        )
    else:
        from tokenspeed_kernel_npu.ops._v41.index_metadata import prepare

        labels = prepare(values, indices, candidates, lengths)
    ordered = labels.sort(dim=1).values
    from tokenspeed_kernel_npu.ops._v41.native_index_selection import finish

    finish(ordered, output)


def block_scores(scores, visible):
    from tokenspeed_kernel_npu.ops._v41.native_index_selection import (
        block_scores as native_block_scores,
    )

    return native_block_scores(scores, visible)
