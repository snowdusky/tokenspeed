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

"""Explicit Index plugin for the packed native scan and TP query ownership."""


def register():
    import torch
    from tokenspeed_kernel.ops.attention.dsv41 import triton as portable
    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures
    from tokenspeed_kernel_npu.ops._v41.index_metadata import selection_table
    from tokenspeed_kernel_npu.ops._v41.native_index_topk import index_topk

    @register_kernel(
        "attention",
        "dsv41_selection_table",
        name="ascend_native_v41_selection_table",
        solution="ascend_native",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=portable._INTEGER_SIGNATURES,
        priority=Priority.SPECIALIZED,
    )
    def table(positions, requests, page_table, ratio):
        return selection_table(positions, requests, page_table, ratio)

    @register_kernel(
        "attention",
        "dsv41_index_topk",
        name="ascend_native_v41_index_topk",
        solution="ascend_native",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
    )
    def index(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
        index_metadata,
    ):
        if index_metadata is not None:
            # Keep the shared geometry ownership/lifetime checks. This scan
            # reads canonical live pages directly and needs no cached schedule.
            index_metadata.validate(page_table, visible_lens)
        out = portable._index_topk_outputs(
            index_q,
            weights,
            index_cache,
            page_table,
            visible_lens,
            candidate_blocks,
            topk,
            candidate_topk,
            candidate_block_size,
            query_chunk_size,
            score_chunk_size,
            process_group,
            out,
        )
        q, w, shards = portable._index_gather_heads(index_q, weights, process_group)
        # Clamp before narrowing so invalid 64-bit IDs cannot wrap into pages.
        if page_table.dtype == torch.int64:
            page_table = page_table.clamp(-1, index_cache.shape[0]).int()
        if visible_lens.dtype == torch.int64:
            visible_lens = visible_lens.clamp(0, page_table.shape[1] * 64).int()
        if candidate_blocks is not None and candidate_blocks.dtype == torch.int64:
            candidate_blocks = candidate_blocks.clamp(-1, page_table.shape[1] * 8).int()
        args = (
            q,
            w,
            index_cache,
            page_table,
            visible_lens,
            candidate_blocks,
            topk,
            candidate_topk,
            query_chunk_size,
            shards,
            out,
        )
        return index_topk(*args)
