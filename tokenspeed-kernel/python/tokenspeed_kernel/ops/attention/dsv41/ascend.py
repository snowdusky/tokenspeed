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

"""Ascend implementations of the shared V4.1 operator contracts."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops import dsv41_cache, dsv41_index, dsv41_metadata

    @register_kernel(
        "attention",
        "dsv41_cache_pack",
        name="ascend_dsv41_cache_pack",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float32}),
        priority=Priority.PERFORMANT,
    )
    def ascend_dsv41_cache_pack(rows, cache_format, out):
        return dsv41_cache.cache_pack(rows, cache_format, out)

    @register_kernel(
        "attention",
        "dsv41_cache_unpack",
        name="ascend_dsv41_cache_unpack",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.uint8}),
        priority=Priority.PERFORMANT,
    )
    def ascend_dsv41_cache_unpack(packed, cache_format, out):
        return dsv41_cache.cache_unpack(packed, cache_format, out)

    @register_kernel(
        "attention",
        "dsv41_cache_scatter",
        name="ascend_dsv41_cache_scatter",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float32}),
        priority=Priority.PERFORMANT,
    )
    def ascend_dsv41_cache_scatter(rows, cache, slots, cache_format):
        dsv41_cache.cache_scatter(rows, cache, slots, cache_format)

    @register_kernel(
        "attention",
        "dsv41_cache_gather",
        name="ascend_dsv41_cache_gather",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.uint8}),
        priority=Priority.PERFORMANT,
    )
    def ascend_dsv41_cache_gather(cache, slots, cache_format, out):
        return dsv41_cache.cache_gather(cache, slots, cache_format, out)

    from tokenspeed_kernel.ops.attention.dsv41 import triton as portable

    def _register_portable(name, role, dtypes):
        return register_kernel(
            "attention",
            "dsv41_" + name,
            name="ascend_dsv41_" + name,
            solution="ascend",
            capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
            signatures=format_signatures(role, "dense", dtypes),
            priority=Priority.PORTABLE,
        )

    @_register_portable("rope_inplace", "values", {torch.bfloat16, torch.float16})
    def ascend_dsv41_rope_inplace(values, positions, cache):
        if values.dtype == torch.bfloat16 and cache.shape[1] == 64:
            from tokenspeed_kernel_npu.ops._v41.rope_native import rope_inplace

            return rope_inplace(values, positions, cache)
        cosine, sine = cache.index_select(0, positions.clamp_min(0).long()).chunk(2, -1)
        if values.ndim == 3:
            cosine, sine = cosine[:, None], sine[:, None]
        tail = values[..., -cache.shape[1] :]
        even, odd = tail.float().unflatten(-1, (cache.shape[1] // 2, 2)).unbind(-1)
        rotated = torch.stack(
            (even * cosine - odd * sine, odd * cosine + even * sine), -1
        )
        tail.copy_(rotated.flatten(-2))
        return values

    @_register_portable("rope_pad_query", "values", {torch.bfloat16, torch.float16})
    def ascend_dsv41_rope_pad_query(values, positions, cache):
        return portable.rope_pad_query(values, positions, cache)

    @_register_portable("decode_rows", "x", {torch.int32, torch.int64})
    def ascend_dsv41_decode_rows(*args):
        return portable.decode_rows(*args)

    @_register_portable("decode_window", "x", {torch.int32, torch.int64})
    def ascend_dsv41_decode_window(*args):
        return portable.decode_window(*args)

    @_register_portable("global_slots", "x", {torch.int32, torch.int64})
    def ascend_dsv41_global_slots(rows, positions, requests, table, ratio, pages):
        # Stage the small page table in UB before vector lookup. The portable
        # path retains support for int64 selections and non-contiguous columns.
        if (
            rows.dtype == torch.int32
            and rows.stride(1) == 1
            and table.dtype == torch.int32
            and table.stride(1) == 1
            and 0 < table.shape[1] <= 4096
            and 0 <= pages <= 16777216
        ):
            from tokenspeed_kernel_npu.ops._v41.native_global_slots import global_slots

            return global_slots(rows, positions, requests, table, ratio, pages)
        return portable.global_slots(rows, positions, requests, table, ratio, pages)

    @_register_portable("selection_table", "x", {torch.int32, torch.int64})
    def ascend_dsv41_selection_table(*args):
        return portable.selection_table(*args)

    @_register_portable("compressor_metadata", "x", {torch.int32, torch.int64})
    def ascend_dsv41_compressor_metadata(*args):
        return dsv41_metadata.compressor_metadata(*args)

    @_register_portable("compressor_pool", "x", {torch.float32})
    def ascend_dsv41_compressor_pool(*args):
        return portable.compressor_pool(*args)

    @_register_portable("compressor_tail_scatter", "x", {torch.bfloat16, torch.float32})
    def ascend_dsv41_compressor_tail_scatter(*args):
        return portable.compressor_tail_scatter(*args)

    @_register_portable("index_select_all", "x", {torch.int32, torch.int64})
    def ascend_dsv41_index_select_all(*args):
        return dsv41_index.index_select_all(*args)

    @_register_portable("compact_slots", "x", {torch.int32, torch.int64})
    def ascend_dsv41_compact_slots(*args):
        return portable.compact_slots(*args)

    @_register_portable("index_q_quantize", "x", {torch.bfloat16, torch.float32})
    def ascend_dsv41_index_q_quantize(q, cache_format, out):
        return portable.index_q_quantize(q, cache_format, out)

    from tokenspeed_kernel_npu.ops import dsv41_attention, dsv41_dspark

    @_register_portable("dspark_rows", "x", {torch.bfloat16})
    def ascend_dsv41_dspark_rows(*args):
        return dsv41_dspark.rows(*args)

    @_register_portable("dspark_workspace", "x", {torch.bfloat16})
    def ascend_dsv41_dspark_workspace(*args):
        return dsv41_dspark.workspace(*args)

    @_register_portable("selected_attention", "x", {torch.bfloat16})
    def ascend_dsv41_selected_attention(*args, **kwargs):
        return dsv41_attention.selected_attention(*args, **kwargs)

    from tokenspeed_kernel.ops.attention.dsv41._triton import rope as portable_rope

    @_register_portable("index_key_norm_rope", "x", {torch.bfloat16})
    def ascend_dsv41_index_key_norm_rope(values, weight, positions, cos_sin_cache, eps):
        portable_rope._validate_index_key_inputs(
            values, weight, positions, cos_sin_cache, eps
        )
        from tokenspeed_kernel.ops.layernorm import reference_rmsnorm

        normalized = reference_rmsnorm(values, weight, eps, None)
        return ascend_dsv41_rope_inplace(normalized, positions, cos_sin_cache)

    @_register_portable("index_key_norm_rope_scatter", "x", {torch.bfloat16})
    def ascend_dsv41_index_key_norm_rope_scatter(
        values, weight, positions, cos_sin_cache, eps, cache, slots
    ):
        rotated = ascend_dsv41_index_key_norm_rope(
            values, weight, positions, cos_sin_cache, eps
        )
        dsv41_cache.cache_scatter(rotated, cache, slots, "index")

    @_register_portable("index_query_rope_weights", "x", {torch.bfloat16})
    def ascend_dsv41_index_query_rope_weights(
        query, head_weights, positions, cos_sin_cache, weight_scale
    ):
        portable_rope._validate_index_query_inputs(
            query, head_weights, positions, cos_sin_cache
        )
        if any(
            size > 1 and stride == 0
            for size, stride in zip(query.shape, query.stride())
        ):
            raise ValueError("Index query must not have broadcast writes")
        ascend_dsv41_rope_inplace(query, positions, cos_sin_cache)
        return query, (head_weights.float() * weight_scale).to(query.dtype).contiguous()

    @_register_portable("global_rope_scatter", "x", {torch.bfloat16})
    def ascend_dsv41_global_rope_scatter(
        values, positions, cos_sin_cache, cache, slots
    ):
        rotated = ascend_dsv41_rope_inplace(values.clone(), positions, cos_sin_cache)
        dsv41_cache.cache_scatter(rotated, cache, slots, "global")

    @_register_portable("swa_rope_scatter", "x", {torch.bfloat16})
    def ascend_dsv41_swa_rope_scatter(
        values,
        positions,
        cos_sin_cache,
        cache,
        slots,
        cache_format,
        out,
        norm_weight,
        norm_eps,
    ):
        if cache_format != "swa":
            raise ValueError("Ascend SWA uses the original V4.1 cache format")
        if norm_weight is None:
            rotated = values.clone()
        else:
            if norm_weight.shape != (512,) or norm_eps <= 0:
                raise ValueError("SWA norm requires [512] weights and positive epsilon")
            from tokenspeed_kernel.ops.layernorm import reference_rmsnorm

            rotated = reference_rmsnorm(values, norm_weight, norm_eps, None)
        ascend_dsv41_rope_inplace(rotated, positions, cos_sin_cache)
        dsv41_cache.cache_scatter(rotated, cache, slots, cache_format)
        if out is not None:
            packed = dsv41_cache.cache_pack(rotated, cache_format, None)
            dsv41_cache.cache_unpack(packed, cache_format, out)

    @_register_portable("index_score", "x", {torch.bfloat16})
    def ascend_dsv41_index_score(*args):
        return portable.index_score(*args)

    @_register_portable("index_topk", "x", {torch.bfloat16})
    def ascend_dsv41_index_topk(
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
        return dsv41_index.index_topk(
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


def prepare_native_tables(device):
    """Upload immutable codec tables before distributed forwards or capture."""
    from tokenspeed_kernel_npu.ops._v41.native_attention_codec_bits import (
        make_tables as attention_tables,
    )
    from tokenspeed_kernel_npu.ops._v41.native_index_reduction import (
        make_tables as index_tables,
    )

    index_tables(device)
    attention_tables(device)
