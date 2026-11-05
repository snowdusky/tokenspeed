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

"""Native page-planar KV attention and compact BF16 prefill.

The existing cache writer and selection semantics are unchanged. QK/PV use
BF16 operands and FP32 accumulators; softmax probabilities round to BF16.
"""

import torch
from tokenspeed_kernel_npu.ops._v41.native_attention_codec_bits import Native


def selected_attention(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
    *,
    lse=None,
    rotary=None,
):
    if schedule is not None or query_chunk_size < 1:
        raise ValueError(
            "native Ascend requires no FlashMLA schedule and a positive query tile"
        )
    if (
        q.ndim != 3
        or q.shape[2] != 512
        or q.dtype != torch.bfloat16
        or not 1 <= q.shape[1] <= 64
    ):
        raise ValueError("expected BF16 query [rows, heads <= 64, 512]")
    packed = prefill_kv is None
    if packed:
        if (
            swa_cache is None
            or swa_slots is None
            or swa_lens is None
            or prefill_indices is not None
        ):
            raise ValueError("paged attention requires SWA cache and selections")
        sw, gl, ss, gs, sl, ll = (
            swa_cache,
            global_cache,
            swa_slots,
            global_slots,
            swa_lens,
            global_lens,
        )
        if (gl is None) != (gs is None) or (gl is None) != (ll is None):
            raise ValueError("global cache and metadata must be supplied together")
    else:
        if prefill_indices is None or any(
            x is not None for x in (swa_slots, swa_lens, global_slots, global_lens)
        ):
            raise ValueError("compact prefill requires only compact indices")
        sw, gl = prefill_kv.reshape(-1, 512), None
        ss, gs, sl, ll = prefill_indices, None, None, None
    for slots in (ss, gs):
        if slots is not None and (
            slots.ndim != 2
            or slots.shape[0] != q.shape[0]
            or slots.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid selection tensor")
    for lengths in (sl, ll):
        if lengths is not None and (
            lengths.shape != q.shape[:1]
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid length tensor")
    for x in (sw, gl, ss, gs, sl, ll, attn_sink, out, lse):
        if x is not None and x.device != q.device:
            raise ValueError("all inputs must share a device")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 with query shape")
    if lse is not None and (
        lse.shape != q.shape[:2] or lse.dtype != torch.float32 or attn_sink is not None
    ):
        raise ValueError("LSE requires FP32 query shape and no sink")
    if q.shape[0] == 0:
        return torch.empty_like(q) if out is None else out
    # Native kernels read contiguous planes. Keep the public strided-cache
    # contract using the existing Ascend implementation for other layouts.
    if packed and any(
        cache is not None
        and (
            cache.shape[1] != 64
            or cache.stride(1) != cache.shape[2]
            or cache.stride(2) != 1
        )
        for cache in (swa_cache, global_cache)
    ):
        from tokenspeed_kernel_npu.ops.dsv41_attention import (
            selected_attention as strided_attention,
        )

        result = strided_attention(
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            softmax_scale,
            out,
            query_chunk_size,
            schedule,
            prefill_kv,
            prefill_indices,
            lse=lse,
        )
        if rotary is not None:
            from tokenspeed_kernel_npu.ops._v41.rope_native import rope_inplace

            positions, cache = rotary
            rope_inplace(result, positions, cache)
        return result
    sink = (
        torch.full((q.shape[1],), -float("inf"), device=q.device, dtype=torch.float32)
        if attn_sink is None
        else attn_sink[: q.shape[1]].contiguous()
    )
    if gl is None:
        shape = (0, 64, 288) if packed else (0, 512)
        gl = torch.empty(shape, dtype=sw.dtype, device=q.device)
    native = Native(
        q.contiguous(),
        sw,
        gl,
        ss,
        gs,
        sl,
        ll,
        sink,
        packed,
        128,
        paged=packed,
        scale=softmax_scale,
        row_owned=False,
    )
    result = native.run(rotary=rotary)
    if lse is not None:
        lse.copy_(native.lse)
    if out is not None:
        out.copy_(result)
        return out
    return result


def selected_attention_rope(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    schedule,
    positions,
    cos_sin_cache,
):
    return selected_attention(
        q,
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        softmax_scale,
        None,
        256,
        schedule,
        None,
        None,
        rotary=(positions, cos_sin_cache),
    )
