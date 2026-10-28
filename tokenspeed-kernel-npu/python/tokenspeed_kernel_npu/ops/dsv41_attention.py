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

"""Selected attention over the original V4.1 cache, without expanded KV scratch."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_cache import _load_rows


@triton.jit
def _segment(
    q,
    C,
    Slots,
    Lengths,
    token,
    CP,
    CR,
    CB,
    S0,
    S1,
    L0,
    capacity,
    width,
    scale,
    maximum,
    denominator,
    accumulator,
    FORMAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    d = tl.arange(0, 512)
    length = (
        width
        if Lengths is None
        else tl.minimum(tl.maximum(tl.load(Lengths + token * L0), 0), width)
    )
    for start in range(0, length, BLOCK):
        col = start + tl.arange(0, BLOCK)
        slots = tl.load(Slots + token * S0 + col * S1, col < length, other=-1).to(
            tl.int64
        )
        valid = (col < length) & (slots >= 0) & (slots < capacity)
        if FORMAT == "dense":
            kv = tl.load(
                C + tl.where(valid, slots, 0)[:, None] * CP + d[None, :],
                valid[:, None],
                other=0.0,
            ).to(tl.float32)
        elif FORMAT == "swa":
            kv = _load_rows(
                C,
                slots[:, None],
                d[None, :],
                valid[:, None],
                CP,
                CR,
                CB,
                64,
                512,
                32,
                512,
                528,
                FORMAT,
            )
        else:
            kv = _load_rows(
                C,
                slots[:, None],
                d[None, :],
                valid[:, None],
                CP,
                CR,
                CB,
                64,
                512,
                16,
                256,
                288,
                FORMAT,
            )
        # The portable numerical contract materializes decoded BF16 keys/values.
        kv = kv.to(tl.bfloat16).to(tl.float32)
        logits = tl.where(valid, tl.sum(kv * q[None, :], 1) * scale, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 0))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.where(valid, tl.exp(logits - safe_max), 0.0)
        accumulator = accumulator * alpha + tl.sum(p[:, None] * kv, 0)
        denominator = denominator * alpha + tl.sum(p, 0)
        maximum = next_max
    return maximum, denominator, accumulator


@triton.jit
def _attention(
    Q,
    S,
    SS,
    SL,
    G,
    GS,
    GL,
    Sink,
    O,
    LSE,
    Q0,
    Q1,
    O0,
    O1,
    HEADS,
    SP,
    SR,
    SB,
    SS0,
    SS1,
    SL0,
    SCAP,
    SW,
    GP,
    GR,
    GB,
    GS0,
    GS1,
    GL0,
    GCAP,
    GW,
    scale,
    FORMAT: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 512)
    q = tl.load(Q + token * Q0 + head * Q1 + d).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    if Sink is not None:
        maximum = tl.load(Sink + head).to(tl.float32)
        denominator = tl.where(maximum == -float("inf"), 0.0, 1.0)
    accumulator = tl.full((512,), 0.0, tl.float32)
    maximum, denominator, accumulator = _segment(
        q,
        S,
        SS,
        SL,
        token,
        SP,
        SR,
        SB,
        SS0,
        SS1,
        SL0,
        SCAP,
        SW,
        scale,
        maximum,
        denominator,
        accumulator,
        FORMAT,
        4,
    )
    if G is not None:
        maximum, denominator, accumulator = _segment(
            q,
            G,
            GS,
            GL,
            token,
            GP,
            GR,
            GB,
            GS0,
            GS1,
            GL0,
            GCAP,
            GW,
            scale,
            maximum,
            denominator,
            accumulator,
            "global",
            4,
        )
    result = tl.where(denominator > 0.0, accumulator / denominator, 0.0)
    tl.store(O + token * O0 + head * O1 + d, result)
    if LSE is not None:
        tl.store(
            LSE + token * HEADS + head,
            tl.where(denominator > 0.0, maximum + tl.log(denominator), -float("inf")),
        )


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
):
    if (
        q.ndim != 3
        or q.shape[-1] != 512
        or q.dtype != torch.bfloat16
        or q.stride(-1) != 1
    ):
        raise ValueError(
            "query must be BF16 [tokens, heads, 512] with contiguous dimensions"
        )
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    if schedule is not None:
        raise ValueError("Ascend attention does not use a FlashMLA schedule")
    if out is None:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError("output must be contiguous BF16 with query shape")
    if lse is not None and (
        lse.shape != q.shape[:2]
        or lse.dtype != torch.float32
        or not lse.is_contiguous()
        or attn_sink is not None
    ):
        raise ValueError("LSE requires contiguous FP32 [tokens, heads] and no sink")
    if attn_sink is not None and (
        attn_sink.ndim != 1
        or attn_sink.numel() < q.shape[1]
        or attn_sink.dtype != torch.float32
    ):
        raise ValueError("sink must provide one FP32 logit per head")
    for tensor in (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        out,
        prefill_kv,
        prefill_indices,
        lse,
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("attention inputs must share the query device")

    def segment(cache, slots, lens, width):
        if cache is None:
            if slots is not None or lens is not None:
                raise ValueError("absent cache requires absent selection metadata")
            return (0,) * 8
        if (
            cache.dtype != torch.uint8
            or cache.ndim != 3
            or cache.shape[1:] != (64, width)
        ):
            raise ValueError(f"cache must be uint8 [pages, 64, {width}]")
        if (
            slots is None
            or slots.ndim != 2
            or slots.shape[0] != q.shape[0]
            or slots.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("slots must be integer [tokens, width]")
        if (
            lens is None
            or lens.shape != q.shape[:1]
            or lens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("lengths must be integer [tokens]")
        return (
            *cache.stride(),
            *slots.stride(),
            lens.stride(0),
            cache.shape[0] * 64,
            slots.shape[1],
        )

    if prefill_kv is not None:
        if any(x is not None for x in (swa_slots, swa_lens, global_slots, global_lens)):
            raise ValueError("compact prefill cannot include paged selections")
        if (
            prefill_kv.dtype != torch.bfloat16
            or prefill_kv.shape[-1] != 512
            or not prefill_kv.is_contiguous()
        ):
            raise ValueError("compact KV must be contiguous BF16 rows of width 512")
        if (
            prefill_indices is None
            or prefill_indices.ndim != 2
            or prefill_indices.shape[0] != q.shape[0]
            or prefill_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("compact prefill requires integer [tokens, width] indices")
        swa_cache, swa_slots, swa_lens = prefill_kv, prefill_indices, None
        swa = (
            512,
            0,
            0,
            *swa_slots.stride(),
            0,
            prefill_kv.numel() // 512,
            swa_slots.shape[1],
        )
        # The backend supplies cache handles even when its compact workspace
        # already contains both SWA and global rows. Read only that workspace.
        global_cache = None
        glob, fmt = (0,) * 8, "dense"
    else:
        if prefill_indices is not None or swa_cache is None:
            raise ValueError(
                "paged attention requires SWA cache and no compact indices"
            )
        swa = segment(swa_cache, swa_slots, swa_lens, 528)
        glob = segment(global_cache, global_slots, global_lens, 288)
        fmt = "swa"
    if q.shape[0] and q.shape[1]:
        with torch.npu.device(q.device):
            _attention[(q.shape[0], q.shape[1])](
                q,
                swa_cache,
                swa_slots,
                swa_lens,
                global_cache,
                global_slots,
                global_lens,
                attn_sink,
                out,
                lse,
                *q.stride()[:2],
                *out.stride()[:2],
                q.shape[1],
                *swa,
                *glob,
                softmax_scale,
                fmt,
                enable_fp_fusion=False,
                multibuffer=False,
            )
    return out
