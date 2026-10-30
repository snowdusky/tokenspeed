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

"""Ascend stop-condition update with DMA-safe masked addresses."""

from tokenspeed_kernel_npu._triton import tl, triton

_BLOCK = 64


@triton.jit(do_not_specialize=["batch_size", "num_extends", "num_prefill_outputs"])
def _apply_stop_conditions_kernel(
    req_pool_indices_ptr,
    output_tokens_ptr,
    accept_lengths_ptr,
    input_lengths_ptr,
    prompt_lengths_ptr,
    valid_cache_lengths_ptr,
    stop_token_ids_ptr,
    token_limits_ptr,
    stopped_ptr,
    batch_size,
    num_extends,
    num_prefill_outputs,
    padding_index,
    BLOCK: tl.constexpr,
    DECODE_WIDTH: tl.constexpr,
    WIDTH_PAD: tl.constexpr,
    NUM_STOP_IDS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = rows < batch_size
    slots = tl.load(req_pool_indices_ptr + rows, mask=row_mask, other=padding_index)
    live = row_mask & (slots != padding_index)
    slots = tl.where(live, slots, 0)
    row_offsets = tl.where(row_mask, rows, 0)
    is_extend = rows < num_extends
    valid = tl.load(valid_cache_lengths_ptr + slots, mask=live, other=0)
    limits = tl.load(token_limits_ptr + slots, mask=live, other=0)

    # Every output token of the row: the prefill sample at its request index,
    # a decode row's verify window at its fixed offset.
    columns = tl.arange(0, WIDTH_PAD)
    width = tl.where(is_extend, 1, DECODE_WIDTH)
    first_token = tl.where(
        is_extend,
        rows,
        num_prefill_outputs + (rows - num_extends) * DECODE_WIDTH,
    )
    has_output = live & (~is_extend | (rows < num_prefill_outputs))
    token_mask = has_output[:, None] & (columns[None, :] < width[:, None])
    tokens = tl.load(
        output_tokens_ptr
        + tl.where(token_mask, first_token[:, None] + columns[None, :], 0),
        mask=token_mask,
        other=-1,
    )
    is_stop = tl.zeros((BLOCK, WIDTH_PAD), dtype=tl.int1)
    for k in tl.static_range(NUM_STOP_IDS):
        stop_id = tl.load(
            stop_token_ids_ptr + slots * NUM_STOP_IDS + k, mask=live, other=-1
        )
        is_stop = is_stop | ((tokens == stop_id[:, None]) & (stop_id[:, None] >= 0))

    # Extend rows: only the chunk that completes the prompt owes a token.
    input_lengths = tl.load(input_lengths_ptr + row_offsets, mask=row_mask, other=0)
    prompt_lengths = tl.load(
        prompt_lengths_ptr + tl.where(row_mask & is_extend, rows, 0),
        mask=row_mask & is_extend,
        other=-1,
    )
    completes_prompt = valid + input_lengths == prompt_lengths
    extend_stops = (
        completes_prompt
        & (rows < num_prefill_outputs)
        & ((tl.sum(is_stop.to(tl.int32), axis=1) > 0) | (prompt_lengths + 1 >= limits))
    )

    # Decode rows: keep through the first accepted stop token and within the
    # budget, nothing once the request has stopped.
    accepted = tl.load(accept_lengths_ptr + row_offsets, mask=row_mask, other=0).to(
        tl.int32
    )
    first_stop = tl.min(
        tl.where(
            is_stop & (columns[None, :] < accepted[:, None]),
            columns[None, :],
            WIDTH_PAD,
        ),
        axis=1,
    )
    # A live request always has room for one more token; the floor keeps a
    # broken invariant from walking the cache length backwards.
    budget = tl.maximum(limits - valid - 1, 0)
    kept = tl.minimum(tl.minimum(accepted, first_stop + 1), budget)
    already_stopped = tl.load(stopped_ptr + slots, mask=live, other=0) != 0
    decode_stops = already_stopped | (first_stop < kept) | (kept >= budget)
    kept = tl.where(already_stopped, 0, kept)

    decode = live & ~is_extend
    tl.store(accept_lengths_ptr + rows, kept, mask=decode)
    stops = tl.where(is_extend, extend_stops, decode_stops)
    tl.store(stopped_ptr + slots, stops.to(tl.int1), mask=live)


def apply_stop_conditions(
    req_pool_indices,
    output_tokens,
    accept_lengths,
    input_lengths,
    prompt_lengths,
    valid_cache_lengths,
    stop_token_ids,
    token_limits,
    stopped,
    num_extends,
    num_prefill_outputs,
    decode_width,
    padding_index,
) -> None:
    """Apply the portable device update after public input validation.

    Registry leaves accept all 13 arguments positionally in this order:
    ``req_pool_indices``, ``output_tokens``, ``accept_lengths``,
    ``input_lengths``, ``prompt_lengths``, ``valid_cache_lengths``,
    ``stop_token_ids``, ``token_limits``, ``stopped``, ``num_extends``,
    ``num_prefill_outputs``, ``decode_width``, ``padding_index``.
    Shapes, dtypes and in-place semantics follow :func:`apply_stop_conditions`.
    """
    batch_size = req_pool_indices.shape[0]
    _apply_stop_conditions_kernel[(triton.cdiv(batch_size, _BLOCK),)](
        req_pool_indices,
        # An unfinished prefill batch can owe no sample. Keep a valid pointer
        # even when every token load is masked (also required by Ascend DMA).
        output_tokens if output_tokens.numel() else accept_lengths,
        accept_lengths,
        input_lengths,
        # An empty extend span still needs a valid pointer.
        prompt_lengths if prompt_lengths.numel() else accept_lengths,
        valid_cache_lengths,
        stop_token_ids,
        token_limits,
        stopped,
        batch_size,
        num_extends,
        num_prefill_outputs,
        padding_index,
        BLOCK=_BLOCK,
        DECODE_WIDTH=decode_width,
        WIDTH_PAD=max(2, triton.next_power_of_2(decode_width)),
        NUM_STOP_IDS=stop_token_ids.shape[1],
        num_warps=4,
    )
