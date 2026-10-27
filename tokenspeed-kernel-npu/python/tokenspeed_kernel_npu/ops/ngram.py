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

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _engram_remainder(value, prime, reciprocal):
    # Use signed 64-bit lanes and explicit masks: this Ascend compiler does
    # not preserve unsigned widening/right-shift semantics. Four 32-bit
    # limbs reconstruct the exact high half, including all carries.
    value = value.to(tl.int64)
    reciprocal = reciprocal.to(tl.int64)
    prime = prime.to(tl.int64)
    mask = 0xFFFFFFFF
    a0, a1 = value & mask, (value >> 32) & mask
    b0, b1 = reciprocal & mask, (reciprocal >> 32) & mask
    p00, p01, p10 = a0 * b0, a0 * b1, a1 * b0
    carry = ((p00 >> 32) & mask) + (p01 & mask) + (p10 & mask)
    quotient = a1 * b1 + ((p01 >> 32) & mask) + ((p10 >> 32) & mask) + (carry >> 32)
    remainder = value - quotient * prime
    # prime is positive int64. A negative remainder bit pattern represents
    # an unsigned value >= 2**63 and therefore also needs the correction.
    return tl.where(
        (remainder < 0) | (remainder >= prime), remainder - prime, remainder
    )


@triton.jit
def _engram_hash_values(
    current,
    previous0,
    previous1,
    previous2,
    live,
    token_map,
    primes,
    reciprocals,
    offsets,
    multipliers,
    layer,
    heads: tl.constexpr,
    pad_id: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    head = tl.arange(0, COLS)
    blocked = (current == -1) | ~live
    token = tl.load(
        token_map + tl.where(~blocked, current, 0), ~blocked, other=pad_id
    ).to(tl.uint64)
    rolling = token * tl.load(multipliers + layer * 4).to(tl.uint64)
    value = tl.full((ROWS, COLS), 0, tl.uint64)
    for shift in tl.static_range(1, 4):
        if shift == 1:
            raw = previous0
        elif shift == 2:
            raw = previous1
        else:
            raw = previous2
        blocked = blocked | (raw == -1)
        token = tl.load(
            token_map + tl.where(~blocked, raw, 0), ~blocked, other=pad_id
        ).to(tl.uint64)
        rolling = rolling ^ (
            token * tl.load(multipliers + layer * 4 + shift).to(tl.uint64)
        )
        value = tl.where((head[None, :] // heads) == shift - 1, rolling[:, None], value)
    column = layer * 3 * heads + head
    prime = tl.load(primes + column, head < 3 * heads, other=1).to(tl.uint64)
    offset = tl.load(offsets + column, head < 3 * heads, other=0).to(tl.uint64)
    reciprocal = tl.load(reciprocals + column, head < 3 * heads, other=0).to(tl.uint64)
    return (
        _engram_remainder(value, prime[None, :], reciprocal[None, :]) + offset[None, :]
    )


# The seed views (snapshots, positions, reset, ends) start at batch-dependent
# offsets of one buffer, so their alignment must not select a compilation.
_SEED_VIEWS = ["snapshots", "positions", "reset", "ends"]
_BATCH_SCALARS = ["total", "capacity", "bs", "uniform_length", "check_history"]


@triton.jit(
    do_not_specialize=_BATCH_SCALARS,
    do_not_specialize_on_alignment=_SEED_VIEWS + _BATCH_SCALARS,
)
def _prepare_ngram_hash(
    snapshots,
    positions,
    reset,
    slots,
    cache_lengths,
    ends,
    tail,
    needs_seed,
    ids,
    previous,
    token_mask,
    prefix,
    hashes,
    token_map,
    primes,
    reciprocals,
    offsets,
    multipliers,
    total,
    capacity,
    bs,
    uniform_length,
    check_history,
    vocab: tl.constexpr,
    layers: tl.constexpr,
    heads: tl.constexpr,
    pad_id: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    layer = tl.program_id(1)
    live = rows < total
    # Uniform lengths index directly; ragged batches count preceding request
    # ends in fixed tiles. Runtime batch sizes do not select a compilation.
    if uniform_length > 0:
        request = rows // uniform_length
        start = (request * uniform_length).to(tl.int64)
    else:
        request = tl.full((ROWS,), 0, tl.int32)
        columns = tl.arange(0, 32)
        for first_request in range(0, bs, 32):
            owner = first_request + columns
            stop = tl.load(ends + tl.where(owner < bs, owner, 0), owner < bs, other=0)
            before = (rows[:, None] >= stop[None, :]) & (owner[None, :] < bs)
            request += tl.sum(before.to(tl.int32), 1)
        start = tl.load(
            ends + tl.where(live & (request > 0), request - 1, 0),
            live & (request > 0),
            other=0,
        )
    local_row = rows - start
    raw = tl.load(ids + tl.where(live, rows, 0), live, other=-1).to(tl.int64)
    valid = live & (raw >= 0) & (raw < vocab)
    from1 = live & (local_row >= 1)
    from2 = live & (local_row >= 2)
    from3 = live & (local_row >= 3)
    init1 = tl.load(
        prefix
        + tl.where(
            live & ~from1, request * 3 + tl.minimum(tl.maximum(-local_row, 0), 2), 0
        ),
        live & ~from1,
        other=-1,
    )
    init2 = tl.load(
        prefix
        + tl.where(
            live & ~from2, request * 3 + tl.minimum(tl.maximum(1 - local_row, 0), 2), 0
        ),
        live & ~from2,
        other=-1,
    )
    init3 = tl.load(
        prefix
        + tl.where(
            live & ~from3, request * 3 + tl.minimum(tl.maximum(2 - local_row, 0), 2), 0
        ),
        live & ~from3,
        other=-1,
    )
    prev1 = tl.load(ids + tl.where(from1, rows - 1, 0), from1, other=-1).to(tl.int64)
    prev2 = tl.load(ids + tl.where(from2, rows - 2, 0), from2, other=-1).to(tl.int64)
    prev3 = tl.load(ids + tl.where(from3, rows - 3, 0), from3, other=-1).to(tl.int64)
    past1 = tl.where(from1, prev1, init1)
    past2 = tl.where(from2, prev2, init2)
    past3 = tl.where(from3, prev3, init3)
    past1 = tl.where(live & (past1 >= 0) & (past1 < vocab), past1, -1)
    past2 = tl.where(live & (past2 >= 0) & (past2 < vocab), past2, -1)
    past3 = tl.where(live & (past3 >= 0) & (past3 < vocab), past3, -1)
    if layer == 0:
        tl.store(
            previous + tl.where(rows < capacity, rows * 3 + 0, 0),
            past1,
            rows < capacity,
        )
        tl.store(
            previous + tl.where(rows < capacity, rows * 3 + 1, 0),
            past2,
            rows < capacity,
        )
        tl.store(
            previous + tl.where(rows < capacity, rows * 3 + 2, 0),
            past3,
            rows < capacity,
        )
    result = _engram_hash_values(
        raw,
        past1,
        past2,
        past3,
        valid,
        token_map,
        primes,
        reciprocals,
        offsets,
        multipliers,
        layer,
        heads,
        pad_id,
        ROWS,
        COLS,
    )
    head = tl.arange(0, COLS)
    write = (rows[:, None] < capacity) & (head[None, :] < 3 * heads)
    address = (rows[:, None] * layers + layer) * 3 * heads + head[None, :]
    tl.store(hashes + tl.where(write, address, 0), result, write)
    if layer == 0:
        tl.store(
            token_mask + tl.where(rows < capacity, rows, 0), valid, rows < capacity
        )


def prepare_ngram_inputs(
    snapshots: torch.Tensor,
    positions: torch.Tensor,
    reset: torch.Tensor,
    slots: torch.Tensor,
    cache_lengths: torch.Tensor,
    request_ends: torch.Tensor,
    tail: torch.Tensor,
    needs_seed: torch.Tensor,
    ids: torch.Tensor,
    previous: torch.Tensor,
    token_mask: torch.Tensor,
    prefix: torch.Tensor,
    hashes: torch.Tensor,
    total: int,
    vocab_size: int,
    uniform_length: int,
    hash_params,
) -> None:
    """Resolve raw lookbacks, token masks and hashes in one read-only state pass.

    ``snapshots`` is [requests,4], current token then newest-first history.
    ``positions`` and ``reset`` describe those host snapshots; only reseeded
    slots require a zero/one-token frontier delta. ``request_ends`` contains
    cumulative packed input lengths, including zero-length requests.
    ``tail`` and ``needs_seed`` are runtime-owned and never mutated here.
    ``prefix`` [requests,3] retains the resolved starting history for commit.
    ``previous`` [capacity,3], ``token_mask`` [capacity] and ``hashes``
    [capacity,layers,3*heads] are caller-owned stable outputs. Rows after
    ``total`` are dead padding. Uniform positive request length selects direct
    indexing; zero selects ragged indexing. All modes use this same operation.
    """
    bs = slots.numel()
    capacity = previous.shape[0]
    if max(capacity, bs) == 0:
        return
    layers, _, heads = hash_params.primes.shape
    rows = 4
    if bs:
        from tokenspeed_kernel_npu.ops._ngram_prefix import resolve

        resolve(
            snapshots,
            positions,
            reset,
            slots,
            cache_lengths,
            tail,
            needs_seed,
            prefix,
            bs,
            vocab_size,
        )
    if not capacity:
        return
    _prepare_ngram_hash[(triton.cdiv(capacity, rows), layers)](
        snapshots,
        positions,
        reset,
        slots,
        cache_lengths,
        request_ends,
        tail,
        needs_seed,
        ids,
        previous,
        token_mask,
        prefix,
        hashes,
        hash_params.token_map,
        hash_params.primes,
        hash_params.reciprocals,
        hash_params.offsets,
        hash_params.multipliers,
        total,
        capacity,
        bs,
        uniform_length,
        int(capacity > 2048),
        vocab_size,
        layers,
        heads,
        hash_params.pad_id,
        rows,
        triton.next_power_of_2(3 * heads),
        num_warps=4,
        multibuffer=False,
        # Reject uncovered snapshots instead of silently hashing stale history.
        debug=True,
    )
