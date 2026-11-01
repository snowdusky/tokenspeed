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

"""Public DSpark block sampling entry points."""

import torch
from tokenspeed_kernel.ops.sampling.triton.dspark_block import (
    dspark_block_candidate_tiles,
)
from tokenspeed_kernel.ops.sampling.triton.dspark_block import (
    dspark_block_greedy_resolve as triton_resolve,
)
from tokenspeed_kernel.ops.sampling.triton.dspark_block import (
    dspark_block_greedy_step as triton_step,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

__all__ = [
    "dspark_block_candidate_tiles",
    "dspark_block_greedy_resolve",
    "dspark_block_greedy_step",
]

register_kernel(
    "sampling",
    "dspark_block_greedy_step",
    name="triton_dspark_block_greedy_step",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("base_logits", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)(triton_step)


register_kernel(
    "sampling",
    "dspark_block_greedy_resolve",
    name="triton_dspark_block_greedy_resolve",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("candidates", "dense", {torch.int64}),
    priority=Priority.PORTABLE,
)(triton_resolve)


def dspark_block_greedy_resolve(
    candidates: torch.Tensor, output: torch.Tensor, step: int
) -> None:
    """Write gathered [tp,rows,tiles] INT64 winners into INT32 output[:,step]."""
    if candidates.ndim != 3 or candidates.dtype != torch.int64:
        raise ValueError("candidates must be [tp, rows, n_tiles] INT64")
    kernel = select_kernel(
        "sampling",
        "dspark_block_greedy_resolve",
        format_signature(candidates=dense_tensor_format(candidates.dtype)),
    )
    kernel(candidates, output, step)


def dspark_block_greedy_step(
    base_logits: torch.Tensor,
    step: int,
    anchor_ids: torch.Tensor,
    candidates: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
    vocab_start: int,
    num_valid: int,
    partials: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Score this rank's vocabulary shard for one block step.

    Args:
        base_logits: ``[rows, block, local_vocab]`` FP32 base logits of this
            rank's shard; column ``v`` is token ``vocab_start + v``.
        step: Block step in ``[0, block)`` being scored.
        anchor_ids: ``[rows]`` int32/int64 tokens preceding step 0, in any
            stride (the drafters pass a column of their token table); read
            only when ``step == 0``.
        candidates: ``[tp, rows, n_tiles]`` int64 candidates gathered after
            the previous step; read only when ``step > 0``, and then disjoint
            from ``partials``.
        embedding: ``[vocab, rank]`` BF16 replicated bigram table.
        projection: ``[padded_local_vocab, rank]`` BF16 projection shard whose
            row ``v`` belongs to token ``vocab_start + v``.
        vocab_start: First token id of this rank's shard.
        num_valid: Shard columns that are real tokens; the rest are padding.
        partials: ``[rows, n_tiles]`` int64 destination for this rank's packed
            candidates.
        output: ``[rows, block]`` int32 block tokens; column ``step - 1``
            receives the previous step's winner when ``step > 0``.

    Returns:
        None. ``partials`` and, for ``step > 0``, one ``output`` column are
        written in place.

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the kernel
            indexes with, or ``step`` is outside the block.
    """
    if base_logits.ndim != 3 or base_logits.dtype != torch.float32:
        raise ValueError("base_logits must be a 3D FP32 tensor")
    kernel = select_kernel(
        "sampling",
        "dspark_block_greedy_step",
        format_signature(base_logits=dense_tensor_format(base_logits.dtype)),
    )
    kernel(
        base_logits,
        step,
        anchor_ids,
        candidates,
        embedding,
        projection,
        vocab_start,
        num_valid,
        partials,
        output,
    )


import tokenspeed_kernel.ops.sampling._ascend.dspark_block  # noqa: E402,F401
