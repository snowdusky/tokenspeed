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

"""Ascend DSpark registration, independent of optional full sampling kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "sampling",
    "dspark_block_greedy_step",
    name="ascend_dspark_block_greedy_step",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures("base_logits", "dense", {torch.float32}),
    priority=Priority.PERFORMANT,
)
def ascend_dspark_block_greedy_step(
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
):
    from tokenspeed_kernel_npu.ops.dspark_block import dspark_block_greedy_step

    return dspark_block_greedy_step(
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
