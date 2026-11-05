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

"""Ascend metadata kernels behind the shared metadata contracts."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "metadata",
    "apply_stop_conditions",
    name="ascend_apply_stop_conditions",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures("req_pool_indices", "dense", {torch.int64}),
    priority=Priority.PERFORMANT,
)
def ascend_apply_stop_conditions(
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
):
    from tokenspeed_kernel_npu.ops.stop_conditions import apply_stop_conditions

    apply_stop_conditions(
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
    )
