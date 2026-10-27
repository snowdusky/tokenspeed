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

"""BF16 operands with FP32 accumulation for V4 projections."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import bf16_linear_fp32

    @register_kernel(
        "gemm",
        "dsv4_linear_fp32",
        name="ascend_dsv4_linear_fp32",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "hidden_rank": frozenset({2}),
            "hidden_inner_stride_one": frozenset({True}),
            "weight_contiguous": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def ascend_dsv4_linear_fp32(hidden_states, weight, enable_pdl=False):
        if enable_pdl:
            raise ValueError("Ascend projection does not support PDL")
        return bf16_linear_fp32(hidden_states, weight)
