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

"""Register native attention through TokenSpeed's kernel plugin boundary."""


def register():
    import torch
    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures
    from tokenspeed_kernel_npu.ops._v41.runtime_attention_rope import selected_attention

    register_kernel(
        "attention",
        "dsv41_selected_attention",
        name="ascend_native_v41_selected_attention",
        solution="ascend_native",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
    )(selected_attention)

    from tokenspeed_kernel_npu.ops._v41.runtime_attention_rope import (
        selected_attention_rope,
    )

    register_kernel(
        "attention",
        "dsv41_selected_attention_rope",
        name="ascend_native_v41_selected_attention_rope",
        solution="ascend_native",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
    )(selected_attention_rope)
