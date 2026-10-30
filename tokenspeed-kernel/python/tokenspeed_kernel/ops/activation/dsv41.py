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

"""Explicit activation contracts for DeepSeek V4.1 components."""

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["shared_expert_swiglu"]


def shared_expert_swiglu(x: torch.Tensor, *, limit: float | None) -> torch.Tensor:
    """Compute the V4.1 shared expert's SwiGLU with one BF16 output rounding.

    Args:
        x: BF16 ``[..., 2 * hidden]`` concatenated gate and up projections.
        limit: Positive clamp or None. Gate is capped above; up is clamped
            symmetrically before SwiGLU.

    Returns:
        BF16 ``[..., hidden]``, with FP32 activation and multiplication.
    """
    kernel = select_kernel(
        "activation",
        "dsv41_shared_expert_swiglu",
        format_signature(x=dense_tensor_format(x.dtype)),
    )
    return kernel(x, limit=limit)


# Register the device leaf without importing NPU dependencies on other platforms.
if current_platform().is_npu:
    import tokenspeed_kernel.ops.activation._ascend.dsv41  # noqa: F401
