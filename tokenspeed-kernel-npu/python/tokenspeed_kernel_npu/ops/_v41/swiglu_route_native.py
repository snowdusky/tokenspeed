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

"""AscendC clipped SwiGLU that reads the route weight from the router output."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _library():
    path = Path(__file__).with_name("libswiglu_route.so")
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    lib.launch_swiglu_route.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_int] * 5
        + [ctypes.c_int64] * 3
        + [ctypes.c_float]
        + [ctypes.c_void_p]
    )
    lib.launch_swiglu_route.restype = None
    return lib


def available() -> bool:
    """Return whether the compiled provider is present."""
    return _library() is not None


def swiglu_route_native(
    gate_up: torch.Tensor,
    mapping: torch.Tensor,
    topk_weights: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
    limit: float,
) -> bool:
    """Run the fused route-weight SwiGLU in place into ``out``.

    Returns False for unsupported metadata or a missing library so the caller
    can use the Triton implementation. The kernel writes one output row per route, so ``out`` must be
    the expanded (expert-sorted) buffer the grouped GEMM consumes.
    """
    if gate_up.ndim != 2 or topk_weights.ndim != 2:
        return False
    rows, two_h = gate_up.shape
    hidden = two_h // 2
    capacity = mapping.numel()
    # The native ABI uses BF16 activations, FP32 weights, INT32 routes and
    # INT64 counts; its fixed UB buffers hold at most 2304 hidden values.
    if (
        gate_up.device.type != "npu"
        or any(t.device != gate_up.device for t in (mapping, topk_weights, counts, out))
        or gate_up.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or mapping.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or counts.dtype != torch.int64
        or two_h != 2 * hidden
        or not 0 < hidden <= 2304
        or hidden % 16 != 0
        or gate_up.stride(1) != 1
        or out.shape != (rows, hidden)
        or not out.is_contiguous()
        or mapping.ndim != 1
        or not mapping.is_contiguous()
        or counts.ndim != 1
        or not counts.is_contiguous()
        or topk_weights.shape[1] == 0
        or topk_weights.numel() != capacity
    ):
        return False
    lib = _library()
    if lib is None:
        return False
    lib.launch_swiglu_route(
        gate_up.data_ptr(),
        mapping.data_ptr(),
        topk_weights.data_ptr(),
        counts.data_ptr(),
        out.data_ptr(),
        rows,
        capacity,
        int(topk_weights.shape[1]),
        hidden,
        int(counts.numel()),
        int(gate_up.stride(0)),
        int(topk_weights.stride(0)),
        int(topk_weights.stride(1)),
        float(limit),
        torch.npu.current_stream().npu_stream,
    )
    return True
