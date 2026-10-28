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

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libindex_selection.so")))
    lib.launch_selection_finish.argtypes = (
        [ctypes.c_void_p] * 2
        + [ctypes.c_int] * 3
        + [ctypes.c_int64] * 2
        + [ctypes.c_void_p]
    )
    lib.launch_selection_finish.restype = None
    lib.launch_selection_block_scores.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int] * 2
        + [ctypes.c_int64] * 2
        + [ctypes.c_void_p]
    )
    lib.launch_selection_block_scores.restype = None
    return lib


def finish(ordered, out):
    if (
        ordered.dtype != torch.float32
        or out.dtype != torch.int32
        or ordered.ndim != 2
        or out.ndim != 2
        or ordered.shape[0] != out.shape[0]
        or ordered.shape[1] > out.shape[1]
        or ordered.stride(1) != 1
        or out.stride(1) != 1
    ):
        raise ValueError(
            "Expected FP32 sorted IDs and INT32 output with contiguous columns"
        )
    if ordered.device.type != "npu" or out.device != ordered.device:
        raise ValueError("Selection tensors must share an NPU")
    if out.numel():
        with torch.npu.device(out.device):
            _library().launch_selection_finish(
                ordered.data_ptr(),
                out.data_ptr(),
                out.shape[0],
                ordered.shape[1],
                out.shape[1],
                ordered.stride(0),
                out.stride(0),
                torch.npu.current_stream().npu_stream,
            )
    return out


def block_scores(scores, visible):
    if (
        scores.device.type != "npu"
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or scores.shape[1] % 8
        or scores.stride(1) != 1
        or visible.dtype != torch.int32
        or visible.shape != scores.shape[:1]
        or visible.device != scores.device
    ):
        raise ValueError(
            "Expected FP32 score rows divisible by eight and colocated INT32 lengths"
        )
    out = torch.empty(
        (scores.shape[0], scores.shape[1] // 8),
        dtype=scores.dtype,
        device=scores.device,
    )
    if out.numel():
        with torch.npu.device(out.device):
            _library().launch_selection_block_scores(
                scores.data_ptr(),
                visible.data_ptr(),
                out.data_ptr(),
                scores.shape[0],
                scores.shape[1],
                scores.stride(0),
                visible.stride(0),
                torch.npu.current_stream().npu_stream,
            )
    return out
