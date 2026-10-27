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

"""Index Q quantize/dequantize without a packed global-memory temporary."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _function():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libindex_query.so")))
    fn = lib.launch_index_query_qdq
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = None
    return lib, fn


def quantize_dequantize(x):
    if (
        x.device.type != "npu"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] != 128
        or x.stride(1) != 1
    ):
        raise ValueError(
            "Index Q QDQ requires BF16 NPU [rows,128] with contiguous columns"
        )
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel():
        _, fn = _function()
        with torch.npu.device(x.device):
            fn(
                x.data_ptr(),
                out.data_ptr(),
                x.shape[0],
                x.stride(0),
                torch.npu.current_stream().npu_stream,
            )
    return out
