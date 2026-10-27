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
def _function():
    lib = ctypes.CDLL(str(Path(__file__).with_name("librope.so")))
    fn = lib.launch_rope
    fn.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int] * 3
        + [ctypes.c_int64] * 4
        + [ctypes.c_void_p]
    )
    fn.restype = None
    return lib, fn


def rope_inplace(values, positions, cache):
    if values.dtype != torch.bfloat16 or cache.shape[1] != 64 or values.stride(-1) != 1:
        raise ValueError(
            "native RoPE requires BF16 values, contiguous columns and rotary dim 64"
        )
    if values.numel():
        _, fn = _function()
        p = positions.to(torch.int64)
        fn(
            values.data_ptr(),
            p.data_ptr(),
            cache.data_ptr(),
            values.shape[0],
            1 if values.ndim == 2 else values.shape[1],
            values.shape[-1],
            values.stride(0),
            64 if values.ndim == 2 else values.stride(1),
            p.stride(0),
            cache.stride(0),
            torch.npu.current_stream().npu_stream,
        )
    return values
