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
    lib = ctypes.CDLL(str(Path(__file__).with_name("libswa_pack.so")))
    fn = lib.launch_swa_pack
    fn.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int] * 4
        + [ctypes.c_int64] * 2
        + [ctypes.c_void_p]
    )
    fn.restype = None
    return lib, fn


def cache_scatter(rows, cache, slots, cache_format):
    if (
        cache_format != "swa"
        or rows.dtype != torch.bfloat16
        or rows.ndim != 2
        or rows.shape[1] != 512
        or rows.stride(1) != 1
    ):
        raise ValueError(
            "native SWA packing requires BF16 [rows,512] with contiguous columns"
        )
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[2] != 528
        or cache.stride(2) != 1
        or cache.stride(1) != 528
    ):
        raise ValueError(
            "native SWA packing requires contiguous planes within each page"
        )
    if slots is not None and (
        slots.shape != (rows.shape[0],)
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
    ):
        raise ValueError("native SWA packing requires contiguous int32/int64 slots")
    if rows.numel():
        _, fn = _function()
        fn(
            rows.data_ptr(),
            0 if slots is None else slots.data_ptr(),
            cache.data_ptr(),
            rows.shape[0],
            cache.shape[1],
            cache.shape[0],
            0 if slots is None else slots.element_size(),
            rows.stride(0),
            cache.stride(0),
            torch.npu.current_stream().npu_stream,
        )
