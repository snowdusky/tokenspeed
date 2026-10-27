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

"""Vector gather from a staged page table, preserving integer slot semantics."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _function():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libglobal_slots.so")))
    fn = lib.launch_global_slots
    fn.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_int] * 8
        + [ctypes.c_int64] * 4
        + [ctypes.c_void_p]
    )
    fn.restype = None
    return lib, fn


def global_slots(rows, positions, requests, table, ratio, pages):
    if rows.dtype != torch.int32 or rows.ndim != 2 or rows.stride(1) != 1:
        raise ValueError("selected rows must be int32 with contiguous columns")
    if (
        table.dtype != torch.int32
        or table.ndim != 2
        or table.stride(1) != 1
        or not 0 < table.shape[1] <= 4096
    ):
        raise ValueError("page table must be int32 with 1..4096 contiguous columns")
    if ratio <= 0 or not 0 <= pages <= 16777216:
        raise ValueError(
            "ratio must be positive and page IDs fit exact FP32 comparison"
        )
    for v in (positions, requests):
        if v.dtype not in (torch.int32, torch.int64) or v.shape != (rows.shape[0],):
            raise ValueError("one int32/int64 position and request per query required")
    if (
        any(x.device != rows.device for x in (positions, requests, table))
        or rows.device.type != "npu"
    ):
        raise ValueError("all inputs must be on the same NPU")
    out = torch.empty(rows.shape, device=rows.device, dtype=torch.int32)
    if out.numel():
        _, fn = _function()
        fn(
            rows.data_ptr(),
            positions.data_ptr(),
            requests.data_ptr(),
            table.data_ptr(),
            out.data_ptr(),
            *rows.shape,
            *table.shape,
            ratio,
            pages,
            positions.element_size(),
            requests.element_size(),
            rows.stride(0),
            positions.stride(0),
            requests.stride(0),
            table.stride(0),
            torch.npu.current_stream().npu_stream,
        )
    return out
