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

"""Native metadata transforms; scoring, TopK and tie choices are unchanged."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libindex_metadata.so")))
    pointers = ctypes.POINTER(ctypes.c_void_p)
    lib.launch_table.argtypes = (
        [pointers] + [ctypes.c_int] * 5 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    )
    lib.launch_prepare.argtypes = (
        [pointers] + [ctypes.c_int] * 3 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    )
    lib.launch_table.restype = lib.launch_prepare.restype = None
    return lib


def selection_table(positions, requests, table, ratio):
    if positions.ndim != 1 or requests.shape != positions.shape or table.ndim != 2:
        raise ValueError(
            "Expected positions/requests [rows] and table [requests,pages]"
        )
    tensors = (positions, requests, table)
    if positions.device.type != "npu" or any(
        t.device != positions.device for t in tensors
    ):
        raise ValueError("Metadata must share an NPU device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise ValueError("Metadata must be INT32 or INT64")
    if (
        table.stride(1) != 1
        or ratio not in (1, 2)
        or any(s < 0 for t in tensors for s in t.stride())
    ):
        raise ValueError(
            "Expected contiguous page columns and compression ratio 1 or 2"
        )
    rows, width = positions.numel(), table.shape[1]
    output = torch.empty((rows, width), device=positions.device, dtype=torch.int32)
    lengths = torch.empty(rows, device=positions.device, dtype=torch.int32)
    if rows:
        pointers = (ctypes.c_void_p * 5)(
            *[t.data_ptr() for t in (*tensors, output, lengths)]
        )
        types = sum(int(t.dtype == torch.int64) << i for i, t in enumerate(tensors))
        with torch.npu.device(positions.device):
            library().launch_table(
                pointers,
                rows,
                table.shape[0],
                width,
                ratio,
                types,
                positions.stride(0),
                requests.stride(0),
                table.stride(0),
                torch.npu.current_stream().npu_stream,
            )
    return output, lengths


def prepare(values, indices, candidates, lengths):
    if (
        values.ndim != 2
        or indices.shape != values.shape
        or lengths.shape != values.shape[:1]
    ):
        raise ValueError("Expected TopK values/indices [rows,k] and lengths [rows]")
    if (
        values.dtype != torch.float32
        or indices.dtype != torch.int64
        or lengths.dtype != torch.int32
    ):
        raise ValueError("Expected FP32 scores, INT64 TopK indices and INT32 lengths")
    if (
        values.stride(1) != 1
        or indices.stride(1) != 1
        or not 1 <= values.shape[1] <= 2048
    ):
        raise ValueError("TopK columns must be contiguous with 1 <= k <= 2048")
    tensors = [values, indices, lengths]
    if candidates is not None:
        if (
            candidates.ndim != 2
            or candidates.shape[0] != values.shape[0]
            or candidates.dtype != torch.int32
            or candidates.stride(1) != 1
            or not 1 <= candidates.shape[1] <= 2048
        ):
            raise ValueError(
                "Candidates must be contiguous INT32 blocks [rows,1..2048]"
            )
        tensors.append(candidates)
    if values.device.type != "npu" or any(t.device != values.device for t in tensors):
        raise ValueError("Selection tensors must share an NPU device")
    labels = torch.empty_like(values, memory_format=torch.contiguous_format)
    if values.shape[0]:
        pointers = (ctypes.c_void_p * 5)(
            values.data_ptr(),
            indices.data_ptr(),
            0 if candidates is None else candidates.data_ptr(),
            labels.data_ptr(),
            lengths.data_ptr(),
        )
        with torch.npu.device(values.device):
            library().launch_prepare(
                pointers,
                *values.shape,
                0 if candidates is None else candidates.shape[1],
                values.stride(0),
                indices.stride(0),
                0 if candidates is None else candidates.stride(0),
                lengths.stride(0),
                torch.npu.current_stream().npu_stream,
            )
    return labels
