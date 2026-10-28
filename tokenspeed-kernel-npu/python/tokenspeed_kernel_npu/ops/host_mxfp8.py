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
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


class HostMXFP8Embedding:
    """Own registered host FP8/E8M0 pages for graph-safe device lookup.

    The owner and its tables must outlive every captured graph using them.
    Synchronize outstanding device work and destroy those graphs before close.
    """

    def __init__(self, weight, scale, row_start, row_end):
        if weight.device.type != "cpu" or scale.device.type != "cpu":
            raise ValueError("Expected CPU tables")
        if not weight.is_contiguous() or not scale.is_contiguous():
            raise ValueError("Expected contiguous host tables")
        if (
            weight.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or scale.dtype != torch.uint8
        ):
            raise ValueError("Expected FP8 codes and raw E8M0 scale bytes")
        if (
            not 0 <= row_start <= row_end
            or weight.ndim != 2
            or weight.shape[1] != 256
            or row_end - row_start > weight.shape[0]
            or scale.shape != (weight.shape[0], 8)
        ):
            raise ValueError("Expected 256-wide Engram with group32 scales")
        self.weight, self.scale = weight, scale
        self.row_start, self.row_end = row_start, row_end
        self.device = torch.npu.current_device()
        self.runtime = ctypes.CDLL("libascendcl.so")
        self.runtime.aclrtHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.runtime.aclrtHostRegister.restype = ctypes.c_int
        self.runtime.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.runtime.aclrtHostUnregister.restype = ctypes.c_int
        self.native = ctypes.CDLL(
            str(Path(__file__).with_name("libgather_fp8_fused.so"))
        )
        self.native.launch_fp8_fused.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self.native.launch_fp8_fused.restype = None
        code = torch.arange(256, dtype=torch.uint8, device="cpu")
        self.lut = torch.cat(
            (
                code.view(torch.float8_e4m3fn).float(),
                code.view(torch.float8_e8m0fnu).float(),
            )
        ).to(f"npu:{self.device}")
        self.mappings = []
        try:
            for table in (weight, scale):
                if table.data_ptr() % 4096:
                    raise ValueError("Host mapping must be page aligned")
                mapped = ctypes.c_void_p()
                rc = self.runtime.aclrtHostRegister(
                    table.data_ptr(),
                    table.numel() * table.element_size(),
                    0,
                    ctypes.byref(mapped),
                )
                if rc:
                    raise RuntimeError(f"aclrtHostRegister: {rc}")
                self.mappings.append((table.data_ptr(), mapped.value))
        except Exception:
            self.close()
            raise

    def __call__(self, indices):
        if (
            indices.device.type != "npu"
            or indices.dtype != torch.int64
            or not indices.is_contiguous()
        ):
            raise ValueError("Expected contiguous NPU int64 indices")
        if indices.device.index != self.device or len(self.mappings) != 2:
            raise ValueError(
                "Host table mapping belongs to another device or is closed"
            )
        out = torch.empty(
            (*indices.shape, 256), device=indices.device, dtype=torch.bfloat16
        )
        if indices.numel():
            self.native.launch_fp8_fused(
                self.mappings[0][1],
                self.mappings[1][1],
                indices.data_ptr(),
                out.data_ptr(),
                self.lut.data_ptr(),
                indices.numel(),
                self.row_start,
                self.row_end,
                torch.npu.current_stream().npu_stream,
            )
        return out

    def close(self):
        # Caller must finish all graph work before closing its mapped table.
        while self.mappings:
            pointer, _ = self.mappings[-1]
            rc = self.runtime.aclrtHostUnregister(pointer)
            if rc:
                raise RuntimeError(f"aclrtHostUnregister: {rc}")
            self.mappings.pop()
