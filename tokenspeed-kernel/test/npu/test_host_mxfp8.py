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

"""Mapped host FP8 lookup must preserve CPU decode bytes across graph replay."""

import mmap

import pytest
import torch
import torch_npu
from tokenspeed_kernel_npu.ops.host_mxfp8 import HostMXFP8Embedding


@pytest.mark.parametrize("valid_rows", [13, 16])
def test_host_lookup_and_live_replay(valid_rows):
    rows, start = 16, 100
    weight_memory = mmap.mmap(-1, 4096)
    scale_memory = mmap.mmap(-1, 4096)
    weight = torch.frombuffer(
        weight_memory, dtype=torch.uint8, count=rows * 256
    ).reshape(rows, 256)
    scales = torch.frombuffer(scale_memory, dtype=torch.uint8, count=rows * 8).reshape(
        rows, 8
    )
    # Cover all finite E4M3 codes, including signed zero and subnormals.
    weight.copy_(torch.arange(rows * 256).remainder(256).byte().reshape_as(weight))
    weight[(weight & 127) == 127] = 0
    scales.copy_(torch.arange(rows * 8).remainder(9).add(123).byte().reshape_as(scales))
    lookup = HostMXFP8Embedding(weight, scales, start, start + valid_rows)
    ids = torch.empty(7, dtype=torch.int64, device="npu")
    graph = torch.npu.NPUGraph()
    try:
        ids.copy_(torch.tensor([99, 100, 101, 107, 115, 116, -1]))
        for _ in range(3):
            lookup(ids)
        torch.npu.synchronize()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            output = lookup(ids)
        for offset in (0, 3, 8):
            cpu_ids = torch.tensor([99, 100 + offset, 101, 107, 115, 116, -1])
            ids.copy_(cpu_ids)
            graph.replay()
            torch.npu.synchronize()
            local = (cpu_ids - start).clamp(0, rows - 1)
            values = weight[local].contiguous().view(torch.float8_e4m3fn).float()
            scale = scales[local].contiguous().view(torch.float8_e8m0fnu).float()
            expected = (
                (values.reshape(-1, 8, 32) * scale.unsqueeze(-1))
                .reshape(-1, 256)
                .bfloat16()
            )
            expected[(cpu_ids < start) | (cpu_ids >= start + valid_rows)] = 0
            torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    finally:
        torch.npu.synchronize()
        graph.reset()
        lookup.close()
    with pytest.raises(ValueError, match="closed"):
        lookup(ids)
