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

"""Live metadata updates must be observed by every native graph replay."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel_npu.ops._v41.index_metadata import prepare, selection_table


@pytest.mark.parametrize("rows", [1, 6, 48])
@pytest.mark.parametrize("ratio", [1, 2])
def test_native_table_replay(rows, ratio):
    positions = torch.zeros(rows, dtype=torch.int64, device="npu")
    requests = torch.zeros_like(positions)
    table = torch.arange(5 * 513, device="npu", dtype=torch.int32).reshape(5, 513)
    selection_table(positions, requests, table, ratio)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output, lengths = selection_table(positions, requests, table, ratio)
    for step in range(3):
        host_positions = torch.arange(rows, dtype=torch.int64) * 65 - step
        host_requests = (torch.arange(rows) + step) % 7 - 1
        positions.copy_(host_positions)
        requests.copy_(host_requests)
        graph.replay()
        torch.npu.synchronize()
        expected = table.cpu()[host_requests.clamp(0, 4)]
        expected[(host_requests < 0) | (host_requests >= 5)] = -1
        expected[expected <= 0] = -1
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(
            lengths.cpu(),
            ((host_positions + 1) // ratio).clamp_min(0).int(),
            atol=0,
            rtol=0,
        )
    graph.reset()


@pytest.mark.parametrize("sparse", [False, True])
def test_native_candidate_mapping_replay(sparse):
    rows, k = 6, 512
    values = torch.ones(rows, k, dtype=torch.float32, device="npu")
    indices = (
        torch.arange(k, dtype=torch.int64, device="npu").expand(rows, -1).contiguous()
    )
    candidates = (
        torch.arange(2048, dtype=torch.int32, device="npu")
        .expand(rows, -1)
        .contiguous()
        if sparse
        else None
    )
    lengths = torch.empty(rows, dtype=torch.int32, device="npu")
    prepare(values, indices, candidates, lengths)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        labels = prepare(values, indices, candidates, lengths)
    for step in range(3):
        host_values = torch.ones(rows, k)
        host_values[:, ::3] = -float("inf") if step < 2 else float("nan")
        if step == 0:
            host_values.fill_(-float("inf"))
        values.copy_(host_values)
        if sparse:
            candidates.add_(1)
        graph.replay()
        torch.npu.synchronize()
        mapped = indices.cpu().int()
        if sparse:
            mapped = candidates.cpu().gather(1, (mapped // 8).long()) * 8 + mapped % 8
        valid = host_values > -float("inf")
        expected = torch.where(valid, mapped.float(), float("inf"))
        torch.testing.assert_close(labels.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(lengths.cpu(), valid.sum(1).int(), atol=0, rtol=0)
    graph.reset()
