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

"""DSpark global argmax includes the last tile of every strided TP shard."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel_npu.ops.dspark_block import dspark_block_greedy_resolve


@pytest.mark.parametrize("tp,tiles", [(1, 505), (2, 253), (4, 127), (8, 64)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_resolve_unaligned_shards_and_replay(tp, tiles, rows):
    tokens = torch.arange(tp * rows * tiles, dtype=torch.int64).reshape(tp, rows, tiles)
    cpu = 0xFFFFFFFF - tokens
    candidates = cpu.to("npu")
    output = torch.full((rows, 5), -1, dtype=torch.int32, device="npu")
    for _ in range(3):
        dspark_block_greedy_resolve(candidates, output, 2)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        dspark_block_greedy_resolve(candidates, output, 2)
    try:
        for tile in (0, 1, tiles // 2, tiles - 2, tiles - 1):
            cpu.copy_(0xFFFFFFFF - tokens)
            expected = []
            for row in range(rows):
                shard = (tile + row) % tp
                token = (shard * rows + row) * tiles + tile
                cpu[shard, row, tile] = (0x40000000 << 32) | (0xFFFFFFFF - token)
                expected.append(token)
            candidates.copy_(cpu)
            graph.replay()
            torch.npu.synchronize()
            assert output[:, 2].cpu().tolist() == expected
            assert torch.all(output[:, [0, 1, 3, 4]] == -1)
    finally:
        torch.npu.synchronize()
        graph.reset()
