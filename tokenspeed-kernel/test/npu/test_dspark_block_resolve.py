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
from tokenspeed_kernel.ops.sampling.dspark_block import (
    dspark_block_candidate_tiles,
    dspark_block_greedy_resolve,
    dspark_block_greedy_step,
)


@pytest.mark.parametrize("rows", [1, 4])
def test_public_greedy_entry_and_replay(rows):
    """Greedy DSpark works independently of the optional full-sampling backend."""
    from tokenspeed_kernel.numerics.torch.sampling import torch_dspark_block_greedy

    generator = torch.Generator().manual_seed(91)
    block, vocab, rank = 3, 1024, 32
    cpu_base = torch.randn(rows, block, vocab, generator=generator) * 8
    cpu_embedding = torch.randn(vocab, rank, generator=generator).bfloat16()
    cpu_projection = torch.randn(vocab, rank, generator=generator).bfloat16()
    cpu_anchors = torch.arange(rows, dtype=torch.int32)
    base = cpu_base.to("npu")
    embedding = cpu_embedding.to("npu")
    projection = cpu_projection.to("npu")
    anchors = cpu_anchors.to("npu")
    tiles = dspark_block_candidate_tiles(vocab)
    candidates = torch.empty((1, rows, tiles), dtype=torch.int64, device="npu")
    partials = torch.empty((rows, tiles), dtype=torch.int64, device="npu")
    output = torch.empty((rows, block), dtype=torch.int32, device="npu")

    def forward():
        for step in range(block):
            dspark_block_greedy_step(
                base,
                step,
                anchors,
                candidates,
                embedding,
                projection,
                0,
                vocab,
                partials,
                output,
            )
            candidates[0].copy_(partials)
        dspark_block_greedy_resolve(candidates, output, block - 1)

    for _ in range(2):
        forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        forward()
    try:
        for shift in (0, 17):
            current = cpu_base.roll(shift, dims=-1)
            base.copy_(current)
            graph.replay()
            torch.npu.synchronize()
            expected = torch_dspark_block_greedy(
                current, cpu_anchors, cpu_embedding, cpu_projection
            )
            torch.testing.assert_close(output.cpu().long(), expected, atol=0, rtol=0)
    finally:
        torch.npu.synchronize()
        graph.reset()


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
