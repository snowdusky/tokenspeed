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

"""Run under torchrun on 32/64/128 Ascend ranks with HCCL_BUFFSIZE=3072."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_npu
    or int(os.environ.get("WORLD_SIZE", "1")) not in (32, 64, 128),
    reason="requires 32, 64 or 128 Ascend ranks",
)


@pytest.fixture(scope="module")
def transport():
    import flash_ops  # noqa: F401
    import torch_npu  # noqa: F401
    from tokenspeed_kernel_npu.ops.dsv41_mc2 import MC2Transport

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    # Native hierarchy owns this communicator; barriers use the default group.
    group = dist.new_group(list(range(world)), backend="hccl")
    name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    yield MC2Transport(name, rank, world, 384 + world, 5120, 6)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


@pytest.mark.parametrize("capacity", [17, 24, 48, 96, 256, 384])
@pytest.mark.parametrize("routing", ["balanced", "hot"])
@torch.no_grad()
def test_mc2_full_batch_roundtrip_and_graph(transport, capacity, routing):
    rank = transport.rank
    # Include idle and partially filled ranks in the same collective.
    rows = 0 if rank % 4 == 0 else capacity - rank % 3
    row_values = (
        (torch.arange(rows, device="npu", dtype=torch.int32) * 7 + rank * 11) % 31 - 15
    ).bfloat16()
    original = row_values[:, None].expand(rows, 5120).contiguous()
    x = original.clone()
    if routing == "balanced":
        ids = (
            torch.arange(rows * 6, dtype=torch.int32, device="npu").reshape(rows, 6)
            + rank * 19
        ) % transport.experts
    else:
        ids = (
            torch.tensor([41, 66, 80, 178, 190, 201], dtype=torch.int32, device="npu")[
                None
            ]
            .expand(rows, -1)
            .contiguous()
        )
    weights = torch.full((rows, 6), 0.125, dtype=torch.float32, device="npu")

    def forward():
        expanded, route_weights, counts, handle = transport.dispatch(
            x, ids, weights, max_rows=capacity
        )
        output = (expanded.float() * route_weights[:, None]).bfloat16()
        return transport.combine(output, handle), counts

    for _ in range(2):
        actual, _ = forward()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, x * 0.75, atol=0, rtol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, counts = forward()
    for value in (2.0, 4.0):
        x.copy_(original * value)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, x * 0.75, atol=0, rtol=0)
        # Includes zero-weight padding, which must survive Dispatch but make
        # no contribution to the caller's output.
        total = counts.sum().clone()
        dist.all_reduce(total)
        assert total.item() == capacity * transport.world_size * 6
    graph.reset()
