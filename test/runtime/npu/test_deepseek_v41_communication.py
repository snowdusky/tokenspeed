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

"""Run with torchrun and pytest to verify the Ascend collective compatibility."""

import os

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attention import dsv41  # noqa: F401
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel_npu import register_torch_compat

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.fixture(scope="module")
def mapping():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        pytest.skip("requires at least two torchrun workers")
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    layout = Mapping(world_size=world, rank=rank, attn_tp_size=world)
    pg_manager.init_distributed(layout, backend="hccl", timeout=300)
    pg_manager.init_process_group(layout.attn.tp_group)
    yield layout
    dist.destroy_process_group()


@pytest.mark.parametrize("rows", [1, 32])
def test_hccl_collectives_and_async_work(mapping, rows):
    group = mapping.attn.tp_group
    process_group = pg_manager.get_process_group("hccl", group)
    rank, world = mapping.rank, mapping.world_size
    gather, scatter = dist.all_gather_single, dist.reduce_scatter_single
    register_torch_compat()
    assert dist.all_gather_single is gather
    assert dist.reduce_scatter_single is scatter

    local = torch.full((rows, 1), rank + 1, dtype=torch.bfloat16, device="npu")
    gathered = torch.empty((world * rows, 1), dtype=local.dtype, device="npu")
    expected = torch.arange(1, world + 1, dtype=torch.bfloat16).repeat_interleave(rows)[
        :, None
    ]
    backend = HcclBackend()
    backend.all_gather_single(gathered, local, group)
    torch.testing.assert_close(gathered.cpu(), expected, atol=0, rtol=0)

    work = dist.all_gather_single(gathered, local, group=process_group, async_op=True)
    assert work is not None
    work.wait()
    torch.testing.assert_close(gathered.cpu(), expected, atol=0, rtol=0)

    partial = local.repeat(world, 1)
    reduced = backend.reduce_scatter(partial, group)
    expected_sum = torch.full((rows, 1), world * (world + 1) // 2, dtype=local.dtype)
    torch.testing.assert_close(reduced.cpu(), expected_sum, atol=0, rtol=0)
    work = dist.reduce_scatter_single(
        reduced, partial, group=process_group, async_op=True
    )
    assert work is not None
    work.wait()
    torch.testing.assert_close(reduced.cpu(), expected_sum, atol=0, rtol=0)


def test_hccl_int64_argmax_acl_replay(mapping):
    backend = HcclBackend()
    values = torch.empty(2, dtype=torch.int64, device="npu")

    def update(step):
        values.copy_(
            torch.tensor(
                [(step << 32) + mapping.rank, -mapping.rank - step], device="npu"
            )
        )

    def forward():
        result = values.clone()
        backend.all_reduce(result, mapping.attn.tp_group, op=dist.ReduceOp.MAX)
        return result

    update(3)
    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        actual = forward()
    for step in (7, 13):
        update(step)
        expected = torch.tensor([(step << 32) + mapping.world_size - 1, -step])
        torch.testing.assert_close(forward().cpu(), expected, atol=0, rtol=0)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("algorithm", ["host", "aiv"])
def test_hccl_fp32_sum_keeps_precision_across_transport_sizes(mapping, algorithm):
    from tokenspeed.runtime.distributed.comm_backend.hccl import (
        AIV_TP_GROUP_NAME,
        HcclTpPolicy,
    )

    if mapping.world_size not in (2, 4, 8):
        pytest.skip("AIV comparison requires two, four or eight local ranks")
    group = mapping.attn.tp_group
    if algorithm == "aiv":
        import torch_npu

        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_op_expansion_mode": 3}
        pg_manager.init_process_group(
            group, backend="hccl", name=AIV_TP_GROUP_NAME, pg_options=options
        )
    backend = HcclBackend()
    policy = HcclTpPolicy(
        reduce_dtype="float32",
        reduce_algorithm=algorithm,
        gather_algorithm=algorithm,
        aiv_max_bytes=128,
    )
    policy.validate_group(group, nprocs_per_node=int(os.environ["LOCAL_WORLD_SIZE"]))
    backend.configure_tp(group, policy=policy)
    # FP32 adds the small terms before the final BF16 rounding. The same
    # values cross the AIV byte threshold as the number of rows changes.
    values = [256.0, 0.5, -256.0, 0.5, 16.0, 0.125, -16.0, 0.125][: mapping.world_size]
    for rows in (1, 16, 64):
        local = torch.full(
            (rows,), values[mapping.rank], dtype=torch.bfloat16, device="npu"
        )
        reduced = torch.empty_like(local)
        gathered = torch.empty(
            (mapping.world_size * rows,), dtype=torch.bfloat16, device="npu"
        )

        def forward():
            reduced.copy_(local)
            backend.all_reduce(reduced, group)
            backend.all_gather_single(gathered, reduced, group)

        forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            forward()
        for scale in (1.0, 2.0):
            local.fill_(values[mapping.rank] * scale)
            graph.replay()
            torch.npu.synchronize()
            expected = torch.full((rows,), sum(values) * scale, dtype=torch.bfloat16)
            torch.testing.assert_close(reduced.cpu(), expected, atol=0, rtol=0)
            torch.testing.assert_close(
                gathered.cpu(), expected.repeat(mapping.world_size), atol=0, rtol=0
            )
        graph.reset()
