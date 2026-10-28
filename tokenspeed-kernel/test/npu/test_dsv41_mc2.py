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

"""Run with one pytest process per rank under a 32/64/128-rank torchrun."""

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
def group():
    import flash_ops  # noqa: F401
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    world_size = int(os.environ["WORLD_SIZE"])
    mapping = Mapping(
        rank=int(os.environ["RANK"]),
        world_size=world_size,
        attn_tp_size=1,
        attn_dp_size=world_size,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=world_size,
        nprocs_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
        nnodes=world_size // int(os.environ["LOCAL_WORLD_SIZE"]),
    )
    process_group_manager.init_distributed(mapping, backend="hccl", timeout=120)
    group = dist.distributed_c10d._get_default_group()
    process_group_manager.register_process_group("hccl", mapping.moe.ep_group, group)
    previous = global_server_args_dict["mapping"]
    global_server_args_dict["mapping"] = mapping
    yield group
    global_server_args_dict["mapping"] = previous
    dist.destroy_process_group()


@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("at_capacity", [False, True])
@pytest.mark.parametrize("routing", ["balanced", "concentrated", "spread"])
def test_route_carrier_and_graph(group, experts, top_k, mixed, at_capacity, routing):
    from tokenspeed_kernel_npu.ops.dsv41_mc2 import MC2Transport

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    rank, world = dist.get_rank(), dist.get_world_size()
    transport_group = process_group_manager.get_dedicated_device_group(
        global_server_args_dict["mapping"].moe.ep_group,
        "ascend_mc2",
    )
    name = transport_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    transport = MC2Transport(name, rank, world, experts, 5120, top_k)
    max_rows = transport.max_dispatch_rows if at_capacity else 6
    rows = (rank % 3) * (max_rows // 2) if mixed else max_rows
    x = torch.empty((rows, 5120), dtype=torch.bfloat16, device="npu")
    ids = torch.empty((rows, top_k), dtype=torch.int32, device="npu")
    weights = torch.empty((rows, top_k), dtype=torch.float32, device="npu")

    def inputs(generation):
        cpu_x = (
            (torch.arange(rows * 5120).reshape(rows, 5120) + generation) % 17 + rank
        ).bfloat16()
        cpu_ids = (
            (
                torch.arange(rows * top_k).reshape(rows, top_k)
                + rank * 19
                + generation * 37
            )
            % experts
        ).int()
        if routing == "concentrated":
            # Identical prompts can send every rank to the same experts.
            # Rotate the receiving owner between replays, including empty
            # owners on other ranks; balanced routes do not cover this case.
            cpu_ids = (
                (
                    torch.arange(top_k).expand(rows, -1)
                    + (generation * 19 % world) * (experts // world)
                )
                .remainder(experts)
                .int()
            )
        elif routing == "spread":
            # All senders select a few hot experts on different owners.
            # This exercises the hierarchy's cross-owner fanout under skew.
            k = torch.arange(top_k)
            owners = (generation * 19 + k * (world // top_k)) % world
            local_experts = experts // world
            cpu_ids = (
                (owners * local_experts + k % local_experts).expand(rows, -1).int()
            )
        # Non-BF16-representable route weights catch accidental downcasts.
        cpu_weights = (
            torch.arange(rows * top_k).float().reshape(rows, top_k) * 0.000123
            + 0.812345
            + generation * 0.0001
        )
        x.copy_(cpu_x)
        ids.copy_(cpu_ids)
        weights.copy_(cpu_weights)
        return (
            (cpu_x.float()[:, None, :] * cpu_weights[:, :, None])
            .bfloat16()
            .float()
            .sum(1)
            .bfloat16()
        )

    def forward():
        expanded, route_weights, counts, handle = transport.dispatch(
            x, ids, weights, max_rows=max_rows
        )
        output = (expanded.float() * route_weights[:, None]).bfloat16()
        return transport.combine(output, handle)

    expected = inputs(0)
    torch.testing.assert_close(forward().cpu(), expected, atol=0, rtol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        result = forward()
    for generation in range(1, 5):
        expected = inputs(generation)
        # Other model collectives may run between MC2 replays. Their scratch
        # must not alias the transport's persistent communication window.
        sentinel = torch.tensor([rank + generation], dtype=torch.float32, device="npu")
        dist.all_reduce(sentinel, group=group)
        torch.testing.assert_close(
            sentinel.cpu(),
            torch.tensor([world * (world - 1) / 2 + world * generation]),
            atol=0,
            rtol=0,
        )
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    graph.reset()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("max_rows", [6, 96, 257, 384])
@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("routing", ["balanced", "concentrated", "spread"])
def test_w4a4_experts_and_graph(group, mixed, max_rows, experts, top_k, routing):
    from tokenspeed_kernel_npu.ops.dsv41_w4a4 import grouped_experts

    from tokenspeed.runtime.layers.moe import utils
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
    from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "w4a4_gmm_is_transposed": True,
            "config_groups": {
                "experts": {
                    "targets": ["Moe"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
    )
    previous = utils.MOE_BACKEND, utils.ALL2ALL_BACKEND, torch.get_default_dtype()
    utils.MOE_BACKEND = utils.MoeBackend("ascend_v41_w4a4")
    utils.ALL2ALL_BACKEND = utils.All2AllBackend.ASCEND_MC2
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("npu"):
            layer = MoELayer(
                top_k=top_k,
                num_experts=experts,
                hidden_size=5120,
                intermediate_size=2304,
                quant_config=config,
                layer_index=0,
                prefix="test",
                tp_rank=0,
                tp_size=1,
                ep_rank=dist.get_rank(),
                ep_size=dist.get_world_size(),
                activation="swiglu",
                swiglu_limit=10.0,
                routing_mode="precomputed_topk",
                internal_activation_dtype_override="int4",
            )
    finally:
        utils.MOE_BACKEND, utils.ALL2ALL_BACKEND, old_dtype = previous
        torch.set_default_dtype(old_dtype)
    generator = torch.Generator().manual_seed(71)
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        parameter = dict(layer.named_parameters())[name]
        for shard in (("w2",) if name.startswith("w2") else ("w1", "w3")):
            shape = tuple(parameter.shape[1:])
            if shard != "w2":
                shape = (shape[0] // 2, *shape[1:])
            value = (
                torch.full(shape, 1 / 128, dtype=torch.float32)
                if name.endswith("scale")
                else torch.randint(
                    0, 2**31 - 1, shape, dtype=torch.int32, generator=generator
                )
            )
            for expert in range(layer.num_local_experts):
                expert_value = (
                    value * (1 + (expert % 4) / 4) if name.endswith("scale") else value
                )
                parameter.weight_loader(parameter, expert_value, shard, expert)
    for stage in (1, 2):
        transform = layer.w1_trans if stage == 1 else layer.w2_trans
        for parameter in (transform.matrix_left, transform.matrix_right):
            parameter.weight_loader(
                parameter, torch.eye(parameter.shape[0], dtype=torch.bfloat16)
            )
        clip = layer.act_quantizer_w1 if stage == 1 else layer.act_quantizer_w2
        clip.clip_factor_a.weight_loader(clip.clip_factor_a, torch.ones(1))
    layer.process_weights_after_loading(layer)
    rank = dist.get_rank()
    rows = (rank % 3) * (max_rows // 2) if mixed else max_rows
    x = torch.empty((rows, 5120), device="npu", dtype=torch.bfloat16)
    ids = torch.empty((rows, top_k), device="npu", dtype=torch.int32)
    weights = torch.empty((rows, top_k), device="npu", dtype=torch.float32)
    routing = StandardTopKOutput(weights, ids, None)

    def inputs(generation):
        cpu_x = (
            (
                (torch.arange(rows * 5120).reshape(rows, 5120) + generation * 13 + rank)
                % 43
                - 21
            ).float()
            / 32
        ).bfloat16()
        x.copy_(cpu_x)
        cpu_ids = (
            torch.arange(rows * top_k).reshape(rows, top_k)
            + rank * 19
            + generation * 37
        ) % experts
        if routing == "concentrated":
            cpu_ids = (
                torch.arange(top_k).expand(rows, -1)
                + (generation * 19 % dist.get_world_size()) * layer.num_local_experts
            ).remainder(experts)
        elif routing == "spread":
            k = torch.arange(top_k)
            world = dist.get_world_size()
            owners = (generation * 19 + k * (world // top_k)) % world
            cpu_ids = (
                owners * layer.num_local_experts + k % layer.num_local_experts
            ).expand(rows, -1)
        ids.copy_(cpu_ids)
        weights.copy_(
            torch.arange(rows * top_k).float().reshape(rows, top_k) * 0.000123
            + 0.112345
            + generation * 0.001
        )
        if rows == 0:
            return x.cpu()
        # Every rank has the same local expert scale classes. Independently
        # sort routes on the CPU and compute them locally, bypassing MC2.
        num_classes = min(4, layer.num_local_experts)
        classes = (
            ids.cpu().flatten().remainder(layer.num_local_experts).remainder(4).long()
        )
        order = torch.argsort(classes, stable=True).to("npu")
        expanded = x.repeat_interleave(top_k, dim=0)[order]
        counts = torch.zeros(layer.num_local_experts, dtype=torch.int64)
        counts[:num_classes] = torch.bincount(classes, minlength=num_classes)
        reference = grouped_experts(
            expanded,
            weights.flatten()[order],
            counts.to("npu"),
            w=layer,
            plan=layer.plan,
        )
        token_order = torch.empty_like(reference)
        token_order[order] = reference
        return token_order.reshape(rows, top_k, 5120).float().sum(1).bfloat16().cpu()

    def forward():
        return layer(
            x,
            routing,
            num_global_tokens=max_rows * dist.get_world_size(),
            max_num_tokens_per_gpu=max_rows,
        )

    expected = inputs(0)
    torch.testing.assert_close(forward().cpu(), expected, atol=0, rtol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        result = forward()
    for generation in range(1, 4):
        expected = inputs(generation)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    graph.reset()
