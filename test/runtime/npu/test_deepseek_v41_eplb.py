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
    try:
        yield group
    finally:
        global_server_args_dict["mapping"] = previous
        dist.destroy_process_group()


@torch.no_grad()
def test_v41_eplb_loaded_slots_migration_and_graph(group):
    from transformers import PretrainedConfig

    from tokenspeed.runtime.layers.moe import utils
    from tokenspeed.runtime.layers.moe.loader import (
        ExpertCheckpointSchema,
        build_moe_checkpoint_loader,
    )
    from tokenspeed.runtime.models.deepseek_v41_ascend910 import (
        Ascend910MoE as DeepseekV41MoE,
    )
    from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
    from tokenspeed.runtime.moe.expert_load_rows import ExpertLoadRowMask
    from tokenspeed.runtime.moe.expert_location import (
        ExpertLocationMetadata,
        get_global_expert_location_metadata,
        set_global_expert_location_metadata,
    )
    from tokenspeed.runtime.moe.expert_location_updater import (
        ExpertLocationUpdater,
        _all_to_all_ep,
    )
    from tokenspeed.runtime.moe.expert_rebalance import plan_slot_moves
    from tokenspeed.runtime.moe.placement_maps import compute_placement_maps
    from tokenspeed.runtime.utils.env import global_server_args_dict

    rank, world = dist.get_rank(), dist.get_world_size()
    mapping = global_server_args_dict["mapping"]
    logical = 384
    physical = logical + world
    initial = torch.arange(physical).remainder(logical).unsqueeze(0)
    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        initial.to("npu"),
        logical,
        ep_size=world,
        ep_rank=rank,
        ep_rank_nodes=tuple(
            r // int(os.environ["LOCAL_WORLD_SIZE"]) for r in range(world)
        ),
    )
    placement.enable_load_recording(mapping)
    placement.load_rows = ExpertLoadRowMask.for_all_to_all(mapping)
    placement.reserve_load_rows(16)
    quant = {
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
    config = PretrainedConfig(
        n_routed_experts=logical,
        num_experts_per_tok=6,
        hidden_size=5120,
        moe_intermediate_size=2304,
        n_shared_experts=None,
        hidden_act="silu",
        swiglu_limit=10.0,
        num_hash_layers=0,
        vocab_size=32,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        expert_dtype="int4",
        expert_quantization_config=quant,
    )
    previous = (
        utils.MOE_BACKEND,
        utils.ALL2ALL_BACKEND,
        torch.get_default_dtype(),
        get_global_expert_location_metadata(),
        global_server_args_dict["ep_dispatch_algorithm"],
    )
    utils.MOE_BACKEND = utils.MoeBackend("ascend_v41_w4a4")
    utils.ALL2ALL_BACKEND = utils.All2AllBackend.ASCEND_MC2
    global_server_args_dict["ep_dispatch_algorithm"] = "static"
    set_global_expert_location_metadata(placement)
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("npu"):
            block = DeepseekV41MoE(config, mapping, None, 0, "model.layers.0.ffn", None)
    finally:
        utils.MOE_BACKEND, utils.ALL2ALL_BACKEND, dtype, old_placement, algorithm = (
            previous
        )
        torch.set_default_dtype(dtype)
        set_global_expert_location_metadata(old_placement)
        global_server_args_dict["ep_dispatch_algorithm"] = algorithm
    layer = block.experts
    assert layer.num_experts == physical
    params = {
        "model.layers.0.ffn.experts." + name: value
        for name, value in layer.named_parameters()
    }
    loader = build_moe_checkpoint_loader(
        params_dict=params,
        expert_schema=ExpertCheckpointSchema(
            gate_proj_name="w1",
            up_proj_name="w3",
            down_proj_name="w2",
            gate_up_fused_name=None,
            extra_names={},
        ),
        num_experts=physical,
        ep_rank=rank,
        ep_size=world,
        expert_placement=placement,
    )
    for expert in sorted(set(placement.local_slot_logical_experts(0, rank))):
        gen = torch.Generator().manual_seed(9000 + expert)
        for projection, shape in (
            ("w1", (2304, 640)),
            ("w3", (2304, 640)),
            ("w2", (5120, 288)),
        ):
            weight = torch.randint(
                0, 2**31 - 1, shape, generator=gen, dtype=torch.int32
            )
            scale = torch.full(
                (shape[0],), 0.001 + expert * 0.000001, dtype=torch.float32
            )
            for suffix, value in (("weight", weight), ("weight_scale", scale)):
                loader.load(
                    f"model.layers.0.ffn.experts.{expert}.{projection}.{suffix}", value
                )
    for stage in (1, 2):
        transform = layer.w1_trans if stage == 1 else layer.w2_trans
        for parameter in (transform.matrix_left, transform.matrix_right):
            parameter.weight_loader(
                parameter, torch.eye(parameter.shape[0], dtype=torch.bfloat16)
            )
        clip = layer.act_quantizer_w1 if stage == 1 else layer.act_quantizer_w2
        clip.clip_factor_a.weight_loader(clip.clip_factor_a, torch.ones(1))
    layer.process_weights_after_loading(layer)
    block.gate.weight.zero_()
    block.gate.e_score_correction_bias.zero_()
    block.gate.e_score_correction_bias[:6].copy_(torch.arange(6, 0, -1, device="npu"))
    tensors = block.get_moe_routed_weights()
    pointers = [t.data_ptr() for t in tensors]

    # Exact packed bytes and scale bits sampled across every physical slot.
    # This checks migration independently of the BF16 output tolerance.
    def slot_samples():
        samples = []
        for tensor in tensors:
            bits = tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor
            flat = bits.reshape(physical // world, -1)
            indices = torch.linspace(0, flat.shape[1] - 1, 64, device="npu").long()
            samples.append(flat[:, indices].to(torch.int64))
        return torch.cat(samples, dim=1).contiguous()

    local_samples = slot_samples()
    all_samples = torch.empty(
        (physical, local_samples.shape[1]), dtype=torch.int64, device="npu"
    )
    dist.all_gather_into_tensor(all_samples, local_samples, group=group)
    # The initial placement starts with one copy of all logical experts.
    logical_samples = all_samples[:logical].clone()
    updater = ExpertLocationUpdater(
        placement,
        {0: tensors},
        process_group=group,
        peer_ranks=tuple(range(world)),
        ep_rank=rank,
        all_to_all_ep=_all_to_all_ep(block),
        num_groups=None,
        num_nodes=world // int(os.environ["LOCAL_WORLD_SIZE"]),
    )
    updater.prewarm()
    x = (
        (
            torch.randn(16, 5120, generator=torch.Generator().manual_seed(rank + 3))
            * 0.05
        )
        .bfloat16()
        .to("npu")
    )

    def forward():
        routes = block._compute_topk_output(x, None)
        return block._forward_routed_experts(x, routes, world * 16, 16, True)

    for _ in range(2):
        baseline = forward()
    torch.npu.synchronize()
    baseline = baseline.clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replay_output = forward()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(replay_output, baseline, atol=0, rtol=0)
    # Count only eight live rows, not the graph's sixteen-row capacity.
    placement.reset_load()
    placement.load_rows.mask[8:].fill_(False)
    graph.replay()
    torch.npu.synchronize()
    snapshot = placement.snapshot_load()
    assert int(snapshot.physical_count.sum()) == 8 * 6
    placement.load_rows.clear()
    counts = torch.zeros((1, logical), dtype=torch.int64)
    counts[0, :6] = 1000
    balanced, _ = compute_placement_maps(
        counts,
        num_physical_experts=physical,
        ep_size=world,
        num_groups=None,
        num_nodes=world // int(os.environ["LOCAL_WORLD_SIZE"]),
        algorithm=EplbAlgorithm.deepseek,
    )

    for target in (balanced, balanced.roll(13, dims=1), initial):
        moves = plan_slot_moves(
            placement.physical_to_logical_map_cpu[0],
            target[0],
            ep_rank=rank,
            ep_rank_nodes=placement.ep_rank_nodes,
            num_local=physical // world,
        )
        updater.apply([0], target, {0: moves})
        graph.replay()
        torch.npu.synchronize()
        # MC2 uses rank-order combine: relocation may change BF16 rounding.
        torch.testing.assert_close(replay_output, baseline, atol=2e-4, rtol=0.016)
        assert [t.data_ptr() for t in tensors] == pointers
        local_ids = target[
            0, rank * (physical // world) : (rank + 1) * (physical // world)
        ]
        torch.testing.assert_close(
            slot_samples(), logical_samples[local_ids.to("npu").long()], atol=0, rtol=0
        )
    torch.testing.assert_close(replay_output, baseline, atol=0, rtol=0)
