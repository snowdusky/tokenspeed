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

"""Run with torchrun and pytest on Ascend to exercise real output-group TP."""

import os
from test.runtime.test_deepseek_v41_model import _config

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm import grouped_bf16_projection
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.models.deepseek_v41_ascend910 import (
    Ascend910Attention as DeepseekV41Attention,
)

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.fixture(scope="module")
def mapping():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2 or world % 2:
        pytest.skip("requires an even number of torchrun workers")
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    layout = Mapping(world_size=world, rank=rank, attn_tp_size=world)
    pg_manager.init_distributed(layout, backend="hccl", timeout=300)
    pg_manager.init_process_group(layout.attn.tp_group)
    yield layout
    dist.destroy_process_group()


@pytest.mark.parametrize("split_group", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 32])
@torch.inference_mode()
def test_output_group_shard_and_acl_replay(mapping, split_group, rows):
    config = _config()
    config.num_attention_heads = 2 * mapping.world_size
    config.o_groups = mapping.world_size // (2 if split_group else 1)
    config.o_lora_rank = 64
    config.index_n_heads = mapping.world_size
    with torch.device("npu"):
        attn = DeepseekV41Attention(
            config, mapping, 0, 20, None, "layers.0.attn", aux_stream=None
        )
    torch.manual_seed(91)
    heads_per_group = config.num_attention_heads // config.o_groups
    weight = torch.randn(
        config.o_groups * config.o_lora_rank,
        heads_per_group * config.head_dim,
        dtype=torch.bfloat16,
    )
    attn.wo_a.weight.weight_loader(attn.wo_a.weight, weight)
    output_tp = 2 if split_group else 1
    group_id = mapping.attn.tp_rank // output_tp
    output_slice = slice(
        (mapping.attn.tp_rank % output_tp) * attn.o_lora_rank,
        (mapping.attn.tp_rank % output_tp + 1) * attn.o_lora_rank,
    )
    full_weight = weight.reshape(config.o_groups, config.o_lora_rank, -1).to("npu")
    local = torch.empty(
        rows, attn.n_local_heads, config.head_dim, dtype=torch.bfloat16, device="npu"
    )

    def forward():
        grouped = attn._group_attention_output(local)
        return grouped_bf16_projection(
            grouped, attn.wo_a.weight.reshape(1, attn.o_lora_rank, -1), None, None
        )

    def update(seed):
        torch.manual_seed(seed)
        full = torch.randn(
            rows, config.num_attention_heads, config.head_dim, dtype=torch.bfloat16
        ).to("npu")
        start = mapping.attn.tp_rank * attn.n_local_heads
        local.copy_(full[:, start : start + attn.n_local_heads])
        group_input = full.reshape(rows, config.o_groups, -1)[:, group_id].contiguous()
        torch.testing.assert_close(
            attn._group_attention_output(local)[:, 0], group_input, atol=0, rtol=0
        )
        expected = torch.nn.functional.linear(group_input, full_weight[group_id])[
            :, output_slice
        ]
        return expected

    expected = update(93)
    for _ in range(3):
        torch.testing.assert_close(forward()[:, 0], expected, atol=1e-5, rtol=0.0078125)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        actual = forward()
    expected = update(94)
    eager = forward().clone()
    # The full-width reference GEMM and sharded BMM may use different FP32
    # reduction orders; require one BF16 ULP, and exact same-shape replay.
    torch.testing.assert_close(eager[:, 0], expected, atol=1e-5, rtol=0.0078125)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, eager, atol=0, rtol=0)
