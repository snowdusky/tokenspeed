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

"""V4.1 dense loading uses the production loader and NPU GEMM."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.deepseek_v41 import (
    _merged,
    _replicated,
    v41_mxfp8_config,
)

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.mark.parametrize("merged", [False, True])
def test_fp8_dense_load_and_forward(merged):
    config = v41_mxfp8_config(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        )
    )
    with torch.device("npu"):
        layer = (
            _merged(64, [64, 32], torch.bfloat16, config, "test")
            if merged
            else _replicated(64, 96, torch.bfloat16, config, "test")
        )
    expected_weights = []
    for shard_id, rows in enumerate([64, 32] if merged else [96]):
        torch.manual_seed(123 + shard_id)
        codes = torch.randn(rows, 64).to(torch.float8_e4m3fn)
        scales = torch.randint(124, 129, ((rows + 31) // 32, 2), dtype=torch.uint8)
        args = (shard_id,) if merged else ()
        layer.weight.weight_loader(layer.weight, codes, *args)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scales, *args)
        expected_weights.append(
            (
                codes.float()
                * scales.view(torch.float8_e8m0fnu)
                .float()
                .repeat_interleave(32, 0)
                .repeat_interleave(32, 1)
            ).bfloat16()
        )
    layer.quant_method.process_weights_after_loading(layer)
    expected = torch.cat(expected_weights)
    assert layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(layer.weight.cpu(), expected, atol=0, rtol=0)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device="npu")
    actual = layer(x)[0]
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(x, expected.to("npu")), atol=0, rtol=0
    )


@pytest.mark.parametrize("tp_size,tp_rank", [(8, 0), (16, 0), (16, 1), (16, 15)])
def test_fp8_row_shard_crossing_scale_block(tp_size, tp_rank):
    from tokenspeed.runtime.layers.linear import RowParallelLinear

    config = v41_mxfp8_config(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        )
    )
    with torch.device("npu"):
        layer = RowParallelLinear(
            2304,
            64,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tuple(range(tp_size)),
            reduce_results=False,
        )
    torch.manual_seed(314)
    codes = torch.randn(64, 2304).to(torch.float8_e4m3fn)
    scales = torch.randint(124, 129, (2, 72), dtype=torch.uint8)
    layer.weight.weight_loader(layer.weight, codes)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scales)
    layer.quant_method.process_weights_after_loading(layer)
    width = 2304 // tp_size
    expected = (
        (
            codes.float()
            * scales.view(torch.float8_e8m0fnu)
            .float()
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        )
        .bfloat16()[:, tp_rank * width : (tp_rank + 1) * width]
        .contiguous()
        .to("npu")
    )
    assert layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
    x = torch.randn(3, width, dtype=torch.bfloat16, device="npu")
    torch.testing.assert_close(
        layer(x)[0], torch.nn.functional.linear(x, expected), atol=0, rtol=0
    )


@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("backend", ["ascend_v41_bf16", "ascend_v41_mxfp4"])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_mxfp4_moe_load_and_acl_replay(rows, backend, ep_size):
    from tokenspeed.runtime.layers.moe import utils
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
    from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config

    previous = utils.MOE_BACKEND
    utils.MOE_BACKEND = utils.MoeBackend(backend)
    try:
        with torch.device("npu"):
            with torch.autocast("npu", enabled=False):
                previous_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    layer = MoELayer(
                        top_k=2,
                        num_experts=4,
                        hidden_size=256,
                        intermediate_size=128,
                        quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
                        layer_index=0,
                        prefix="test",
                        tp_rank=0,
                        tp_size=1,
                        ep_rank=ep_size - 1,
                        ep_size=ep_size,
                        activation="swiglu",
                        swiglu_limit=7.0,
                        routing_mode="precomputed_topk",
                    )
                finally:
                    torch.set_default_dtype(previous_dtype)
    finally:
        utils.MOE_BACKEND = previous
    torch.manual_seed(42)
    for p in (layer.w13_weight, layer.w2_weight):
        p.data.copy_(torch.randint(0, 256, p.shape, dtype=torch.uint8))
    for p in (layer.w13_weight_scale, layer.w2_weight_scale):
        p.data.fill_(122)
    from tokenspeed_kernel_npu.ops.mxfp import dequantize_mxfp_weight

    references = []
    for weight, scale in (
        (layer.w13_weight, layer.w13_weight_scale),
        (layer.w2_weight, layer.w2_weight_scale),
    ):
        references.append(
            dequantize_mxfp_weight(
                weight.flatten(0, 1), scale.flatten(0, 1), 4, (1, 32)
            ).reshape(weight.shape[0], weight.shape[1], weight.shape[2] * 2)
        )
    w13_reference, w2_reference = references
    layer.process_weights_after_loading(layer)
    if backend == "ascend_v41_mxfp4":
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w2_weight.dtype == torch.uint8
    x = torch.randn(rows, 256, dtype=torch.bfloat16, device="npu")
    ids = (
        torch.tensor([[0, 3]], dtype=torch.int32, device="npu")
        .expand(rows, -1)
        .contiguous()
    )
    weights = (
        torch.tensor([[0.3, 0.7]], dtype=torch.float32, device="npu")
        .expand(rows, -1)
        .contiguous()
    )
    topk = StandardTopKOutput(weights, ids, None)

    def forward():
        return layer(x, topk, num_global_tokens=rows, max_num_tokens_per_gpu=rows)

    # Independent route-by-route reference retains the pre-down cast boundary.
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert, route_weight in ((0, weights[0, 0]), (3, weights[0, 1])):
        local_expert = expert - (ep_size - 1) * (4 // ep_size)
        if not 0 <= local_expert < 4 // ep_size:
            continue
        gu = torch.nn.functional.linear(x, w13_reference[local_expert]).float()
        gate, up = gu.chunk(2, -1)
        gate = gate.clamp_max(7.0)
        up = up.clamp(-7.0, 7.0)
        hidden = ((gate / (1 + torch.exp(-gate))) * up * route_weight).bfloat16()
        expected += torch.nn.functional.linear(
            hidden, w2_reference[local_expert]
        ).float()
    torch.testing.assert_close(
        forward(), expected.bfloat16(), atol=0.015625, rtol=0.0078125
    )
    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = forward()
    for seed in (100, 101):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        ids.copy_(torch.randint(0, 4, (rows, 1), device="npu").expand_as(ids))
        if ep_size == 2 and seed == 101:
            ids.zero_()  # All routes are remote on this rank.
        expected = forward()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("width", [32, 96, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_engram_byte_table_and_acl_replay(width, strided):
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding

    mapping = Mapping(world_size=1, rank=0, attn_tp_size=1)
    embed = RowShardedEngramEmbedding(
        37,
        width,
        mapping,
        device=torch.device("npu"),
        host_table=False,
        host_layout="gpu",
        layer_id=0,
    )
    codes = torch.randn(37, width).to(torch.float8_e4m3fn)
    scales = torch.randint(124, 129, (37, width // 32), dtype=torch.uint8)
    embed.weight_loader(embed.weight, codes)
    embed.weight_loader(embed.scale, scales)
    decoded = (
        codes.float()
        * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
    ).bfloat16()
    ids = torch.tensor([[0, 36, -1], [4, 17, 37]], device="npu")
    if strided:
        ids = ids.repeat_interleave(2, dim=1)[:, ::2]
    expected = decoded[ids.cpu().clamp(0, 36)].masked_fill(
        ((ids.cpu() < 0) | (ids.cpu() >= 37)).unsqueeze(-1), 0
    )
    torch.testing.assert_close(embed(ids).cpu(), expected, atol=0, rtol=0)
    for _ in range(3):
        embed(ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = embed(ids)
    ids.copy_(torch.tensor([[2, -1, 11], [36, 36, 5]], device="npu"))
    graph.update(cpu_update_input=[{"actual_seq_lengths_kv": [100, 101]}])
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output, embed(ids), atol=0, rtol=0)


@pytest.mark.parametrize("zero_gate", [False, True])
@torch.inference_mode()
def test_engram_gate_matches_cpu_and_captures(zero_gate):
    from test.runtime.test_deepseek_v41_engram import _model, _weights

    torch.manual_seed(41)
    model = _model("npu", None)
    codes, scales = _weights(72, 32)
    model.embed.weight_loader(model.embed.weight, codes)
    model.embed.weight_loader(model.embed.scale, scales)
    model.wkv.weight.copy_(torch.randn_like(model.wkv.weight) * 0.03)
    model.q_weight.copy_(torch.randn_like(model.q_weight))
    model.k_weight.copy_(torch.randn_like(model.k_weight))
    if zero_gate:
        model.q_weight.zero_()
    ids = torch.randint(72, (2, 3, 6), device="npu")
    hidden = torch.randn(2, 3, 4, 32, dtype=torch.bfloat16, device="npu")
    mask = torch.tensor([[True, True, False], [True, False, True]], device="npu")

    def check(actual):
        # Isolate the gate from GEMM rounding: use the actual projected BF16
        # KV, then evaluate the full normalized-dot gate independently on CPU.
        kv = model.wkv(model.embed(ids).flatten(-2))[0].cpu().float()
        key = kv[..., :128].reshape(2, 3, 4, 32)
        value = kv[..., 128:]
        h = hidden.cpu().float()
        qnorm = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + model.eps)
        knorm = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + model.eps)
        dot = (
            qnorm * model.q_weight.cpu().float() * knorm * model.k_weight.cpu().float()
        ).sum(-1) / 32**0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate.masked_fill_(~mask.cpu().unsqueeze(-1), 0)
        expected = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).bfloat16()
        torch.testing.assert_close(
            actual.cpu(), expected, atol=0.0078125, rtol=0.0078125
        )
        torch.testing.assert_close(actual[~mask], hidden[~mask], atol=0, rtol=0)

    for _ in range(3):
        eager = model(hidden, ids, mask)
    check(eager)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        captured = model(hidden, ids, mask)
    hidden.normal_()
    ids.copy_(torch.randint_like(ids, 72))
    mask.logical_not_()
    graph.replay()
    torch.npu.synchronize()
    check(captured)
    torch.testing.assert_close(captured, model(hidden, ids, mask), atol=0, rtol=0)


@torch.inference_mode()
def test_engram_lookup_exceeds_launch_grid_limit():
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding

    model = RowShardedEngramEmbedding(
        4,
        32,
        Mapping(world_size=1, rank=0, attn_tp_size=1),
        device=torch.device("npu"),
        host_table=False,
        host_layout="gpu",
        layer_id=0,
    )
    codes = torch.arange(128).reshape(4, 32).float().to(torch.float8_e4m3fn)
    scales = torch.tensor([[125], [126], [127], [128]], dtype=torch.uint8)
    model.weight_loader(model.weight, codes)
    model.weight_loader(model.scale, scales)
    # A long prefill expands each token into several Engram hash lookups.
    ids = torch.arange(65538, dtype=torch.int64) % 6 - 1
    decoded = (codes.float() * scales.view(torch.float8_e8m0fnu).float()).bfloat16()
    expected = decoded[ids.clamp(0, 3)].masked_fill(
        ((ids < 0) | (ids >= 4))[:, None], 0
    )
    actual = model(ids.to("npu"))
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
