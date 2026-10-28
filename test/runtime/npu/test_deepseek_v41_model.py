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

"""Full V4.1 layer topology with reduced weights and real NPU operators."""

from dataclasses import replace
from test.runtime.test_deepseek_v41_cache import _backend, _extend, _tables
from test.runtime.test_deepseek_v41_engram import _Tokenizer
from test.runtime.test_deepseek_v41_model import _config, _ctx, _initialize, _quant

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41TextConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.layers.moe import utils
from tokenspeed.runtime.models.deepseek_v41_ascend910 import (
    Ascend910Experts,
)
from tokenspeed.runtime.models.deepseek_v41_ascend910 import (
    Ascend910Model as DeepseekV41Model,
)

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@torch.inference_mode()
def test_backbone_prefill_and_live_decode_acl_replay(tmp_path):
    torch.manual_seed(71)
    settings = vars(_config()).copy()
    settings.update(
        head_dim=512,
        qk_rope_head_dim=64,
        num_attention_heads=2,
        o_groups=2,
        q_lora_rank=128,
        index_n_heads=2,
        index_head_dim=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
    )
    config = DeepseekV41TextConfig(**settings)
    mapping = Mapping(world_size=1, rank=0, attn_tp_size=1)
    backend = _backend("npu", 2)
    tables = _tables("npu")
    previous_backend = utils.MOE_BACKEND
    previous_dtype = torch.get_default_dtype()
    try:
        utils.MOE_BACKEND = utils.MoeBackend("ascend_v41_bf16")
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("npu"):
            model = DeepseekV41Model(config, mapping, _quant(), "model", False, "gpu")
    finally:
        utils.MOE_BACKEND = previous_backend
        torch.set_default_dtype(previous_dtype)
    _initialize(model)
    model.initialize_engram(_Tokenizer())
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.quant_method.process_weights_after_loading(module)
        if isinstance(module, Ascend910Experts):
            module.process_weights_after_loading(module)

    arena = backend.cache_pool.arena.buffer
    arena.zero_()
    meta = _extend(backend, tables, [4, 4], [0, 0], [0, 0], [4, 4])
    ids = torch.tensor([0, 3, 4, 1, 3, 1, 0, 4], device="npu")
    previous = torch.tensor(
        [
            [-1, -1, -1],
            [0, -1, -1],
            [3, 0, -1],
            [4, 3, 0],
            [-1, -1, -1],
            [3, -1, -1],
            [1, 3, -1],
            [0, 1, 3],
        ],
        device="npu",
    )
    mask = torch.ones(8, dtype=torch.bool, device="npu")
    ctx = replace(
        _ctx(backend, 8, ForwardMode.EXTEND),
        bs=2,
        num_extends=2,
        output_layout=ForwardOutputLayout(2, 2, 0, 1),
    )
    prefill = model(
        ids,
        meta.positions,
        ctx,
        None,
        None,
        engram_hash_ids=model.engram_hash(ids, previous, mask),
        engram_token_mask=mask,
        image_mask=None,
    )[0]
    assert prefill.shape == (8, config.hidden_size)
    assert torch.isfinite(prefill).all()
    prefetched = arena.clone()

    def refresh(lengths, replay):
        backend.refresh_decode_metadata(
            2,
            2,
            torch.tensor([0, 1], device="npu"),
            torch.tensor(lengths, device="npu"),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=replay,
        )

    refresh([5, 5], False)
    ctx = replace(
        _ctx(backend, 2, ForwardMode.DECODE),
        bs=2,
        output_layout=ForwardOutputLayout(0, 0, 2, 1),
    )
    ids = torch.tensor([0, 3], device="npu")
    previous = torch.tensor([[1, 4, 3], [4, 0, 1]], device="npu")
    mask = torch.ones(2, dtype=torch.bool, device="npu")
    hashes = model.engram_hash(ids, previous, mask)

    def run():
        return model(
            ids,
            backend.query_metadata(ForwardMode.DECODE).positions,
            ctx,
            None,
            None,
            engram_hash_ids=hashes,
            engram_token_mask=mask,
            image_mask=None,
        )[0]

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        captured = run()
    arena.copy_(prefetched)
    for step, lengths in enumerate(([5, 5], [6, 6], [7, 0])):
        refresh(lengths, True)
        ids.copy_(torch.tensor([step, 4 - step], device="npu"))
        mask.copy_(torch.tensor([True, step != 1], device="npu"))
        hashes.copy_(model.engram_hash(ids, previous, mask))
        before = arena.clone()
        eager = run().clone()
        expected_cache = arena.clone()
        arena.copy_(before)
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        torch.npu.synchronize()
        assert torch.isfinite(captured).all()
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
        torch.testing.assert_close(arena, expected_cache, atol=0, rtol=0)
        previous.copy_(torch.cat((ids[:, None], previous[:, :2]), dim=1))

    _check_forward_runner(model, backend, tables, config, previous, hashes, tmp_path)


def _check_forward_runner(model, backend, tables, config, previous, hashes, tmp_path):
    import torch.distributed as dist

    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    dist.init_process_group(
        "hccl", init_method=(tmp_path / "hccl").as_uri(), rank=0, world_size=1
    )
    ib = InputBuffers(
        max_bs=2, max_num_tokens=8, state_write_padding_pool_index=3, device="npu"
    )
    settings = ModelExecutorConfig(
        max_req_pool_size=3,
        output_length=1,
        enforce_eager=False,
        prefix_granularity=256,
        max_num_seqs=2,
        chunked_prefill_size=8,
        vocab_size=config.vocab_size,
        context_len=512,
        physical_context_len=512,
        device="npu",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=[2],
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=2,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=None,
        input_logprob_chunk_tokens=8,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        grammar_backend="none",
        overlap_schedule_depth=1,
    )
    head = torch.randn(
        config.hidden_size, config.vocab_size, dtype=torch.bfloat16, device="npu"
    )

    def forward(*, bs, ctx, sampling_info):
        hidden = model(
            ib.input_ids_buf[:bs],
            backend.query_metadata(ctx.forward_mode).positions,
            ctx,
            None,
            None,
            engram_hash_ids=hashes[:bs],
            engram_token_mask=ib.active_request_mask_buf[:bs],
            image_mask=None,
        )[0]
        logits = (hidden @ head).float()
        tokens = logits.argmax(-1)
        logprobs = logits.log_softmax(-1).gather(1, tokens[:, None]).flatten()
        return tokens, torch.ones(bs, dtype=torch.int32, device="npu"), logprobs, None

    runner = ForwardStepRunner(forward, backend, backend.cache_pool, ib, settings)
    try:
        runner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
        for step, bs in enumerate((2, 1, 2)):
            ib.input_ids_buf[:2].copy_(torch.tensor([step, 4 - step], device="npu"))
            ib.seq_lens_buf[:2].fill_(8 + step)
            ib.req_pool_indices_buf[:2].copy_(torch.tensor([0, 1], device="npu"))
            ib.active_request_mask_buf[:2].fill_(True)
            ib.active_request_mask_buf[bs:2].zero_()
            hashes.copy_(
                model.engram_hash(
                    ib.input_ids_buf[:2], previous, ib.active_request_mask_buf[:2]
                )
            )
            ctx = replace(
                _ctx(backend, bs, ForwardMode.DECODE),
                bs=bs,
                output_layout=ForwardOutputLayout(0, 0, bs, 1),
            )
            sampling = SamplingBatchInfo(
                req_pool_indices=ib.req_pool_indices_buf[:bs],
                vocab_size=config.vocab_size,
                device="npu",
            )
            before = backend.cache_pool.arena.buffer.clone()
            actual = runner(
                bs,
                ctx,
                sampling,
                extend_with_prefix=False,
                extend_prefix_lens=ib.extend_prefix_lens_buf[:0],
                extend_prefix_lens_cpu=ib.extend_prefix_lens_cpu[:0],
                extend_seq_lens=ib.extend_seq_lens_buf[:0],
                extend_seq_lens_cpu=ib.extend_seq_lens_cpu[:0],
                extend_replay_lens_cpu=ib.extend_replay_lens_cpu[:0],
                extend_prompt_lens_cpu=ib.extend_prompt_lens_cpu[:0],
                block_tables_cpu={},
                settle=lambda: None,
                block_tables={key: table[:bs] for key, table in tables.items()},
            )
            actual = tuple(t.clone() if t is not None else None for t in actual)
            torch.npu.synchronize()
            assert runner.last_graph_bs == 2
            expected_cache = backend.cache_pool.arena.buffer.clone()
            backend.cache_pool.arena.buffer.copy_(before)
            # Replay leaves the same padded metadata live. Compare the model
            # at that geometry, so GEMM batch-shape changes cannot mask bugs.
            padded = replace(
                ctx,
                bs=2,
                input_num_tokens=2,
                output_layout=ForwardOutputLayout(0, 0, 2, 1),
            )
            expected = forward(bs=2, ctx=padded, sampling_info=sampling)
            torch.npu.synchronize()
            for result, ref in zip(actual[:3], expected[:3], strict=True):
                torch.testing.assert_close(result, ref[:bs], atol=0, rtol=0)
            torch.testing.assert_close(
                backend.cache_pool.arena.buffer, expected_cache, atol=0, rtol=0
            )
    finally:
        runner.release_graphs()
        dist.destroy_process_group()
