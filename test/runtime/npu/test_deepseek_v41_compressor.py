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

"""Real compressor and LCM metadata under native NPU graph replay."""

from test.runtime.test_deepseek_v41_cache import _backend, _tables

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41TextConfig
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.deepseek_v41_ascend910 import (
    Ascend910Compressor as DeepseekV41Compressor,
)

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@torch.inference_mode()
def test_compressor_uses_live_metadata_in_acl_graph():
    torch.manual_seed(19)
    backend = _backend("npu", 2)
    tables = _tables("npu")
    config = DeepseekV41TextConfig(
        hidden_size=128, head_dim=512, compress_ratios=[0, 0, 2], rms_norm_eps=1e-6
    )
    with torch.device("npu"):
        layer = DeepseekV41Compressor(config, 2, "compressor")
    layer.wkv_wgate.weight.normal_(std=0.03)
    layer.norm.weight.fill_(1.0)
    x = torch.randn(2, 128, dtype=torch.bfloat16, device="npu")
    tail = backend.cache_pool.compressor_tail(2)

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

    def run():
        meta = backend.query_metadata(ForwardMode.DECODE)
        return layer(
            x, 2, meta.positions, meta.request_indices, backend, ForwardMode.DECODE
        )

    refresh([9, 10], False)
    tail.zero_()
    projected = torch.nn.functional.linear(x.float(), layer.wkv_wgate.weight.float())
    content, scores = projected.chunk(2, -1)
    expected = backend.compress(
        2, content, scores, ForwardMode.DECODE, layer.norm.weight, 1e-6
    )
    expected_tail = tail.clone()
    tail.zero_()
    actual = run()
    torch.testing.assert_close(actual[0], expected[0], atol=0.0078125, rtol=0.0078125)
    torch.testing.assert_close(tail, expected_tail, atol=0, rtol=0)
    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        captured = run()
    for lengths in ([10, 11], [11, 12], [12, 13]):
        refresh(lengths, True)
        x.normal_()
        tail.zero_()
        eager = run()[0].clone()
        eager_tail = tail.clone()
        tail.zero_()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured[0], eager, atol=0, rtol=0)
        torch.testing.assert_close(tail, eager_tail, atol=0, rtol=0)
