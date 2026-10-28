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

"""Both native head capacities consume live compact prefill rows on replay."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel_npu.ops._v41.runtime_attention_rope import selected_attention


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("rows", [1, 4])
def test_compact_attention_head_capacities(heads, rows):
    torch.manual_seed(42)
    q = (torch.randn(rows, heads, 512) * 0.1).bfloat16().to("npu")
    kv = (torch.randn(96, 512) * 0.1).bfloat16().to("npu")
    ids = (
        torch.arange(64, dtype=torch.int32, device="npu").expand(rows, -1).contiguous()
    )
    sink = torch.zeros(heads, dtype=torch.float32, device="npu")
    out = torch.empty_like(q)

    def run():
        return selected_attention(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            sink,
            512**-0.5,
            out,
            32,
            None,
            kv,
            ids,
        )

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        run()
    try:
        for offset in (0, 17):
            ids.copy_(
                torch.arange(offset, offset + 64, dtype=torch.int32).expand(rows, -1)
            )
            graph.replay()
            torch.npu.synchronize()
            keys = kv.cpu()[ids.cpu().long()].float()
            logits = torch.bmm(q.cpu().float(), keys.transpose(1, 2)) * 512**-0.5
            logits = torch.cat((logits, torch.zeros(rows, heads, 1)), -1)
            probabilities = logits.softmax(-1)[..., :-1].bfloat16().float()
            expected = torch.bmm(probabilities, keys).bfloat16()
            torch.testing.assert_close(out.cpu(), expected, atol=0.0005, rtol=0.015625)
    finally:
        torch.npu.synchronize()
        graph.reset()
