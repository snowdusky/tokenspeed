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

"""Direct packed-page attention, compact prefill, and live ACL replay inputs."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import torch_dsv41_cache_quantize
from tokenspeed_kernel.ops.attention import dsv41 as ops
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.plugins import discover_plugins

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.fixture(scope="module", autouse=True)
def native_providers():
    discover_plugins()


def reference(q, parts, sink):
    keys, masks = [], []
    for rows, slots, lengths in parts:
        valid = (
            (slots >= 0)
            & (slots < len(rows))
            & (torch.arange(slots.shape[1])[None] < lengths[:, None])
        )
        keys.append(rows[slots.clamp(0, len(rows) - 1)].float())
        masks.append(valid)
    kv, valid = torch.cat(keys, 1), torch.cat(masks, 1)
    logits = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
    logits.masked_fill_(~valid[:, None], -torch.inf)
    if sink is not None:
        logits = torch.cat((logits, sink[None, :, None].expand(q.shape[0], -1, 1)), -1)
        kv = torch.cat((kv, torch.zeros(q.shape[0], 1, 512)), 1)
    lse = logits.logsumexp(-1)
    probs = logits.softmax(-1).nan_to_num()
    return torch.bmm(probs, kv).bfloat16(), lse


@pytest.mark.parametrize("global_enabled", [False, True])
@pytest.mark.parametrize("use_sink", [False, True])
@pytest.mark.parametrize("byte_stride", [1, 2])
def test_packed_attention_and_replay(global_enabled, use_sink, byte_stride):
    torch.manual_seed(69)
    qcpu = torch.randn(4, 2, 512, dtype=torch.bfloat16)
    slots = torch.tensor(
        [
            [0, 63, 64, 127, 128, -1, 320],
            [129, 128, 127, 3, 2, 1, 0],
            [-1] * 7,
            [0, 1, 2, 3, 4, 5, 6],
        ],
        dtype=torch.int32,
    )
    lengths = torch.tensor([7, 4, 7, 0], dtype=torch.int32)
    qs = qcpu.to("npu")
    sink = torch.tensor([0.5, -torch.inf]) if use_sink else None
    sn = None if sink is None else sink.to("npu")
    parts, caches, ns, nl = [], [], [], []
    for fmt, width in [("swa", 528)] + ([("global", 288)] if global_enabled else []):
        raw = torch.randn(192, 512, dtype=torch.bfloat16)
        _, decoded = torch_dsv41_cache_quantize(raw, fmt)
        # The field view has gaps between bytes and between physical pages.
        storage = torch.zeros(
            3, 64, width * byte_stride, dtype=torch.uint8, device="npu"
        )
        cache = storage[..., ::byte_stride]
        ops.cache_scatter(raw.to("npu"), cache, torch.arange(192, device="npu"), fmt)
        caches.append(cache)
        ns.append(slots.to("npu"))
        nl.append(lengths.to("npu"))
        parts.append((decoded.bfloat16(), slots.clone(), lengths.clone()))
    out = torch.empty_like(qs)
    lse = None if use_sink else torch.empty(qs.shape[:2], device="npu")

    def run():
        return ops.selected_attention(
            qs,
            caches[0],
            ns[0],
            nl[0],
            caches[1] if global_enabled else None,
            ns[1] if global_enabled else None,
            nl[1] if global_enabled else None,
            sn,
            512**-0.5,
            out,
            2,
            None,
            None,
            None,
            lse=lse,
        )

    for _ in range(3):
        run()
    expected, expected_lse = reference(qcpu, parts, sink)
    torch.testing.assert_close(out.cpu(), expected, atol=0.00390625, rtol=0.0078125)
    if lse is not None:
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-5, rtol=2e-5)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        run()
    qs.copy_(qcpu * 0.5)
    for i in range(len(parts)):
        rows, _, _ = parts[i]
        newslots = slots.flip(1).contiguous()
        newlengths = torch.tensor([3, 7, 0, 7], dtype=torch.int32)
        ns[i].copy_(newslots)
        nl[i].copy_(newlengths)
        parts[i] = rows, newslots, newlengths
    graph.replay()
    torch.npu.synchronize()
    captured = out.cpu()
    expected, _ = reference(qcpu * 0.5, parts, sink)
    torch.testing.assert_close(captured, expected, atol=0.00390625, rtol=0.0078125)
    run()
    torch.testing.assert_close(out.cpu(), captured, atol=0, rtol=0)


@pytest.mark.parametrize("use_sink", [False, True])
@pytest.mark.parametrize("cache_handles", [False, True])
def test_compact_attention(use_sink, cache_handles):
    torch.manual_seed(17)
    q = torch.randn(3, 4, 512, dtype=torch.bfloat16)
    kv = torch.randn(20, 1, 512, dtype=torch.bfloat16)
    ids = torch.tensor(
        [[0, 1, 4, 19], [-1, -1, -1, -1], [0, 20, -1, 5]], dtype=torch.int32
    )
    sink = torch.randn(4) if use_sink else None
    expected, expected_lse = reference(q, [(kv[:, 0], ids, torch.full((3,), 4))], sink)
    lse = None if use_sink else torch.empty(3, 4, device="npu")
    actual = ops.selected_attention(
        q.to("npu"),
        (
            torch.zeros(1, 64, 528, dtype=torch.uint8, device="npu")
            if cache_handles
            else None
        ),
        None,
        None,
        (
            torch.zeros(1, 64, 288, dtype=torch.uint8, device="npu")
            if cache_handles
            else None
        ),
        None,
        None,
        None if sink is None else sink.to("npu"),
        512**-0.5,
        None,
        1,
        None,
        kv.to("npu"),
        ids.to("npu"),
        lse=lse,
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0.00390625, rtol=0.0078125)
    if lse is not None:
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-5, rtol=2e-5)
