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

"""Reference-rounded index scoring and query tiling on Ascend."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import torch_dsv41_cache_quantize
from tokenspeed_kernel.ops.attention import dsv41 as ops
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


def inputs(rows, heads=2, history=192, weight_dtype=torch.bfloat16):
    torch.manual_seed(71)
    q = torch.randn(rows, heads, 128, dtype=torch.bfloat16)
    weights = (torch.randn(rows, heads).bfloat16() * 0.04).to(weight_dtype)
    k = torch.randn(history, 128, dtype=torch.bfloat16)
    cache = torch.zeros((history + 63) // 64, 64, 68, dtype=torch.uint8, device="npu")
    ops.cache_scatter(k.to("npu"), cache, torch.arange(history, device="npu"), "index")
    _, qdecoded = torch_dsv41_cache_quantize(q.flatten(0, 1), "index")
    _, kdecoded = torch_dsv41_cache_quantize(k, "index")
    qdecoded = qdecoded.bfloat16().reshape_as(q)
    dots = torch.bmm(qdecoded, kdecoded.bfloat16().T.expand(rows, -1, -1))
    scores = (dots.relu() * weights[:, :, None]).sum(1)
    return q, weights, cache, scores


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_index_score(weight_dtype):
    q, w, cache, _ = inputs(3)
    w = w.to(weight_dtype)
    slots = torch.tensor(
        [[0, 63, 64, 191, -1, 192], [2, 3, 4, 5, 6, 7], [-1] * 6], dtype=torch.int32
    )
    actual = ops.index_score(
        q.to("npu"), w.to("npu"), cache, slots.to("npu"), None, None
    )
    _, qd = torch_dsv41_cache_quantize(q.flatten(0, 1), "index")
    kd = ops.cache_gather(cache, slots.to("npu"), "index", None).cpu()
    expected = (
        torch.bmm(qd.reshape_as(q).bfloat16(), kd.transpose(1, 2)).relu()
        * w[:, :, None]
    ).sum(1)
    expected.masked_fill_((slots < 0) | (slots >= 192), -torch.inf)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_topk_query_chunk_and_live_lengths(weight_dtype):
    q, w, cache, scores = inputs(33, weight_dtype=weight_dtype)
    table = torch.tensor([[0, -1, 2]], dtype=torch.int32).expand(33, -1).contiguous()
    visible = torch.full((33,), 150, dtype=torch.int32)
    visible[0] = 0
    qn, wn, tn, vn = (t.to("npu") for t in (q, w, table, visible))

    def run(chunk):
        return ops.index_topk(
            qn, wn, cache, tn, vn, None, 8, 4, 8, chunk, 16, None, None, None
        )

    outputs = run(32)
    other = run(33)
    for a, b in zip(outputs, other):
        torch.testing.assert_close(a.cpu(), b.cpu(), atol=0, rtol=0)

    def check(result, lens):
        row_ids, row_lens, block_ids, block_lens = (t.cpu() for t in result)
        for i in range(33):
            valid = (torch.arange(192) < lens[i]) & (torch.arange(192) // 64 != 1)
            s = scores[i].float().masked_fill(~valid, -torch.inf)
            n = min(8, int(valid.sum()))
            assert row_lens[i] == n
            selected = row_ids[i, :n].long()
            assert (row_ids[i, n:] == -1).all()
            if n:
                assert (selected[1:] > selected[:-1]).all()
                torch.testing.assert_close(
                    s[selected].sort().values,
                    s.topk(n).values.sort().values,
                    atol=0,
                    rtol=0,
                )
            blocks = s.reshape(24, 8).amax(1)
            last = (int(lens[i]) - 1) // 8
            if last >= 0 and blocks[last] > -torch.inf:
                blocks[last] = torch.inf
            nb = min(4, int((blocks > -torch.inf).sum()))
            assert block_lens[i] == nb
            assert (block_ids[i, nb:] == -1).all()
            if nb:
                torch.testing.assert_close(
                    blocks[block_ids[i, :nb].long()].sort().values,
                    blocks.topk(nb).values.sort().values,
                    atol=0,
                    rtol=0,
                )

    check(outputs, visible)
    for _ in range(2):
        run(32)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        captured = run(32)
    newlens = torch.full_like(visible, 50)
    vn.copy_(newlens)
    graph.replay()
    torch.npu.synchronize()
    check(captured, newlens)


@pytest.mark.parametrize("heads", [4, 32])
@pytest.mark.parametrize("score_chunk", [8, 2048])
def test_reindex_scoring_geometry(heads, score_chunk):
    q, w, cache, scores = inputs(2, heads)
    table = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32, device="npu")
    visible = torch.tensor([150, 190], dtype=torch.int32, device="npu")
    candidates = torch.tensor(
        [[0, 7, 8, 18, -1], [2, 4, 20, 23, -1]], dtype=torch.int32
    )
    actual = ops.index_topk(
        q.to("npu"),
        w.to("npu"),
        cache,
        table,
        visible,
        candidates.to("npu"),
        16,
        0,
        8,
        1,
        score_chunk,
        None,
        None,
        None,
    )
    row_ids, lengths, _, _ = (x.cpu() for x in actual)
    for i in range(2):
        ids = (candidates[i, :, None] * 8 + torch.arange(8)).flatten()
        valid = (ids >= 0) & (ids < int(visible[i].cpu()))
        s = scores[i, ids.clamp_min(0)].float().masked_fill(~valid, -torch.inf)
        assert lengths[i] == 16
        torch.testing.assert_close(
            scores[i, row_ids[i].long()].float().sort().values,
            s.topk(16).values.sort().values,
            atol=0,
            rtol=0,
        )


def test_long_history_full_selection_and_live_acl_metadata():
    # The old one-block-per-16-keys grid exceeds Ascend's 65535-block limit
    # at this actual decode shape. Exercise full selection, not select-all.
    rows, history = 32, 50048
    q, w, cache, scores = inputs(rows, heads=32, history=history)
    table = torch.arange(history // 64, dtype=torch.int32)[None].repeat(rows, 1)
    table[:, 11] = -1
    visible = torch.tensor([50000 - 37 * i for i in range(rows)], dtype=torch.int32)
    qn, wn, tn, vn = (x.to("npu") for x in (q, w, table, visible))

    def run():
        return ops.index_topk(
            qn, wn, cache, tn, vn, None, 512, 2048, 8, 32, 2048, None, None, None
        )

    for _ in range(2):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        result = run()
    for step in range(2):
        if step:
            visible = torch.tensor(
                [16389 - 17 * i for i in range(rows)], dtype=torch.int32
            )
            visible[0] = 0
            table[:, 12:16] = -1
            vn.copy_(visible)
            tn.copy_(table)
        graph.replay()
        torch.npu.synchronize()
        row_ids, lengths, block_ids, block_lengths = (x.cpu() for x in result)
        positions = torch.arange(history)
        for i in range(rows):
            valid = (positions < visible[i]) & (table[i, positions // 64] >= 0)
            score = scores[i].float().masked_fill(~valid, -torch.inf)
            count = min(512, int(valid.sum()))
            assert lengths[i] == count
            assert (row_ids[i, count:] == -1).all()
            assert (row_ids[i, 1:count] > row_ids[i, : max(0, count - 1)]).all()
            torch.testing.assert_close(
                score[row_ids[i, :count].long()].sort().values,
                score.topk(count).values.sort().values,
                atol=0,
                rtol=0,
            )
            blocks = score.reshape(-1, 8).amax(1)
            latest = (int(visible[i]) - 1) // 8
            if latest >= 0 and blocks[latest] > -torch.inf:
                blocks[latest] = torch.inf
            count = min(2048, int((blocks > -torch.inf).sum()))
            assert block_lengths[i] == count
            assert (block_ids[i, count:] == -1).all()
            torch.testing.assert_close(
                blocks[block_ids[i, :count].long()].sort().values,
                blocks.topk(count).values.sort().values,
                atol=0,
                rtol=0,
            )
