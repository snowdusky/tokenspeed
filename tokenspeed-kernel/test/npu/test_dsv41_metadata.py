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

"""Shared V4.1 metadata and RoPE contracts exercised on Ascend."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41 as ops
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.mark.parametrize("width", [1, 4])
def test_decode_metadata_matches_cpu(width):
    seq = torch.tensor([1, 64, 129], dtype=torch.int32)
    slots = torch.tensor([7, 2, 9], dtype=torch.int64)

    def allocate(device):
        return [
            torch.empty(n, dtype=torch.int64, device=device)
            for n in (4 * width, 4 * width, 4, 4)
        ]

    cpu = allocate("cpu")
    npu = allocate("npu")
    seq_npu, slots_npu = seq.to("npu"), slots.to("npu")
    for length in (seq, seq + 3):
        seq_npu.copy_(length)
        ops.decode_rows(length, slots, *cpu, 3, 1, width)
        ops.decode_rows(seq_npu, slots_npu, *npu, 3, 1, width)
        for expected, actual in zip(cpu, npu):
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def test_window_and_global_slot_mapping():
    positions = torch.tensor([0, 63, 64, 127, 128, -1], dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 0, 0, -1], dtype=torch.int64)
    table = torch.tensor([[1, 3, 2]], dtype=torch.int32)

    def run(device):
        pos, req, tab = (x.to(device) for x in (positions, requests, table))
        outputs = [
            torch.empty(s, dtype=torch.int32, device=device)
            for s in ((6,), (6, 128), (6,))
        ]
        ops.decode_window(pos, req, *outputs, tab, 4)
        selected = (
            torch.tensor([[0, 1, 62, 63, 64, -1]], dtype=torch.int32, device=device)
            .expand(6, -1)
            .contiguous()
        )
        outputs.append(ops.global_slots(selected, pos, req, tab, 2, 4))
        return outputs

    for expected, actual in zip(run("cpu"), run("npu")):
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("heads", [1, 4])
def test_rope_native_replay(heads):
    torch.manual_seed(42)
    source = torch.randn(4, heads, 128, dtype=torch.bfloat16)
    angles = torch.randn(130, 32)
    table = torch.cat((angles.cos(), angles.sin()), -1)
    positions = torch.tensor([-1, 1, 64, 129], dtype=torch.int64)

    def reference(x, pos):
        cosine, sine = table[pos.clamp_min(0)].chunk(2, -1)
        even, odd = x[..., -64:].float().unflatten(-1, (32, 2)).unbind(-1)
        rotated = torch.stack(
            (
                even * cosine[:, None] - odd * sine[:, None],
                odd * cosine[:, None] + even * sine[:, None],
            ),
            -1,
        ).flatten(-2)
        result = x.clone()
        result[..., -64:] = rotated.bfloat16()
        return result

    x, p, t = source.to("npu"), positions.to("npu"), table.to("npu")
    for _ in range(3):
        x.copy_(source)
        ops.rope_inplace(x, p, t, None)
    torch.testing.assert_close(x.cpu(), reference(source, positions), atol=0, rtol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        ops.rope_inplace(x, p, t, None)
    x.copy_(source * 2)
    p.copy_(torch.tensor([5, 2, 3, 0], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(x.cpu(), reference(source * 2, p.cpu()), atol=0, rtol=0)


@pytest.mark.parametrize("candidates", [0, 2048])
def test_select_all_official_width_and_live_replay(candidates):
    table = torch.tensor(
        [[1, 0, 2, -1, 9, 3, 4, 2], [0, 3, 2, 1, 1, 2, 3, 4]], dtype=torch.int32
    )
    visible = torch.tensor([512, 181], dtype=torch.int32)
    shapes = [(2, 512), (2,), (2, candidates), (2,)]
    expected = tuple(torch.empty(shape, dtype=torch.int32) for shape in shapes)
    actual = tuple(
        torch.empty(shape, dtype=torch.int32, device="npu") for shape in shapes
    )
    npu_table, npu_visible = table.to("npu"), visible.to("npu")

    def run():
        return ops.index_select_all(npu_table, npu_visible, 5, 8, actual)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        run()
    for lengths in ([512, 181], [0, 256], [127, 1]):
        visible.copy_(torch.tensor(lengths))
        npu_visible.copy_(visible)
        table.copy_(table.roll(1, 1))
        npu_table.copy_(table)
        ops.index_select_all(table, visible, 5, 8, expected)
        graph.replay()
        torch.npu.synchronize()
        for ref, result in zip(expected, actual, strict=True):
            torch.testing.assert_close(result.cpu(), ref, atol=0, rtol=0)
