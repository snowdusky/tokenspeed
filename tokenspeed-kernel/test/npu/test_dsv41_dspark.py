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

"""DSpark FP8 round trips and workspace replay on Ascend."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel.numerics.torch.attention.dsv41 import torch_dsv41_cache_quantize
from tokenspeed_kernel.ops.attention import dsv41


def _reference(x, weight, eps, positions, table):
    y = x.float()
    if weight is not None:
        y = (
            (
                weight.float()
                * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps))
            )
            .bfloat16()
            .float()
        )
    if table is not None:
        rotary = table.shape[1]
        cs = table[positions.clamp_min(0)]
        pairs = y[:, -rotary:].reshape(-1, rotary // 2, 2)
        c, s = cs.chunk(2, -1)
        y[:, -rotary:] = torch.stack(
            (
                pairs[..., 0] * c - pairs[..., 1] * s,
                pairs[..., 1] * c + pairs[..., 0] * s,
            ),
            -1,
        ).flatten(1)
        y = y.bfloat16().float()
    return torch_dsv41_cache_quantize(y.bfloat16(), "swa")[1].bfloat16()


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("batch,block", [(1, 3), (4, 5)])
def test_dspark_rows_workspace_replay(normalize, rotate, batch, block):
    torch.manual_seed(74)
    tokens = batch * block
    x = torch.randn(tokens, 1024, dtype=torch.bfloat16)[:, :512]
    # Include zero and tiny groups, exercising the reference lower scale bound.
    x[0, :32] = 0
    x[0, 32:64] *= 1e-8
    w = torch.randn(512, dtype=torch.bfloat16) if normalize else None
    eps = 1e-6 if normalize else None
    positions = torch.arange(tokens) - 1 if rotate else None
    table = torch.randn(64, 64) if rotate else None
    cpu_window = torch.randn(5, 8, 544, dtype=torch.bfloat16)[..., :512]
    slots = torch.arange(tokens)
    slots[-1] = -1
    page = torch.arange(batch * 9).reshape(batch, 9) % 5
    row = torch.arange(batch * 9).reshape(batch, 9) % 8

    def dev(t):
        return t.to("npu") if t is not None else None

    # Keep the field's padding: copying a sliced tensor alone makes it contiguous.
    storage = torch.randn(5, 8, 544, dtype=torch.bfloat16, device="npu")
    window = storage[..., :512]
    window.copy_(cpu_window)
    padding = storage[..., 512:].clone()
    xd, wd, pd, td = map(dev, (x, w, positions, table))
    sd, paged, rowd = map(dev, (slots, page, row))
    out = torch.empty_like(xd)

    def forward():
        dsv41.dspark_rows(xd, wd, eps, pd, td, window, sd, out)
        return dsv41.dspark_workspace(xd, wd, eps, pd, td, window, paged, rowd, block)

    for _ in range(2):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        result = forward()
    for i in range(3):
        current = (x.float() + i * 0.125).bfloat16()
        xd.copy_(current)
        graph.replay()
        torch.npu.synchronize()
        expected = _reference(current, w, eps, positions, table)
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
        for j, slot in enumerate(slots.tolist()):
            if slot >= 0:
                cpu_window[slot // 8, slot % 8] = expected[j]
        torch.testing.assert_close(window.cpu(), cpu_window, atol=0, rtol=0)
        expected_workspace = torch.cat(
            (cpu_window[page, row], expected.reshape(batch, block, 512)), 1
        )
        torch.testing.assert_close(result.cpu(), expected_workspace, atol=0, rtol=0)
        torch.testing.assert_close(storage[..., 512:], padding, atol=0, rtol=0)
