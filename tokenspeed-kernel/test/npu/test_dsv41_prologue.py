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

"""Norm/RoPE rounding and the original cache formats through public operators."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import torch_dsv41_cache_quantize
from tokenspeed_kernel.ops.attention import dsv41 as ops
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


def rotate(x, positions, table):
    c, s = table[positions.clamp_min(0)].chunk(2, -1)
    if x.ndim == 3:
        c, s = c[:, None], s[:, None]
    even, odd = x[..., -64:].float().unflatten(-1, (32, 2)).unbind(-1)
    result = x.clone()
    result[..., -64:] = (
        torch.stack((even * c - odd * s, odd * c + even * s), -1).flatten(-2).bfloat16()
    )
    return result


@pytest.mark.parametrize(
    "fmt,dim,width", [("index", 128, 68), ("swa", 512, 528), ("global", 512, 288)]
)
def test_key_prologue(fmt, dim, width):
    torch.manual_seed(81)
    x = torch.randn(5, dim, dtype=torch.bfloat16)
    weight = torch.randn(dim, dtype=torch.bfloat16)
    positions = torch.tensor([-1, 0, 17, 63, 64], dtype=torch.int64)
    angles = torch.randn(128, 32)
    table = torch.cat((angles.cos(), angles.sin()), -1)
    slots = torch.tensor([0, 63, 64, 127, -1], dtype=torch.int64)
    normalized = (
        x
        if fmt == "global"
        else (
            weight.float()
            * (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            )
        ).bfloat16()
    )
    expected = rotate(normalized, positions, table)
    _, decoded = torch_dsv41_cache_quantize(expected, fmt)
    xn, wn, pn, tn, sn = (v.to("npu") for v in (x, weight, positions, table, slots))
    cache = torch.zeros(2, 64, width, dtype=torch.uint8, device="npu")
    if fmt == "index":
        actual = ops.index_key_norm_rope(xn, wn, pn, tn, 1e-6)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        ops.index_key_norm_rope_scatter(xn, wn, pn, tn, 1e-6, cache, sn)
    elif fmt == "global":
        ops.global_rope_scatter(xn, pn, tn, cache, sn)
    else:
        out = torch.empty_like(xn)
        ops.swa_rope_scatter(xn, pn, tn, cache, sn, fmt, out, wn, 1e-6)
        torch.testing.assert_close(out.cpu(), decoded.bfloat16(), atol=0, rtol=0)
    decoded[-1].zero_()
    actual = ops.cache_gather(cache, sn, fmt, None).cpu()
    torch.testing.assert_close(actual, decoded.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(xn.cpu(), x, atol=0, rtol=0)


def test_query_prologue():
    torch.manual_seed(92)
    q = torch.randn(5, 2, 128, dtype=torch.bfloat16)
    w = torch.randn(5, 2, dtype=torch.bfloat16)
    positions = torch.tensor([-1, 0, 17, 63, 64])
    angles = torch.randn(128, 32)
    table = torch.cat((angles.cos(), angles.sin()), -1)
    actual, weights = ops.index_query_rope_weights(
        q.to("npu"), w.to("npu"), positions.to("npu"), table.to("npu"), 0.015625
    )
    torch.testing.assert_close(
        actual.cpu(), rotate(q, positions, table), atol=0, rtol=0
    )
    torch.testing.assert_close(
        weights.cpu(), (w.float() * 0.015625).bfloat16(), atol=0, rtol=0
    )
