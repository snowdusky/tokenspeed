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

"""Checkpoint byte expansion against CPU PyTorch, including scale extremes."""

import pytest
import torch
from tokenspeed_kernel.ops.quantization import dequantize_mxfp_weight
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


def reference(weight, scales, bits, block_size):
    if bits == 8:
        values = weight.contiguous().view(torch.float8_e4m3fn).float()
    else:
        codes = torch.stack((weight & 15, weight >> 4), -1).flatten(1).long()
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        values = table[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    bn, bk = block_size
    scale = scales.contiguous().view(torch.float8_e8m0fnu).float()
    expanded = scale.repeat_interleave(bn, 0).repeat_interleave(bk, 1)
    return (values * expanded[: values.shape[0], : values.shape[1]]).bfloat16()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("block_size", [(1, 32), (32, 32)])
def test_dequantize_checkpoint_bytes(bits, block_size):
    torch.manual_seed(123)
    weight = torch.randint(0, 256, (65, 160), dtype=torch.uint8)
    bn, bk = block_size
    scales = torch.randint(
        100, 140, ((65 + bn - 1) // bn, 160 * (8 // bits) // bk), dtype=torch.uint8
    )
    # Strided tensors exercise partial row groups and the shared TP loader's views.
    weight = weight[:, ::2]
    scales = scales[:, ::2]
    expected = reference(weight, scales, bits, block_size)
    actual = dequantize_mxfp_weight(
        weight.to("npu"), scales.to("npu"), bits=bits, block_size=block_size
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("bits", [4, 8])
def test_all_codes_and_scale_exponents(bits):
    weight = torch.arange(256, dtype=torch.uint8).expand(256, -1).contiguous()
    scales = (
        torch.arange(256, dtype=torch.uint8)[:, None]
        .expand(-1, 8 * (8 // bits))
        .contiguous()
    )
    actual = dequantize_mxfp_weight(
        weight.to("npu"), scales.to("npu"), bits=bits, block_size=(1, 32)
    )
    torch.testing.assert_close(
        actual.cpu(),
        reference(weight, scales, bits, (1, 32)),
        atol=0,
        rtol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_weight_expansion_exceeds_launch_grid_limit(bits):
    # More than 65535 1024-element tiles, covering multiple grid strides.
    rows, width = 65537, 1024
    weight = torch.full(
        (rows, width * bits // 8),
        0x22 if bits == 4 else 0x38,
        dtype=torch.uint8,
        device="npu",
    )
    scales = torch.full((rows, width // 32), 127, dtype=torch.uint8, device="npu")
    # Distinguish rows, columns and later grid strides rather than only checking
    # a constant fill; the E2M1 0x22 and E4M3 0x38 codes both represent one.
    weight[::17].zero_()
    scales[::19, ::3] = 128
    actual = dequantize_mxfp_weight(weight, scales, bits=bits, block_size=(1, 32))
    for begin in range(0, rows, 1024):
        end = min(begin + 1024, rows)
        expected = reference(
            weight[begin:end].cpu(), scales[begin:end].cpu(), bits, (1, 32)
        )
        torch.testing.assert_close(actual[begin:end].cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(4608, 5120), (5120, 2304)])
def test_active_expert_expansion_real_projection_dimensions(shape):
    from tokenspeed_kernel_npu.ops.mxfp import expand_active_mxfp4_experts

    n, k = shape
    weight = torch.full((3, n, k // 2), 0x22, dtype=torch.uint8, device="npu")
    row_scales = (torch.arange(n) % 7 + 124).to(torch.uint8)
    scales = row_scales[None, :, None].expand(3, n, k // 32).contiguous().to("npu")
    weight = weight.transpose(1, 2).contiguous()
    scales = scales.transpose(1, 2).contiguous()
    expert_ids = torch.arange(3, device="npu")
    counts = torch.tensor([1, 0, 2], dtype=torch.int64, device="npu")
    expected = row_scales.view(torch.float8_e8m0fnu).float().bfloat16().expand(k, n)
    for _ in range(3):
        expand_active_mxfp4_experts(weight, scales, counts, expert_ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        actual = expand_active_mxfp4_experts(weight, scales, counts, expert_ids)
    for live in ([1, 0, 2], [0, 3, 0], [2, 1, 1]):
        counts.copy_(torch.tensor(live))
        graph.replay()
        torch.npu.synchronize()
        for expert, count in enumerate(live):
            if count:
                torch.testing.assert_close(
                    actual[expert].cpu(), expected, atol=0, rtol=0
                )


def test_active_expert_partial_tiles_and_live_ids():
    from tokenspeed_kernel_npu.ops.mxfp import expand_active_mxfp4_experts

    torch.manual_seed(83)
    weight = torch.randint(0, 256, (3, 67, 80), dtype=torch.uint8)
    scales = torch.randint(120, 130, (3, 67, 5), dtype=torch.uint8)
    expected = torch.stack(
        [reference(weight[i], scales[i], 4, (1, 32)).T for i in range(3)]
    ).to("npu")
    weight = weight.transpose(1, 2).contiguous().to("npu")
    scales = scales.transpose(1, 2).contiguous().to("npu")
    ids = torch.tensor([2, 0, 3], device="npu")
    counts = torch.tensor([1, 2, 0], device="npu")

    def forward():
        return expand_active_mxfp4_experts(weight, scales, counts, ids)

    for _ in range(3):
        forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        actual = forward()
    for live_ids, live_counts in (([2, 0, 3], [1, 2, 0]), ([1, 3, 0], [1, 0, 2])):
        ids.copy_(torch.tensor(live_ids))
        counts.copy_(torch.tensor(live_counts))
        graph.replay()
        torch.npu.synchronize()
        for group, count in enumerate(live_counts):
            if count:
                torch.testing.assert_close(
                    actual[group], expected[live_ids[group]], atol=0, rtol=0
                )
