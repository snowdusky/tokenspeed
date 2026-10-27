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

"""V4.1 original cache bytes and page ownership on Ascend."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import torch_dsv41_cache_quantize
from tokenspeed_kernel.ops.attention.dsv41 import (
    cache_gather,
    cache_pack,
    cache_scatter,
    cache_unpack,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.mark.parametrize(
    "cache_format,dim,values,width",
    [("swa", 512, 512, 528), ("global", 512, 256, 288), ("index", 128, 64, 68)],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("row_gap", [0, 3])
def test_packed_cache_and_replay(cache_format, dim, values, width, strided, row_gap):
    torch.manual_seed(123)
    rows = torch.randn(7, dim, dtype=torch.bfloat16) * 2
    rows[0].zero_()
    rows[1, ::2] *= -1
    expected_bytes, expected_values = torch_dsv41_cache_quantize(rows, cache_format)
    x = rows.to("npu")
    packed = cache_pack(x, cache_format, None)
    torch.testing.assert_close(packed.cpu(), expected_bytes, atol=0, rtol=0)
    torch.testing.assert_close(
        cache_unpack(packed, cache_format, None).cpu(),
        expected_values.bfloat16(),
        atol=0,
        rtol=0,
    )
    payload_width = width * (2 if strided else 1)
    storage = torch.full(
        (5, 64, payload_width + row_gap), 85, dtype=torch.uint8, device="npu"
    )
    payload = storage[..., :payload_width]
    cache = payload[..., ::2] if strided else payload
    slots = torch.tensor([0, 63, 64, 127, 128, -1, 320], device="npu")
    cache_scatter(x, cache, slots, cache_format)
    # Check the page-planar payload, including the 63 -> 64 page boundary.
    actual_bytes = cache.cpu().reshape(5, -1)
    for i, slot in enumerate(slots.cpu().tolist()[:5]):
        page, row = divmod(slot, 64)
        torch.testing.assert_close(
            actual_bytes[page, row * values : (row + 1) * values],
            expected_bytes[i, :values],
            atol=0,
            rtol=0,
        )
        start = 64 * values + row * (width - values)
        torch.testing.assert_close(
            actual_bytes[page, start : start + width - values],
            expected_bytes[i, values:],
            atol=0,
            rtol=0,
        )
    assert (actual_bytes[3:] == 85).all()
    if strided:
        assert (storage[..., 1::2].cpu() == 85).all()
    if row_gap:
        assert (storage[..., payload_width:].cpu() == 85).all()
    expected = expected_values.bfloat16()
    expected[-2:].zero_()
    torch.testing.assert_close(
        cache_gather(cache, slots, cache_format, None).cpu(), expected, atol=0, rtol=0
    )
    for _ in range(3):
        cache_scatter(x, cache, slots, cache_format)
        cache_gather(cache, slots, cache_format, None)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        cache_scatter(x, cache, slots, cache_format)
        output = cache_gather(cache, slots, cache_format, None)
    x.mul_(0.75)
    slots.copy_(torch.tensor([128, 127, 64, 63, 0, -1, 320], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    _, expected = torch_dsv41_cache_quantize(x.cpu(), cache_format)
    expected[-2:].zero_()
    torch.testing.assert_close(output.cpu(), expected.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("cache_format,width", [("index", 128), ("swa", 512)])
def test_codec_rows_exceed_device_launch_limit(cache_format, width):
    # More than 65535 rows (and four tiles/row for SWA) must use bounded workers.
    # These values are exactly representable in both cache codecs.
    rows = 65537
    values = (torch.arange(rows, device="npu") % 4).to(torch.bfloat16)
    source = values[:, None].expand(rows, width)
    packed = cache_pack(source, cache_format, None)
    restored = cache_unpack(packed, cache_format, None)
    torch.testing.assert_close(restored, source, atol=0, rtol=0)
