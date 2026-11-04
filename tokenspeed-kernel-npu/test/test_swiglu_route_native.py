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

"""Route-weight SwiGLU agrees with the existing scatter/activation path."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu._triton import triton
from tokenspeed_kernel_npu.ops._v41 import swiglu_route_native
from tokenspeed_kernel_npu.ops.dsv41_moe import (
    _scatter_route_weights,
    _swiglu_with_route_weights,
    _weighted_swiglu_active,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")


@pytest.mark.parametrize("implementation", ["native", "triton"])
@pytest.mark.parametrize("limit", [7.0, 0.0, -1.0])
@pytest.mark.parametrize("capacity,valid", [(96, 73), (768, 93), (768, 0)])
def test_routed_swiglu_matches_scatter(implementation, limit, capacity, valid):
    if implementation == "native" and not swiglu_route_native.available():
        pytest.skip("native provider is not built")
    hidden, top_k, experts = 2304, 6, 48
    generator = torch.Generator().manual_seed(capacity + valid)
    gate_up = (
        (torch.randn(capacity, 2 * hidden, generator=generator) * 4).bfloat16().npu()
    )
    # Non-identity routing, remote routes and a non-contiguous weight view.
    permutation = torch.randperm(capacity, generator=generator).to(torch.int32)
    permutation[permutation >= valid] = -1
    mapping = permutation.npu()
    storage = torch.rand(capacity // top_k, top_k * 2, generator=generator).npu()
    weights = storage[:, ::2]
    counts = torch.zeros(experts, dtype=torch.int64)
    counts[0] = valid // 3
    counts[-1] = valid - valid // 3
    counts = counts.npu()
    expanded_weights = torch.empty(capacity, device="npu", dtype=torch.float32)
    _scatter_route_weights[(triton.cdiv(capacity, 256),)](
        mapping,
        weights,
        expanded_weights,
        capacity,
        top_k,
        weights.stride(0),
        weights.stride(1),
        256,
        num_warps=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )
    expected = torch.full((capacity, hidden), -13, dtype=torch.bfloat16, device="npu")
    _weighted_swiglu_active[(48,)](
        gate_up,
        expanded_weights,
        counts,
        expected,
        hidden,
        experts,
        capacity,
        limit,
        48,
        triton.next_power_of_2(hidden),
        num_warps=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )
    actual = torch.full_like(expected, -13)
    if implementation == "native":
        assert swiglu_route_native.swiglu_route_native(
            gate_up, mapping, weights, counts, actual, limit
        )
    else:
        _swiglu_with_route_weights[(48,)](
            gate_up,
            mapping,
            weights,
            counts,
            actual,
            capacity,
            hidden,
            top_k,
            experts,
            weights.stride(0),
            weights.stride(1),
            limit,
            48,
            triton.next_power_of_2(hidden),
            num_warps=1,
            multibuffer=False,
            enable_fp_fusion=False,
        )
    torch.testing.assert_close(actual[:valid], expected[:valid], atol=0, rtol=0)
    assert torch.equal(actual[valid:], torch.full_like(actual[valid:], -13))


@pytest.mark.parametrize(
    "hidden,dtype", [(4096, torch.bfloat16), (2304, torch.float16)]
)
def test_native_declines_unsupported_abi(hidden, dtype):
    gate_up = torch.zeros(6, 2 * hidden, dtype=dtype, device="npu")
    mapping = torch.arange(6, dtype=torch.int32, device="npu")
    weights = torch.ones(1, 6, dtype=torch.float32, device="npu")
    counts = torch.tensor([6], dtype=torch.int64, device="npu")
    out = torch.full((6, hidden), -13, dtype=dtype, device="npu")
    assert not swiglu_route_native.swiglu_route_native(
        gate_up, mapping, weights, counts, out, 7.0
    )
    assert torch.equal(out, torch.full_like(out, -13))
