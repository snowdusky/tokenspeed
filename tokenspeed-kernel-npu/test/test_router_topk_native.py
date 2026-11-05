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

"""Contract tests for the native sqrt-softplus router top-k.

Skipped when the native provider has not been built. The comparison is against
``torch_sqrt_softplus_topk`` semantics: sqrt-softplus scores, selection-only
bias, renormalisation, routed scaling, int32 ids and the caller's weight dtype.
"""

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops._v41 import router_topk_native

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or not router_topk_native.available(),
    reason="native router top-k requires an NPU and the built provider",
)

EXPERTS = 256
TOP_K = 6


def reference(logits, top_k, renormalize, bias, scale, weights_dtype):
    scores = torch.sqrt(F.softplus(logits.float()))
    selection = scores if bias is None else scores + bias.float()
    ids = torch.topk(selection, top_k, dim=-1, sorted=True).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
    if scale != 1.0:
        weights = weights * scale
    return weights.to(weights_dtype), ids.to(torch.int32)


def untied(logits, bias, top_k):
    """Rows whose selection keys are separated by more than a few FP32 ulps.

    Near-tied keys cannot be ordered the same way on CPU and device, so those
    rows are checked as a set instead of slot by slot.
    """
    scores = torch.sqrt(F.softplus(logits.float()))
    selection = scores if bias is None else scores + bias.float()
    values = selection.sort(dim=-1, descending=True).values
    gaps = values[:, :top_k] - values[:, 1 : top_k + 1]
    ulp = torch.finfo(torch.float32).eps * values[:, :top_k].abs().clamp_min(1.0)
    return ~(gaps <= 16.0 * ulp).any(dim=1)


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_reference_selection(renormalize, scale, with_bias):
    generator = torch.Generator().manual_seed(7)
    logits = (torch.randn(96, EXPERTS, generator=generator) * 2.0).npu()
    bias = (
        (torch.randn(EXPERTS, generator=generator) * 0.05).npu() if with_bias else None
    )

    weights, ids = router_topk_native.router_topk_sqrt_softplus(
        logits, TOP_K, renormalize, bias, scale, torch.float32
    )
    expected_weights, expected_ids = reference(
        logits.cpu(),
        TOP_K,
        renormalize,
        None if bias is None else bias.cpu(),
        scale,
        torch.float32,
    )

    rows = untied(logits.cpu(), None if bias is None else bias.cpu(), TOP_K)
    assert ids.dtype == torch.int32
    assert torch.equal(ids.cpu()[rows], expected_ids[rows])
    torch.testing.assert_close(
        weights.cpu()[rows], expected_weights[rows], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_logits_and_weights(dtype):
    generator = torch.Generator().manual_seed(11)
    logits = (torch.randn(32, EXPERTS, generator=generator) * 2.0).to(dtype).npu()
    bias = (torch.randn(EXPERTS, generator=generator) * 0.05).npu()

    weights, ids = router_topk_native.router_topk_sqrt_softplus(
        logits, TOP_K, True, bias, 2.5, torch.bfloat16
    )
    expected_weights, expected_ids = reference(
        logits.cpu(), TOP_K, True, bias.cpu(), 2.5, torch.bfloat16
    )

    assert weights.dtype == torch.bfloat16
    rows = untied(logits.cpu(), bias.cpu(), TOP_K)
    assert torch.equal(ids.cpu()[rows], expected_ids[rows])
    assert torch.equal(weights.cpu()[rows], expected_weights[rows])


def test_bias_changes_selection_but_not_weights():
    logits = torch.randn(16, EXPERTS, generator=torch.Generator().manual_seed(3)).npu()
    bias = torch.zeros(EXPERTS, device="npu")
    bias[0] = 5.0
    bias[1] = 4.0

    weights, ids = router_topk_native.router_topk_sqrt_softplus(
        logits, TOP_K, False, bias, 1.0, torch.float32
    )
    scores = torch.sqrt(F.softplus(logits.cpu().float()))
    selection = scores + bias.cpu()
    expected_ids = torch.topk(selection, TOP_K, dim=-1).indices

    rows = untied(logits.cpu(), bias.cpu(), TOP_K)
    assert torch.equal(ids.cpu()[rows], expected_ids[rows])
    # Weights are the unbiased scores at the returned ids.
    torch.testing.assert_close(
        weights.cpu(), scores.gather(1, ids.cpu().long()), atol=1e-6, rtol=1e-6
    )


def test_ties_keep_k_distinct_experts():
    logits = torch.zeros(8, EXPERTS, device="npu")
    logits[:, 3] = 1.0
    logits[:, 7] = 1.0
    logits[:, EXPERTS - 1] = 1.0

    weights, ids = router_topk_native.router_topk_sqrt_softplus(
        logits, TOP_K, False, None, 1.0, torch.float32
    )
    expected_ids = torch.tensor([3, 7, EXPERTS - 1, 0, 1, 2], dtype=torch.int32)
    assert torch.equal(ids.cpu(), expected_ids.expand(8, -1))
    for row in ids.cpu():
        assert torch.unique(row).numel() == TOP_K
    assert torch.equal(ids, ids[:1].expand_as(ids))
    # Equal keys still produce equal weights.
    assert torch.allclose(weights, weights[:1].expand_as(weights))


@pytest.mark.parametrize("experts", [128, 256, 384, 448, 512])
def test_expert_width_matches_reference(experts):
    """384 is the deployed V4.1 width; 128 is the DSPARK drafter's."""
    generator = torch.Generator().manual_seed(experts)
    logits = (torch.randn(96, experts, generator=generator) * 2.0).npu()
    bias = (torch.randn(experts, generator=generator) * 0.05).npu()

    weights, ids = router_topk_native.router_topk_sqrt_softplus(
        logits, TOP_K, True, bias, 1.0, torch.float32
    )
    expected_weights, expected_ids = reference(
        logits.cpu(), TOP_K, True, bias.cpu(), 1.0, torch.float32
    )
    rows = untied(logits.cpu(), bias.cpu(), TOP_K)
    assert torch.equal(ids.cpu()[rows], expected_ids[rows])
    torch.testing.assert_close(
        weights.cpu()[rows], expected_weights[rows], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("rows", [0, 1, 16, 95, 96, 97, 128, 129, 384])
def test_shape_sweep(rows):
    generator = torch.Generator().manual_seed(rows)
    logits = torch.randn(rows, EXPERTS, generator=generator).npu()
    bias = (torch.randn(EXPERTS, generator=generator) * 0.05).npu()
    weights, ids = router_topk_native.router_topk_sqrt_softplus(
        logits, TOP_K, True, bias, 2.5, torch.float32
    )
    expected_weights, expected_ids = reference(
        logits.cpu(), TOP_K, True, bias.cpu(), 2.5, torch.float32
    )
    untied_rows = untied(logits.cpu(), bias.cpu(), TOP_K)
    assert torch.equal(ids.cpu()[untied_rows], expected_ids[untied_rows])
    torch.testing.assert_close(
        weights.cpu()[untied_rows], expected_weights[untied_rows], atol=1e-6, rtol=1e-6
    )


def test_declines_unsupported_geometry():
    logits = torch.randn(8, 1024, device="npu")
    assert (
        router_topk_native.router_topk_sqrt_softplus(
            logits, TOP_K, False, None, 1.0, torch.float32
        )
        is None
    )
    logits = torch.randn(8, EXPERTS, device="npu")
    assert (
        router_topk_native.router_topk_sqrt_softplus(
            logits, 16, False, None, 1.0, torch.float32
        )
        is None
    )
    per_token_bias = torch.randn(8, EXPERTS, device="npu")
    assert (
        router_topk_native.router_topk_sqrt_softplus(
            logits, TOP_K, False, per_token_bias, 1.0, torch.float32
        )
        is None
    )
