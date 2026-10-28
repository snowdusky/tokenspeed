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

"""Shared expert activation keeps the model's BF16 numerical contract."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.models.deepseek_v41_ascend910 import (
    Ascend910MLP as DeepseekV4MLP,
)

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.mark.parametrize("rows", [0, 1, 32])
@pytest.mark.parametrize("limit", [None, 1.0])
def test_shared_expert_activation_matches_reference(rows, limit):
    with torch.device("npu"):
        model = DeepseekV4MLP(
            128,
            128,
            "silu",
            Mapping(rank=0, world_size=1),
            None,
            "shared",
            swiglu_limit=limit,
        ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for layer in (model.gate_up_proj, model.down_proj):
            layer.weight.normal_(std=0.05)
            layer.quant_method.process_weights_after_loading(layer)
        x = torch.randn((rows, 128), device="npu", dtype=torch.bfloat16)
        actual = model(x)
        if rows == 0:
            assert actual.shape == (0, 128)
            return
        gate_up, _ = model.gate_up_proj(x)
        expected, _ = model.down_proj(model.act_fn.forward_native(gate_up))
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)
