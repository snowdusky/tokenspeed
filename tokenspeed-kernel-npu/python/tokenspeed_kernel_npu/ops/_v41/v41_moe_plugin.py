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

"""Register the native V4.1 router top-k through the plugin boundary."""


def register():
    import torch
    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures
    from tokenspeed_kernel_npu.ops._v41.router_topk_native import (
        MAX_EXPERTS,
        MAX_TOP_K,
        router_topk_sqrt_softplus,
        supported,
    )

    @register_kernel(
        "moe",
        "topk",
        name="ascend_native_v41_router_topk",
        solution="ascend_native",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            "router_logits",
            "dense",
            {torch.float32, torch.bfloat16, torch.float16},
        ),
        traits={
            # V4.1 model configurations use several routed-expert widths.
            "experts": frozenset({128, 256, 384, 448, MAX_EXPERTS}),
            "top_k": frozenset(range(1, MAX_TOP_K + 1)),
            "routing_kind": frozenset({"plain", "bias"}),
            "score_function": frozenset({"sqrt_softplus"}),
            "renormalize": frozenset({True, False}),
            # Ties resolve to the lowest index and the remaining tied entries
            # stay eligible, which is not torch.topk's undefined tie order.
            "tie_order": frozenset({"lowest_index"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def ascend_native_v41_router_topk(
        router_logits,
        top_k,
        renormalize,
        correction_bias,
        hash_indices_table,
        input_ids,
        need_scores,
        routed_scaling_factor,
        weights_dtype,
    ):
        # Hash routing selects from the checkpoint table, and need_scores asks
        # for the unbiased score tensor this kernel does not materialise. Both
        # are excluded by traits; reaching here means a caller bypassed them.
        if hash_indices_table is not None or input_ids is not None:
            raise ValueError("native router top-k does not serve hash routing")
        if need_scores:
            raise ValueError("native router top-k does not materialise scores")
        if correction_bias is not None and correction_bias.ndim != 1:
            raise ValueError("native router top-k requires a 1-D correction bias")
        if not supported(router_logits, top_k, correction_bias):
            raise ValueError(
                f"native router top-k cannot serve {tuple(router_logits.shape)} "
                f"experts with top_k={top_k}"
            )
        weights, ids = router_topk_sqrt_softplus(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            routed_scaling_factor,
            weights_dtype,
        )
        return weights, ids, router_logits
