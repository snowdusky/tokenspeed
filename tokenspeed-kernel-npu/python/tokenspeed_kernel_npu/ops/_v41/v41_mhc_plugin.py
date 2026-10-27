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

"""Ascend mHC kernels through the existing residual operator contracts."""


def register():
    import torch
    from tokenspeed_kernel_npu.ops._v41.mhc_fp32_projection_wide import (
        mixes as vector_mixes,
    )
    from tokenspeed_kernel_npu.ops._v41.mhc_fused_tail import (
        apply_pre,
    )
    from tokenspeed_kernel_npu.ops._v41.mhc_fused_tail import mixes as baseline_mixes
    from tokenspeed_kernel_npu.ops._v41.mhc_post_native import (
        post_mapping as native_post,
    )

    def post_mapping(hidden_states, residual, post, comb):
        rows = residual.numel() // (4 * residual.shape[-1])
        return native_post(
            hidden_states, residual, post, comb, block=1024 if rows <= 6 else 2048
        )

    def mixes(x, weight, scale, base, norm_eps, hc_eps, sinkhorn_iters):
        # FP32 semantics on both paths; large prefill batches reuse GEMM weights.
        fn = vector_mixes if x.shape[0] <= 96 else baseline_mixes
        return fn(x, weight, scale, base, norm_eps, hc_eps, sinkhorn_iters)

    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    bf = dense_tensor_format(torch.bfloat16)
    fp = dense_tensor_format(torch.float32)
    for mode, fn, sig, traits in (
        ("mhc_mixes", mixes, format_signature(residual=bf), None),
        (
            "mhc_apply_pre",
            apply_pre,
            format_signature(residual=bf, pre=fp),
            {"hc_mult": frozenset({4})},
        ),
        (
            "mhc_post",
            post_mapping,
            format_signature(hidden_states=bf, residual=bf, post=fp, comb=fp),
            {"hc_mult": frozenset({4})},
        ),
    ):
        register_kernel(
            "residual",
            mode,
            name="ascend_v41_" + mode,
            solution="ascend",
            capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
            signatures=frozenset({sig}),
            traits=traits,
            priority=Priority.SPECIALIZED,
        )(fn)
