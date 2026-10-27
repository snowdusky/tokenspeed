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

"""FP32 mHC mixing transforms, with complete 32-byte output blocks per program."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def tail(
    P,
    S,
    Scale,
    Base,
    Pre,
    Post,
    Comb,
    rows,
    hidden: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    iters: tl.constexpr,
):
    r = tl.program_id(0) * 2 + tl.arange(0, 2)
    h = tl.arange(0, 4)
    valid = r < rows
    rms = tl.rsqrt(tl.load(S + r, valid, other=0) / (4 * hidden) + rms_eps)
    a = (
        tl.load(P + r[:, None] * 24 + h[None, :], valid[:, None], other=0)
        * rms[:, None]
    )
    b = (
        tl.load(P + r[:, None] * 24 + 4 + h[None, :], valid[:, None], other=0)
        * rms[:, None]
    )
    pre = tl.sigmoid(a * tl.load(Scale) + tl.load(Base + h)[None, :]) + hc_eps
    post = 2 * tl.sigmoid(b * tl.load(Scale + 1) + tl.load(Base + 4 + h)[None, :])
    ci = h[:, None] * 4 + h[None, :]
    c = tl.load(
        P + r[:, None, None] * 24 + 8 + ci[None, :, :], valid[:, None, None], other=0
    )
    c = c * rms[:, None, None] * tl.load(Scale + 2) + tl.load(Base + 8 + ci)[None, :, :]
    c = tl.exp(c - tl.max(c, 2)[:, :, None])
    c = tl.div_rn(c, tl.sum(c, 2)[:, :, None]) + hc_eps
    c = tl.div_rn(c, tl.sum(c, 1)[:, None, :] + hc_eps)
    for _ in tl.static_range(1, iters):
        c = tl.div_rn(c, tl.sum(c, 2)[:, :, None] + hc_eps)
        c = tl.div_rn(c, tl.sum(c, 1)[:, None, :] + hc_eps)
    tl.store(Pre + r[:, None] * 4 + h[None, :], pre, valid[:, None])
    tl.store(Post + r[:, None] * 4 + h[None, :], post, valid[:, None])
    tl.store(Comb + r[:, None, None] * 16 + ci[None, :, :], c, valid[:, None, None])


def mixes(x, w, scale, base, rms_eps, hc_eps, sinkhorn_iters):
    rows, _, hidden = x.shape
    flat = x.flatten(1).float()
    projection = torch.nn.functional.linear(flat, w).contiguous()
    sums = flat.square().sum(-1)
    pre = torch.empty((rows, 4), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((rows, 4, 4), device=x.device, dtype=torch.float32)
    tail[(triton.cdiv(rows, 2),)](
        projection,
        sums,
        scale,
        base,
        pre,
        post,
        comb,
        rows,
        hidden,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return pre, post, comb


def apply_pre(residual, pre, *, norm_weight, norm_eps):
    from tokenspeed_kernel.ops.residual import triton as common

    rows = residual.numel() // (4 * residual.shape[-1])
    hidden = residual.shape[-1]
    out = residual.new_empty((*residual.shape[:-2], hidden))
    if rows:
        if norm_weight is None:
            common._mhc_pre_layer_triton_kernel[(rows, triton.cdiv(hidden, 1024))](
                pre,
                residual,
                out,
                hidden_size=hidden,
                hc_mult=4,
                block_h=1024,
                num_warps=4,
                enable_fp_fusion=False,
                multibuffer=False,
            )
        else:
            common._mhc_pre_layer_norm_hc4_kernel[(rows,)](
                pre,
                residual,
                norm_weight,
                out,
                hidden_size=hidden,
                eps=norm_eps,
                block_h=triton.next_power_of_2(hidden),
                num_warps=8,
                enable_fp_fusion=False,
                multibuffer=False,
            )
    return out


def post_mapping(hidden_states, residual, post, comb):
    from tokenspeed_kernel.ops.residual import triton as common

    hidden = residual.shape[-1]
    rows = residual.numel() // (4 * hidden)
    out = torch.empty_like(residual)
    if rows:
        common._mhc_post_hc4_triton_kernel[(rows, triton.cdiv(hidden, 1024))](
            comb,
            residual,
            post,
            hidden_states,
            out,
            hidden_size=hidden,
            block_h=1024,
            num_warps=4,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
