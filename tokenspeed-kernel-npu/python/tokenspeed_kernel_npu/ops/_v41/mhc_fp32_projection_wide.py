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

"""FP32 projection and RMS partials, then fused reduction and mHC coefficients.

The BF16 residual is promoted to FP32 for products. Weights stay FP32;
no quantized projection or per-forward weight expansion is introduced.
"""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def project(
    X,
    W,
    P,
    rows,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    base = tl.program_id(0) * R
    group, part = tl.program_id(1), tl.program_id(2)
    cols = group * C + tl.arange(0, C)
    k = part * B + tl.arange(0, B)
    weights = tl.load(
        W + cols[:, None] * K + k[None, :],
        (cols[:, None] < 24) & (k[None, :] < K),
        other=0,
    )
    for step in tl.static_range(R):
        row = base + step
        x = tl.load(X + row * K + k, (row < rows) & (k < K), other=0).to(tl.float32)
        value = tl.sum(weights * x[None, :], 1)
        tl.store(
            P + row * PARTS * 32 + part * 32 + cols, value, (row < rows) & (cols < 24)
        )
        if group == 0:
            tl.store(
                P + row * PARTS * 32 + part * 32 + 24, tl.sum(x * x, 0), row < rows
            )


@triton.jit
def partial_tail(
    P,
    Scale,
    Base,
    Pre,
    Post,
    Comb,
    rows,
    PARTS: tl.constexpr,
    PB: tl.constexpr,
    hidden: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    iters: tl.constexpr,
):
    r = tl.program_id(0) * 2 + tl.arange(0, 2)
    h = tl.arange(0, 4)
    valid = r < rows
    parts = tl.arange(0, PB)
    cols = tl.arange(0, 32)
    partial = tl.load(
        P
        + r[:, None, None] * PARTS * 32
        + parts[None, :, None] * 32
        + cols[None, None, :],
        valid[:, None, None]
        & (parts[None, :, None] < PARTS)
        & (cols[None, None, :] < 25),
        other=0,
    )
    m = tl.sum(partial, 1)
    sums = tl.sum(tl.where(cols[None, :] == 24, m, 0), 1)
    rms = tl.rsqrt(sums / (4 * hidden) + rms_eps)
    a = tl.gather(m, tl.broadcast_to(h[None, :], (2, 4)), 1) * rms[:, None]
    b = tl.gather(m, tl.broadcast_to((4 + h)[None, :], (2, 4)), 1) * rms[:, None]
    pre = tl.sigmoid(a * tl.load(Scale) + tl.load(Base + h)[None, :]) + hc_eps
    post = 2 * tl.sigmoid(b * tl.load(Scale + 1) + tl.load(Base + 4 + h)[None, :])
    ci = h[:, None] * 4 + h[None, :]
    c = tl.reshape(
        tl.gather(m, tl.broadcast_to(tl.reshape(8 + ci, (1, 16)), (2, 16)), 1),
        (2, 4, 4),
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
    rows, copies, hidden = x.shape
    if copies != 4 or x.dtype != torch.bfloat16 or w.dtype != torch.float32:
        raise ValueError("FP32 HC4 projection requires BF16 residuals and FP32 weights")
    if not x.is_contiguous() or not w.is_contiguous() or w.shape != (24, 4 * hidden):
        raise ValueError("HC4 projection requires contiguous input and [24,4H] weights")
    k = 4 * hidden
    block = 4096
    parts = triton.cdiv(k, block)
    partial = torch.empty((rows, parts, 32), device=x.device, dtype=torch.float32)
    pre = torch.empty((rows, 4), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((rows, 4, 4), device=x.device, dtype=torch.float32)
    if rows:
        project[(triton.cdiv(rows, 2), 6, parts)](
            x,
            w,
            partial,
            rows,
            k,
            parts,
            block,
            4,
            2,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        partial_tail[(triton.cdiv(rows, 2),)](
            partial,
            scale,
            base,
            pre,
            post,
            comb,
            rows,
            parts,
            triton.next_power_of_2(parts),
            hidden,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return pre, post, comb
