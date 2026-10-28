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

"""BF16 operands with FP32 accumulation for V4 projections."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["M", "N"])
def _bf16_linear_fp32_kernel(
    X,
    W,
    O,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        k = start + inner
        x = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        O + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def bf16_linear_fp32(x, weight):
    """Project contiguous BF16 [M,K] and [N,K] operands with FP32 accumulation.

    No operand conversion or persistent FP32 weight copy is needed. Weight
    and input updates are read afresh on every graph replay.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Expected compatible [M,K] and [N,K] operands")
    if x.device.type != "npu" or x.device != weight.device:
        raise ValueError("Operands must share an NPU device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 operands required")
    if not weight.is_contiguous():
        raise ValueError("Contiguous projection operands required")
    x = x.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m and n:
        bm, bn, bk = 16 if m <= 16 else 32, 128, 512
        _bf16_linear_fp32_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x,
            weight,
            output,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=True,
        )
    return output
