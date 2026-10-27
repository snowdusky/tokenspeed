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

"""V4.1 local experts: clipped SwiGLU, pre-down FP32 route weights and sum."""

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.mxfp import expand_active_mxfp4_experts


@triton.jit
def _scatter_route_weights(
    Map, Weights, Out, N, K: tl.constexpr, W0, W1, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    destination = tl.load(Map + index, index < N, other=-1)
    live = (index < N) & (destination >= 0) & (destination < N)
    weight = tl.load(
        Weights + (index // K) * W0 + (index % K) * W1, index < N, other=0
    ).to(tl.float32)
    # Routing destinations are unique; remote routes do not write a dummy row.
    tl.store(Out + tl.where(live, destination, 0), weight, live)


@triton.jit
def _weighted_swiglu_active(
    X,
    W,
    Counts,
    Y,
    H: tl.constexpr,
    E: tl.constexpr,
    N,
    LIMIT: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    experts = tl.arange(0, triton.next_power_of_2(E))
    valid = tl.sum(tl.load(Counts + experts, experts < E, other=0).to(tl.int32), 0)
    valid = tl.minimum(valid, N)
    col = tl.arange(0, BLOCK)
    # A fixed number of workers avoids launching a program for every reserved
    # row. Counts stay on device and can change on every graph replay.
    for row in range(tl.program_id(0), valid, CORES):
        gate = tl.load(X + row * 2 * H + col, col < H, other=0).to(tl.float32)
        up = tl.load(X + row * 2 * H + H + col, col < H, other=0).to(tl.float32)
        if LIMIT > 0:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        w = tl.load(W + row).to(tl.float32)
        result = ((gate / (1.0 + tl.exp(-gate))) * up) * w
        tl.store(Y + row * H + col, result, col < H)


@triton.jit
def _combine(
    Expanded,
    Map,
    Out,
    H: tl.constexpr,
    K: tl.constexpr,
    CAPACITY,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for k in tl.static_range(K):
        dst = tl.load(Map + row * K + k)
        x = tl.load(
            Expanded + dst.to(tl.int64) * H + col,
            (dst >= 0) & (dst < CAPACITY) & (col < H),
            other=0,
        ).to(tl.float32)
        value = value + x
    tl.store(Out + row * H + col, value, col < H)


def mxfp4_bf16_moe_apply(
    plan,
    x,
    w,
    router_logits,
    topk_weights=None,
    topk_ids=None,
    num_tokens_global=None,
    max_num_tokens_per_gpu=None,
    do_finalize=True,
    enable_pdl=False,
):
    if not do_finalize or topk_weights is None or topk_ids is None:
        raise ValueError("V4.1 experts require precomputed routes and finalization")
    if w.swiglu_arg is None or w.swiglu_arg.limit is None:
        raise ValueError("V4.1 experts require an explicit SwiGLU limit")
    if w.w13_input_layout != "concatenated":
        raise ValueError("V4.1 experts require concatenated gate/up weights")
    if x.shape[0] == 0:
        return torch.empty_like(x)
    ids = topk_ids.to(torch.int32)
    capacity = ids.numel()
    lo = w.ep_rank * w.num_local_experts
    expanded, mapping, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        expert_idx=ids,
        active_num=capacity,
        expert_num=w.num_experts,
        quant_mode=-1,
        active_expert_range=[lo, lo + w.num_local_experts],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    # Only the contiguous active prefix is consumed by SwiGLU and GMM.
    weights = torch.empty(capacity, dtype=torch.float32, device=x.device)
    _scatter_route_weights[(triton.cdiv(capacity, 256),)](
        mapping,
        topk_weights,
        weights,
        capacity,
        ids.shape[1],
        topk_weights.stride(0),
        topk_weights.stride(1),
        256,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    group_counts = counts
    expert_ids = None
    if w.w13_weight.dtype == torch.uint8:
        # Routing already orders rows by expert. Removing empty groups keeps
        # that row order while bounding weight scratch by the route count.
        experts = torch.arange(w.num_local_experts, device=x.device)
        expert_ids = (
            torch.where(counts > 0, experts, w.num_local_experts)
            .sort()
            .values[: min(w.num_local_experts, capacity)]
        )
        group_counts = counts[
            expert_ids.clamp_max(w.num_local_experts - 1)
        ].masked_fill(expert_ids == w.num_local_experts, 0)
    w13 = (
        expand_active_mxfp4_experts(
            w.w13_weight, w.w13_weight_scale, group_counts, expert_ids
        )
        if w.w13_weight.dtype == torch.uint8
        else w.w13_weight
    )
    gate_up = torch_npu.npu_grouped_matmul(
        [expanded],
        [w13],
        group_list=group_counts,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.bfloat16,
    )[0]
    del w13
    hidden = torch.empty(
        (expanded.shape[0], w.intermediate_size), dtype=x.dtype, device=x.device
    )
    _weighted_swiglu_active[(48,)](
        gate_up,
        weights,
        counts,
        hidden,
        w.intermediate_size,
        w.num_local_experts,
        expanded.shape[0],
        float(w.swiglu_arg.limit),
        48,
        triton.next_power_of_2(w.intermediate_size),
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    w2 = (
        expand_active_mxfp4_experts(
            w.w2_weight, w.w2_weight_scale, group_counts, expert_ids
        )
        if w.w2_weight.dtype == torch.uint8
        else w.w2_weight
    )
    expanded_out = torch_npu.npu_grouped_matmul(
        [hidden],
        [w2],
        group_list=group_counts,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.bfloat16,
    )[0]
    del w2
    out = torch.empty_like(x)
    _combine[(x.shape[0], triton.cdiv(x.shape[1], 4096))](
        expanded_out,
        mapping,
        out,
        x.shape[1],
        ids.shape[1],
        expanded_out.shape[0],
        4096,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return out
