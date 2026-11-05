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

"""V4.1 FlatQuant experts over device-owned, expert-sorted active rows.

Dispatch/Combine remain outside this provider. The precomputed routing entry
uses prepared NZ weights and grouped W4A4 arithmetic.
"""

import functools

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_moe import (
    _combine,
    _scatter_route_weights,
    _weighted_swiglu_active,
)
from tokenspeed_kernel_npu.ops.dsv41_w4a4_native import grouped_w4a4
from tokenspeed_kernel_npu.ops.w4a4 import (
    prepare_flatquant_stage,
    prepare_w4a4_grouped_plan,
    prepare_w4a4_moe_weights,
)


def process_weights(*, plan, w):
    import flash_ops  # noqa: F401 -- registers the native FlatQuant operation

    quantize = getattr(torch.ops.custom, "flat_quant", None)
    if quantize is None:
        raise RuntimeError(
            "V4.1 W4A4 requires flash_ops with the fused FlatQuant operation"
        )
    arguments = {arg.name for arg in quantize.default._schema.arguments}
    if not {"clip_ratio", "dst_dtype", "group_list", "group_list_type"} <= arguments:
        raise RuntimeError(
            "FlatQuant must support clipping and device-owned expert counts"
        )

    if w._int4_plan is not None:
        if "w4a4_grouped_plan" not in plan:
            raise RuntimeError("Prepared weights belong to another W4A4 provider")
        return
    if len(w._int4_loaded) != 6 * w.num_local_experts:
        raise ValueError("Missing W4A4 expert weights or scales")
    for module in (w.w1_trans, w.w2_trans, w.act_quantizer_w1, w.act_quantizer_w2):
        module.validate_loaded()
    if not w.w4a4_gmm_is_transposed or w.w13_input_layout != "concatenated":
        raise ValueError("Native W4A4 requires packed [E,N,K/8] gate/up weights")
    if (w.hidden_size, w.intermediate_size) != (5120, 2304):
        raise ValueError("This native provider supports V4.1 Flash expert geometry")
    stages = [
        prepare_flatquant_stage(
            left.matrix_left, left.matrix_right, clip.clip_factor_a, width=width
        )
        for left, clip, width in (
            (w.w1_trans, w.act_quantizer_w1, w.hidden_size),
            (w.w2_trans, w.act_quantizer_w2, w.intermediate_size),
        )
    ]
    w._int4_plan = prepare_w4a4_moe_weights(
        w.w13_weight,
        w.w2_weight,
        w.w13_weight_scale,
        w.w2_weight_scale,
        *stages,
        transposed=True,
        native_packed_nz=True,
    )
    w.w13_weight.data, w.w2_weight.data = w._int4_plan.w13, w._int4_plan.w2
    plan["w4a4_grouped_plan"] = prepare_w4a4_grouped_plan(w._int4_plan)
    plan["expert_slot_tensors"] = functools.partial(expert_slot_tensors, w=w, plan=plan)


def expert_slot_tensors(*, w, plan) -> list[torch.Tensor]:
    """Return transportable aliases for every prepared expert slot.

    Called only when preparing expert migration, before graph capture. The
    aliases preserve the packed NZ data and include both scale representations.
    """
    grouped = plan["w4a4_grouped_plan"]

    # HCCL transports ND buffers. Expose each aligned expert's packed NZ
    # bytes without format conversion, retaining the addresses graphs read.
    def packed_slots(weight):
        if (
            weight.storage_offset()
            or weight.untyped_storage().nbytes()
            != weight.numel() * weight.element_size()
        ):
            raise ValueError("EPLB requires unpadded, whole-storage NZ expert slots")
        # A set_ alias of the original storage inherits its NZ descriptor;
        # HCCL rejects offset slices of that internal format. Give the same
        # bytes a separate ND descriptor without copying or reordering them.
        # This non-owning storage stays valid while w._int4_plan owns weights.
        storage = torch_npu._C._construct_storage_from_data_pointer(
            weight.data_ptr(), weight.device, weight.untyped_storage().nbytes()
        )
        return torch.empty(0, dtype=torch.int32, device=weight.device).set_(
            storage, 0, (w.num_local_experts, weight.numel() // w.num_local_experts)
        )

    return [
        packed_slots(w._int4_plan.w13),
        packed_slots(w._int4_plan.w2),
        w._int4_plan.scale13,
        w._int4_plan.scale2,
        grouped.scale13.view(torch.int64),
        grouped.scale2.view(torch.int64),
    ]


def grouped_experts(x, route_weights, counts, *, w, plan):
    """Apply clipped SwiGLU and FP32 route weights before Down projection.

    Counts remain on device. Static graph capacity may exceed their sum; no
    transform, activation or GEMM reads the inactive tail as a real row.
    """
    state = plan["w4a4_grouped_plan"]
    weights = state.weights
    if (
        x.dtype != torch.bfloat16
        or counts.dtype != torch.int64
        or counts.shape != (w.num_local_experts,)
        or x.shape[1] != w.hidden_size
    ):
        raise ValueError("Invalid sorted V4.1 expert rows/counts")
    if w.swiglu_arg is None or w.swiglu_arg.limit is None:
        raise ValueError("V4.1 W4A4 requires an explicit SwiGLU limit")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    def quantize(value, stage):
        packed, token_scale = torch.ops.custom.flat_quant(
            value.reshape(value.shape[0], stage.left.shape[0], stage.right.shape[0]),
            stage.left,
            stage.right,
            clip_ratio=stage.clip,
            dst_dtype=16,
            group_list=counts,
            group_list_type=1,
        )
        return packed.flatten(1), token_scale.float()

    packed, token_scale = quantize(x, weights.first)
    gate_up = torch.empty(
        (x.shape[0], 2 * w.intermediate_size), device=x.device, dtype=x.dtype
    )
    grouped_w4a4(packed, weights.w13, state.scale13, token_scale, counts, gate_up)
    hidden = torch.empty(
        (x.shape[0], w.intermediate_size), device=x.device, dtype=x.dtype
    )
    _weighted_swiglu_active[(48,)](
        gate_up,
        route_weights,
        counts,
        hidden,
        w.intermediate_size,
        w.num_local_experts,
        x.shape[0],
        float(w.swiglu_arg.limit),
        48,
        triton.next_power_of_2(w.intermediate_size),
        num_warps=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )
    packed, token_scale = quantize(hidden, weights.second)
    out = torch.empty_like(x)
    return grouped_w4a4(packed, weights.w2, state.scale2, token_scale, counts, out)


def moe_apply(
    *,
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
    if not do_finalize or enable_pdl or topk_weights is None or topk_ids is None:
        raise ValueError("V4.1 W4A4 requires finalized precomputed routes without PDL")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("Routing IDs and weights must have matching shapes")
    if not x.shape[0]:
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
        multibuffer=False,
        enable_fp_fusion=False,
    )
    expanded_out = grouped_experts(expanded, weights, counts, w=w, plan=plan)
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
        multibuffer=False,
        enable_fp_fusion=False,
    )
    return out
