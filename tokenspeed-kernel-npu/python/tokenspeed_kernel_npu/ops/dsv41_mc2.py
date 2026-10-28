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

"""Hierarchy MC2 transport for V4.1 FlatQuant experts.

Route IDs and FP32 weights travel as raw bits beside BF16 activations. MC2
does not round those weights or apply them after Down: grouped_experts applies
them at the model's pre-Down boundary. All counts stay on the device.
"""

from dataclasses import dataclass

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_w4a4 import grouped_experts
from tokenspeed_kernel_npu.ops.dsv41_w4a4 import process_weights as prepare_experts

# Bound each sender's burst: concentrated routes stall the native hierarchy
# with larger batches even when its communication window has enough space.
# Larger forwards split within the provider, not the scheduler.
_MAX_DISPATCH_ROWS = 16


@triton.jit
def _pack(
    X,
    IDs,
    Weights,
    Carrier,
    Bits,
    H: tl.constexpr,
    K: tl.constexpr,
    XS,
    IS,
    WS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row * XS + col, col < H, 0)
    tl.store(Carrier + row * (H + 64) + col, values, col < H)
    if tl.program_id(1) == 0:
        k = tl.arange(0, 32)
        ids = tl.load(IDs + row * IS + k, k < K, 0).to(tl.int32)
        # A fixed 64-BF16 suffix leaves room for up to eight routes.
        metadata = tl.where(k < K, ids, 0)
        weight_bits = (
            tl.load(Weights + row * WS + k - K, (k >= K) & (k < 2 * K), 0)
            .to(tl.float32)
            .to(tl.int32, bitcast=True)
        )
        metadata = tl.where((k >= K) & (k < 2 * K), weight_bits, metadata)
        tl.store(Bits + row * ((H + 64) // 2) + H // 2 + k, metadata)


@triton.jit
def _unpack(
    X,
    Bits,
    Counts,
    Y,
    Weights,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    R,
    BLOCK: tl.constexpr,
):
    e = tl.arange(0, triton.next_power_of_2(E))
    counts = tl.load(Counts + e, e < E, 0).to(tl.int32)
    ends = tl.cumsum(counts)
    active = tl.sum(counts)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, triton.next_power_of_2(K))
    for row in range(tl.program_id(0), active, 32):
        values = tl.load(X + row * (H + 64) + col, col < H, 0)
        tl.store(Y + row * H + col, values, col < H)
        if tl.program_id(1) == 0:
            expert = tl.sum(((row >= ends) & (e < E)).to(tl.int32)) + R * E
            ids = tl.load(Bits + row * ((H + 64) // 2) + H // 2 + ks, ks < K, -1)
            weights = tl.load(
                Bits + row * ((H + 64) // 2) + H // 2 + K + ks, ks < K, 0
            ).to(tl.float32, bitcast=True)
            tl.store(Weights + row, tl.sum(tl.where(ids == expert, weights, 0)))


@dataclass(frozen=True)
class MC2Transport:
    group: str
    rank: int
    world_size: int
    experts: int
    hidden_size: int
    top_k: int

    @property
    def max_dispatch_rows(self):
        return _MAX_DISPATCH_ROWS

    def arguments(self, max_rows):
        return dict(
            group_ep=self.group,
            group_tp="",
            ep_world_size=self.world_size,
            tp_world_size=0,
            ep_rank_id=self.rank,
            tp_rank_id=0,
            moe_expert_num=self.experts,
            shared_expert_num=0,
            shared_expert_rank_num=0,
            global_bs=max(1, max_rows) * self.world_size,
            comm_alg="hierarchy",
        )

    def dispatch(self, x, ids, weights, *, max_rows):
        """Return contiguous expert rows, exact route weights and a combine handle."""
        if not 0 <= max_rows <= self.max_dispatch_rows:
            raise ValueError(
                f"MC2 dispatch requires a row capacity between 0 and {self.max_dispatch_rows}"
            )
        if (
            x.shape[0] > max_rows
            or ids.shape != weights.shape
            or ids.shape != (x.shape[0], self.top_k)
        ):
            raise ValueError("Invalid MC2 row capacity or routing shape")
        if (
            x.dtype != torch.bfloat16
            or weights.dtype != torch.float32
            or x.shape[1] != self.hidden_size
        ):
            raise ValueError("MC2 requires BF16 activations and FP32 routing weights")
        if not 0 < self.top_k <= 8:
            raise ValueError("MC2 route carrier supports one to eight distinct experts")
        ids = ids.to(torch.int32).contiguous()
        weights = weights.contiguous()
        # The native hierarchy requires uniform row capacity across ranks.
        # Padding contributes zero before Down and is discarded after Combine.
        original_rows = x.shape[0]
        capacity = max(1, max_rows)
        if original_rows < capacity:
            padded_x = x.new_zeros((capacity, self.hidden_size))
            padded_x[:original_rows].copy_(x)
            # Spread zero-weight routes across the EP group instead of
            # directing every idle rank to the same few experts.
            padded_ids = (
                (
                    (
                        torch.arange(self.top_k, device=x.device, dtype=torch.int32)
                        + self.rank * (self.experts // self.world_size)
                    )
                    % self.experts
                )[None]
                .expand(capacity, -1)
                .clone()
            )
            padded_ids[:original_rows].copy_(ids)
            padded_weights = weights.new_zeros((capacity, self.top_k))
            padded_weights[:original_rows].copy_(weights)
            x, ids, weights = padded_x, padded_ids, padded_weights
        carrier = x.new_empty((x.shape[0], self.hidden_size + 64))
        _pack[(x.shape[0], triton.cdiv(self.hidden_size, 256))](
            x,
            ids,
            weights,
            carrier,
            carrier.view(torch.int32),
            self.hidden_size,
            self.top_k,
            x.stride(0),
            ids.stride(0),
            weights.stride(0),
            256,
            num_warps=1,
            multibuffer=False,
            enable_fp_fusion=False,
        )
        ones = torch.ones_like(weights)
        args = self.arguments(max_rows)
        result = torch.ops.custom.npu_moe_distribute_dispatch_v2(
            x=carrier,
            expert_ids=ids,
            expert_scales=ones,
            quant_mode=0,
            expert_token_nums_type=1,
            **args,
        )
        sorted_x = x.new_empty((result[0].shape[0], self.hidden_size))
        sorted_weights = weights.new_empty((result[0].shape[0],))
        _unpack[(32, triton.cdiv(self.hidden_size, 256))](
            result[0],
            result[0].view(torch.int32),
            result[3],
            sorted_x,
            sorted_weights,
            self.hidden_size,
            self.top_k,
            self.experts // self.world_size,
            self.rank,
            256,
            num_warps=1,
            multibuffer=False,
            enable_fp_fusion=False,
        )
        return (
            sorted_x,
            sorted_weights,
            result[3],
            (result, ids, ones, x, args, original_rows),
        )

    def combine(self, output, handle):
        result, ids, ones, x, args, original_rows = handle
        combined = torch.ops.custom.npu_moe_distribute_combine_v2(
            expand_x=output,
            expert_ids=ids,
            assist_info_for_combine=result[2],
            ep_send_counts=result[4],
            tp_send_counts=result[5],
            expert_scales=ones,
            expand_scales=result[6],
            ori_x=x,
            comm_quant_mode=0,
            **args,
        )
        return combined[:original_rows]


def process_weights(*, plan, w):
    if (
        w.tp_size != 1
        or w.ep_size not in (32, 64, 128)
        or plan["combine_order"] != "rank"
    ):
        raise ValueError("V4.1 MC2 requires TP1, EP32/64/128 and rank combine")
    group = plan["process_group"]
    if group is None or group.size() != w.ep_size or group.rank() != w.ep_rank:
        raise ValueError("MC2 requires the matching EP device process group")
    prepare_experts(plan=plan, w=w)
    name = group._get_backend(w.w13_weight.device).get_hccl_comm_name(w.ep_rank)
    plan["mc2_transport"] = MC2Transport(
        name, w.ep_rank, w.ep_size, w.num_experts, w.hidden_size, w.top_k
    )


def moe_apply(
    *,
    plan,
    x,
    w,
    router_logits,
    topk_weights,
    topk_ids,
    num_tokens_global,
    max_num_tokens_per_gpu,
    do_finalize,
    enable_pdl,
    low_latency,
    overlap_fn,
):
    if not do_finalize or enable_pdl or topk_weights is None or topk_ids is None:
        raise ValueError("MC2 requires finalized precomputed routes without PDL")
    if low_latency is not None or overlap_fn is not None:
        raise ValueError(
            "MC2 hierarchy does not expose DeepEP modes or overlap callbacks"
        )
    if max_num_tokens_per_gpu is None or max_num_tokens_per_gpu < x.shape[0]:
        raise ValueError("MC2 needs a rank-consistent capacity covering all local rows")
    transport = plan["mc2_transport"]
    outputs = []
    capacity = max(1, max_num_tokens_per_gpu)
    # All ranks execute the same number of collectives, including ranks whose
    # local slice is empty. Token quantization and routing remain independent.
    for start in range(0, capacity, transport.max_dispatch_rows):
        end = min(start + transport.max_dispatch_rows, capacity)
        rows, weights, counts, handle = transport.dispatch(
            x[start:end],
            topk_ids[start:end],
            topk_weights[start:end],
            max_rows=end - start,
        )
        output = grouped_experts(rows, weights, counts, w=w, plan=plan)
        outputs.append(transport.combine(output, handle))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
