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

"""Ascend 910 execution components for V4.1 target and DSpark.

Shared modeling owns the checkpoint contract, routing semantics and CED/DSpark
forward. This module owns device weight layouts, communication and warmup.
All compute continues through tokenspeed-kernel's registered operations.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from copy import copy

import torch
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.quantization import dequantize_mxfp_weight
from torch import nn

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.activation import SwigluArg
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.utils import (
    All2AllBackend,
    get_all2all_backend,
    get_moe_backend,
)
from tokenspeed.runtime.layers.moe.weights import create_layer_weights
from tokenspeed.runtime.layers.parameter import ModelWeightParameter
from tokenspeed.runtime.layers.quantization.utils import (
    should_exclude_quant_module,
    should_ignore_quant_layer,
)
from tokenspeed.runtime.models import deepseek_v41 as shared
from tokenspeed.runtime.models.deepseek_v41_dspark import (
    DeepseekV41DSparkModel as SharedDSparkModel,
)
from tokenspeed.runtime.models.deepseek_v41_dspark import (
    DeepseekV41ForCausalLMDSpark as SharedDSparkForCausalLM,
)
from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding
from tokenspeed.runtime.models.deepseek_v41_moe import DeepseekV4MLP
from tokenspeed.runtime.models.deepseek_v41_parallel import (
    create_decoder_cpu_group,
    decoder_global_rows,
    replicated_rows_moe,
    supports_ascend_dp_moe,
)
from tokenspeed.runtime.utils.device_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict


class Ascend910Fp8LinearMethod(shared._ReferenceFp8LinearMethod):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.load_as_bf16 = True

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        target_device = torch.empty(0).device
        # Ascend stores checkpoint codes as bytes; the ordinary loader still
        # selects TP shards. Postprocessing decodes only the local weight once.
        # A TP shard can cut a checkpoint scale block (2304 / TP16 = 144).
        # Repeat its exponent into smaller blocks without requantizing.
        self.quant_config = copy(self.quant_config)
        self.quant_config.weight_block_size = [
            1,
            math.gcd(32, input_size_per_partition),
        ]
        with torch.device("cpu") if target_device.type != "meta" else nullcontext():
            Fp8LinearMethod.create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
        codes = layer.weight
        layer.weight = ModelWeightParameter(
            data=torch.empty_like(codes.data, dtype=torch.uint8, device=target_device),
            input_dim=codes.input_dim,
            output_dim=codes.output_dim,
            weight_loader=codes.weight_loader,
        )
        layer.weight.loads_fp8_codes = True
        weight_loader = layer.weight.weight_loader

        def load_codes(param, loaded_weight, *shard_ids):
            if loaded_weight.dtype != torch.float8_e4m3fn:
                raise TypeError("V4.1 dense weights require FP8 E4M3 codes")
            weight_loader(param, loaded_weight.view(torch.uint8), *shard_ids)

        layer.weight._weight_loader = load_codes
        layer.weight_scale_inv.data = layer.weight_scale_inv.data.to(target_device)
        scale = layer.weight_scale_inv
        original_loader = scale.weight_loader

        def load_scale(
            param: nn.Parameter, loaded_weight: torch.Tensor, *shard_ids
        ) -> None:
            if (
                loaded_weight.dtype not in (torch.uint8, torch.float8_e8m0fnu)
                or loaded_weight.ndim != 2
            ):
                raise TypeError(
                    "V4.1 projection scales must be a 2D E8M0 checkpoint tensor"
                )
            if isinstance(layer, ReplicatedLinear):
                if loaded_weight.shape != ((param.shape[0] + 31) // 32, param.shape[1]):
                    raise ValueError(
                        "V4.1 projection scales must use checkpoint 32x32 blocks"
                    )
            expanded = loaded_weight.view(torch.uint8).repeat_interleave(32, dim=0)
            block_k = self.quant_config.weight_block_size[1]
            if block_k != 32:
                expanded = expanded.repeat_interleave(32 // block_k, dim=1)
            if isinstance(layer, ReplicatedLinear):
                # Engram can have a partial final output block.
                expanded = expanded[: param.shape[0]]
            original_loader(param, expanded, *shard_ids)

        scale._weight_loader = load_scale

    def process_weights_after_loading(self, layer):
        layer.weight.data = dequantize_mxfp_weight(
            layer.weight,
            layer.weight_scale_inv,
            bits=8,
            block_size=tuple(self.quant_config.weight_block_size),
        )
        layer.register_parameter("weight_scale_inv", None)
        layer.quant_method = UnquantizedLinearMethod()
        layer.quant_method.process_weights_after_loading(layer)


class Ascend910Fp8Config(shared._V41Fp8Config):
    def get_quant_method(self, layer, prefix):
        return Ascend910Fp8LinearMethod(self)


def dense_quant_config(quant_config):
    config = shared.v41_mxfp8_config(quant_config)
    if config is None:
        return None
    return Ascend910Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=config.ignored_layers,
        weight_block_size=[1, 32],
        scale_fmt="ue8m0",
    )


class Ascend910MLP(DeepseekV4MLP):
    def forward(self, x):
        if x.shape[0] == 0:
            return x.new_empty((0, self.down_proj.output_size))
        gate_up, _ = self.gate_up_proj(x)
        if gate_up.dtype != torch.bfloat16:
            return self.down_proj.forward_with_activation(gate_up, self.act_fn)[0]
        return self.down_proj(silu_and_mul(gate_up, limit=self.act_fn.swiglu_limit))[0]


class Ascend910Compressor(shared.DeepseekV41Compressor):
    def _compress(self, owner, content, scores, backend, mode):
        # LCM owns history; masked device operations avoid dynamic NonZero in ACL capture.
        active, positions, requests, previous, slots, write_slots = (
            backend.query_metadata(mode).compressor
        )
        tail = backend.cache_pool.compressor_tail(owner)
        pooled = dsv41.compressor_pool(
            content,
            scores,
            previous,
            tail,
            slots,
            active,
            None,
            self.norm.weight,
            self.norm.variance_epsilon,
        )
        dsv41.compressor_tail_scatter(content, scores, tail, write_slots)
        return pooled, positions, requests


class Ascend910Attention(shared.DeepseekV41Attention):
    stream_fork_cls = StreamFork
    compressor_cls = Ascend910Compressor

    def _configure_output_geometry(self, config, mapping):
        self.n_local_groups = max(1, config.o_groups // mapping.attn.tp_size)
        output_tp = max(1, mapping.attn.tp_size // config.o_groups)
        self.o_lora_rank = config.o_lora_rank // output_tp
        self.output_group: tuple[int, ...] = ()
        self.output_group_rank = mapping.attn.tp_rank % output_tp
        if output_tp > 1:
            if mapping.attn.tp_size % config.o_groups or config.o_lora_rank % output_tp:
                raise ValueError("Output-group sharding requires divisible Ascend TP")
            start = mapping.attn.tp_rank - self.output_group_rank
            self.output_group = mapping.attn.tp_group[start : start + output_tp]
            if torch.empty(0).device.type != "meta":
                pg_manager.init_process_group(self.output_group)

    def _group_attention_output(self, out: torch.Tensor) -> torch.Tensor:
        if self.output_group:
            # TP wider than o_groups splits one group's heads across ranks.
            # Assemble disjoint heads before GEMM so each output-row shard
            # retains the original full dot product and BF16 cast boundary.
            left = self.output_group_rank * self.n_local_heads
            right = (
                len(self.output_group) - self.output_group_rank - 1
            ) * self.n_local_heads
            out = shared.all_reduce(
                torch.nn.functional.pad(out, (0, 0, left, right)),
                group=self.output_group,
                backend=None,
                op=torch.distributed.ReduceOp.SUM,
            )
        return out.reshape(out.shape[0], self.n_local_groups, -1)

    def _can_overlap_index(self, hidden_states, rows):
        return rows.keep_rows is None

    def _output_rope_cache(self, q):
        return self.rotary_emb.inverse_cos_sin_cache


def _replicate_moe_rows(mapping):
    if get_all2all_backend() is All2AllBackend.ASCEND_MC2:
        return False
    enabled = supports_ascend_dp_moe(mapping, get_moe_backend().value, True)
    if enabled and global_server_args_dict["moe_combine_order"] != "rank":
        raise ValueError("Ascend V4.1 DP MoE requires --moe-combine-order rank")
    return enabled


class Ascend910Experts(nn.Module):
    """910 experts with model-owned preparation and ordinary checkpoint loaders.

    The module owns its plan and weights; there is no nested MoELayer. MC2 is
    warmed by the complete model forward with real distributed row geometry.
    """

    def __init__(
        self,
        *,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        quant_config,
        internal_activation_dtype_override,
        layer_index,
        prefix,
        tp_rank,
        tp_size,
        ep_rank,
        ep_size,
        activation,
        swiglu_limit,
        with_bias,
        routing_mode,
        routing_config,
        persistent_max_num_tokens_per_gpu,
    ):
        super().__init__()
        if activation != "swiglu" or with_bias or routing_mode != "precomputed_topk":
            raise ValueError(
                "V4.1 910 experts require bias-free SwiGLU with precomputed routing"
            )
        if tp_size > 1 and ep_size > 1 or num_experts % ep_size:
            raise ValueError(
                "V4.1 910 experts require unmixed TP/EP and equal expert shards"
            )
        if global_server_args_dict["moe_mxfp4_fp8_activation"]:
            raise ValueError("V4.1 910 experts do not implement FP8 activations")
        self.layer_index, self.prefix = layer_index, prefix
        self.top_k, self.num_experts = top_k, num_experts
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.tp_rank, self.tp_size, self.moe_tp_size = tp_rank, tp_size, tp_size
        self.ep_rank, self.ep_size = ep_rank, ep_size
        self.num_local_experts = num_experts // ep_size
        self.activation = activation
        self.swiglu_arg = SwigluArg(alpha=None, limit=swiglu_limit)
        self.swiglu_beta = None
        self.w13_input_layout = "concatenated"
        self.routing_config = routing_config
        self.input_dtype = torch.get_default_dtype()
        if self.input_dtype != torch.bfloat16:
            raise ValueError("V4.1 910 experts require BF16 input")
        self.quant_config = quant_config
        self._quant_kind = "unquant"
        if quant_config is not None and not (
            should_ignore_quant_layer(prefix, quant_config.ignored_layers)
            or should_exclude_quant_module(
                f"{prefix}.experts", quant_config.exclude_modules
            )
        ):
            self.quant_config = quant_config.get_moe_quant_config(prefix)
            self._quant_kind = self.quant_config.moe_weight_dtype(prefix)
        if self._quant_kind not in {"unquant", "mxfp4", "int4", "int8"}:
            raise ValueError(f"Unsupported V4.1 910 expert format: {self._quant_kind}")
        a2a = get_all2all_backend().value
        if a2a not in {"none", "ascend_mc2"}:
            raise ValueError(f"Unsupported V4.1 910 expert transport: {a2a}")
        self.combine_order = global_server_args_dict["moe_combine_order"]
        if self.combine_order != "rank":
            raise ValueError("V4.1 910 experts require rank combine")
        self._spec = MoELayerSpec(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            tp_rank=tp_rank,
            tp_size=tp_size,
            ep_rank=ep_rank,
            ep_size=ep_size,
            prefix=prefix,
            a2a_backend=a2a,
        )
        process_group = None
        if a2a == "ascend_mc2":
            process_group = pg_manager.get_dedicated_device_group(
                global_server_args_dict["mapping"].moe.ep_group, "ascend_mc2"
            )
        internal_activation_dtype = "input"
        if self._quant_kind == "mxfp4":
            if self.quant_config.is_w4a8_fp8:
                internal_activation_dtype = "fp8"
            elif self.quant_config.use_dynamic_mxfp4_activations:
                internal_activation_dtype = "mxfp4"
        if internal_activation_dtype_override is not None:
            internal_activation_dtype = internal_activation_dtype_override
        backend = get_moe_backend().value
        bitwise = global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        self.plan = moe_plan(
            self._quant_kind,
            input_dtype=self.input_dtype,
            activation=activation,
            routing_mode=routing_mode,
            a2a_backend=a2a,
            ep_size=ep_size,
            ispp=intermediate_size // tp_size,
            hidden=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            swiglu_form="standard",
            activation_clamped=swiglu_limit is not None,
            expert_id_repeats=False,
            internal_activation_dtype=internal_activation_dtype,
            with_bias=False,
            process_group=process_group,
            persistent_max_num_tokens_per_gpu=persistent_max_num_tokens_per_gpu,
            features=frozenset({"batch_invariant"}) if bitwise else frozenset(),
            solution=None if backend == "auto" else backend,
            fast_math=not bitwise,
            combine_order=self.combine_order,
        )
        create_layer_weights(
            self._spec,
            self,
            self._quant_kind,
            self.quant_config,
            with_bias=False,
            solution=self.plan["solution"],
        )
        self._weights_processed = False
        self._moe_backend_state: object | None = None

    def process_weights_after_loading(self, module):
        if self._weights_processed:
            return
        moe_process_weights(self.plan, module)
        self._weights_processed = True

    @property
    def topk_weights_dtype(self):
        return self.plan["topk_weights_dtype"]

    @property
    def topk_output_format(self):
        return TopKOutputFormat.STANDARD

    @property
    def supports_precomputed_topk(self):
        return self.plan["supports_precomputed_topk"]

    @property
    def supports_all_to_all_ep(self):
        return self.plan["supports_all_to_all_ep"]

    @property
    def supports_deferred_cuda_finalize(self):
        return False

    def forward(
        self,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
        *,
        do_finalize,
    ):
        if not do_finalize or topk_output.format.is_bypassed():
            raise ValueError("V4.1 910 experts require finalized precomputed routes")
        return moe_apply(
            self.plan,
            hidden_states,
            self,
            topk_output.router_logits,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            num_tokens_global=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=True,
        )


class Ascend910MoE(shared.DeepseekV41MoE):
    experts_cls = Ascend910Experts
    shared_expert_cls = Ascend910MLP

    def _replicate_dp_rows(self, mapping):
        return _replicate_moe_rows(mapping)

    def _placement_enabled(self, backend):
        return backend.value == "ascend_v41_w4a4"

    def _owns_ep_communication(self):
        return get_all2all_backend() is All2AllBackend.ASCEND_MC2


class Ascend910EngramEmbedding(RowShardedEngramEmbedding):
    device_weight_dtype = torch.uint8

    def _register_host_for_gpu(self):
        if self._host_gpu_registered or not self.host_table:
            return
        from tokenspeed_kernel.ops.embedding import prepare_host_mxfp8_embedding

        self._host_lookup = prepare_host_mxfp8_embedding(
            self.weight, self.scale, self.row_start, self.row_end
        )
        self._host_gpu_registered = True

    def _lookup(self, indices):
        if not self.host_table:
            return super()._lookup(indices)
        if self._host_lookup is None:
            raise RuntimeError("Host Engram mapping must be prepared before forward")
        return self._host_lookup(indices.to(torch.int64).contiguous())


class Ascend910Engram(shared.DeepseekV41Engram):
    embedding_cls = Ascend910EngramEmbedding

    def _table_mapping(self, mapping, device, host_table):
        from tokenspeed.runtime.models.deepseek_v41_parallel import (
            resolve_engram_table_mapping,
        )

        return resolve_engram_table_mapping(
            mapping,
            global_server_args_dict["engram_tp_size"],
            device_type="npu",
            host_table=host_table,
        )

    def _signed_magnitude(self, dot, magnitude):
        # copysign falls back to CPU on 910; preserve negative zero with the FP32 sign bit.
        return torch.where(dot.view(torch.int32) < 0, -magnitude, magnitude)


class Ascend910DecoderLayer(shared.DeepseekV41DecoderLayer):
    attention_cls = Ascend910Attention
    moe_cls = Ascend910MoE
    engram_cls = Ascend910Engram
    stream_fork_cls = StreamFork
    dense_quant_config = staticmethod(dense_quant_config)

    def _expert_aux_stream(self, aux_stream):
        # MC2 owns its shared-expert stream.
        return None

    def _forward_ffn(self, x, image_mask, ctx, *, defer_reduce):
        if self.ffn.dp_replicated_rows:
            return replicated_rows_moe(x, image_mask, ctx, self.ffn, self.comm_manager)
        return super()._forward_ffn(x, image_mask, ctx, defer_reduce=defer_reduce)


class Ascend910Model(shared.DeepseekV41Model):
    layer_cls = Ascend910DecoderLayer
    dense_quant_config = staticmethod(dense_quant_config)

    def _configure_parallelism(self, mapping):
        local_fused_ep = (
            get_all2all_backend() is All2AllBackend.ASCEND_MC2
            and mapping.attn.tp_size
            == mapping.dense.tp_size
            == mapping.moe.tp_size
            == 1
        )
        if get_all2all_backend() is All2AllBackend.ASCEND_MC2 and not local_fused_ep:
            raise ValueError("Ascend MC2 requires attention, dense and MoE TP1")
        self.dp_replicated_rows = _replicate_moe_rows(mapping)
        self.decoder_cpu_group = None
        if self.dp_replicated_rows:
            self.decoder_cpu_group = create_decoder_cpu_group(mapping)
        if mapping.attn.tp_size != mapping.moe.tp_ep_size and not (
            local_fused_ep or self.dp_replicated_rows
        ):
            raise NotImplementedError(
                "V4.1 requires attention TP == MoE TPxEP or fused EP with attention/dense/MoE TP1"
            )

    def _output_groups_divide(self, config, mapping):
        return config.o_groups % mapping.attn.tp_size == 0 or (
            mapping.attn.tp_size % config.o_groups == 0
            and config.o_lora_rank % (mapping.attn.tp_size // config.o_groups) == 0
        )

    def _decoder_counts(self, ctx):
        # Window-only drafts never perform the target's decoder-count exchange.
        if self.dp_replicated_rows and self.ced_decoder_start < len(self.layers):
            return decoder_global_rows(ctx, self.mapping, self.decoder_cpu_group)
        return None


class DeepseekV41ForCausalLM(shared.DeepseekV41ForCausalLM):
    model_cls = Ascend910Model
    supports_expert_placement = True

    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
        """Prepare the selected Ascend TP policy before cache profiling or capture."""
        from tokenspeed.runtime.distributed.comm_backend.hccl_tp import (
            prepare_hccl_tp_groups,
        )

        return prepare_hccl_tp_groups(
            self.mapping.attn.tp_group, nprocs_per_node=self.mapping.nprocs_per_node
        )

    def _decode_output_weight(self, linear, weight, scale, start, rows):
        expanded = scale.repeat_interleave(32, dim=0)[start % 32 : start % 32 + rows]
        return dequantize_mxfp_weight(
            weight.view(torch.uint8).to(linear.weight.device),
            expanded.to(linear.weight.device),
            bits=8,
            block_size=(1, 32),
        )

    def post_load_weights(self):
        super().post_load_weights()
        for module in self.modules():
            if isinstance(module, Ascend910Attention) and module.n_local_groups == 1:
                weight = module.wo_a.weight
                if weight.is_contiguous():
                    weight.data = weight.T.contiguous().T
            elif isinstance(module, Ascend910Experts):
                module.process_weights_after_loading(module)

    def post_quant_warmup(self):
        super().post_quant_warmup()
        # Codec constants are immutable data, not an executable kernel selection.
        from tokenspeed_kernel.ops.attention.dsv41.ascend import prepare_native_tables

        prepare_native_tables(torch.device("npu", torch.npu.current_device()))


class Ascend910DSparkModel(SharedDSparkModel, Ascend910Model):
    """Keep DSpark's stage/context protocol with the target's 910 components."""

    def local_base_logits(self, hidden_states, head):
        if head.ndim != 2 or head.dtype != hidden_states.dtype:
            raise ValueError(
                "DSpark base logits require a rank-2 head in the hidden dtype"
            )
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = shared.dsv4_linear_fp32(flat, head)
        return logits.view(*hidden_states.shape[:-1], head.shape[0])


class DeepseekV41ForCausalLMDSpark(SharedDSparkForCausalLM, DeepseekV41ForCausalLM):
    model_cls = Ascend910DSparkModel
