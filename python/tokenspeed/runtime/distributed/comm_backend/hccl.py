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

"""HCCL communication backend."""

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

AIV_TP_GROUP_NAME = "ascend_tp_aiv"


@dataclass(frozen=True)
class HcclTpPolicy:
    """Explicit BF16 TP reduction numerics and independent transport choices.

    The policy applies to every collective on the configured rank group,
    including Dense/LM-head groups when they share those ranks. The message
    size limit selects transport only; it never changes accumulation dtype.
    """

    reduce_dtype: Literal["input", "float32"]
    reduce_algorithm: Literal["host", "aiv"]
    gather_algorithm: Literal["host", "aiv"]
    aiv_max_bytes: int

    def __post_init__(self) -> None:
        if self.reduce_dtype not in ("input", "float32"):
            raise ValueError(
                f"Unsupported HCCL TP reduction dtype: {self.reduce_dtype}"
            )
        if self.reduce_algorithm not in ("host", "aiv"):
            raise ValueError(f"Unsupported HCCL TP reduction: {self.reduce_algorithm}")
        if self.gather_algorithm not in ("host", "aiv"):
            raise ValueError(f"Unsupported HCCL TP gather: {self.gather_algorithm}")
        if self.aiv_max_bytes < 0 or (self.uses_aiv and self.aiv_max_bytes == 0):
            raise ValueError("AIV requires a positive message size limit")

    @property
    def uses_aiv(self) -> bool:
        return "aiv" in (self.reduce_algorithm, self.gather_algorithm)

    def validate_group(self, group: Group, *, nprocs_per_node: int) -> None:
        if not self.uses_aiv:
            return
        if (
            len(group) not in (2, 4, 8)
            or group != tuple(range(group[0], group[0] + len(group)))
            or group[0] % len(group) != 0
            or len({rank // nprocs_per_node for rank in group}) != 1
        ):
            raise ValueError(
                "The validated AIV transport requires one aligned local TP2/TP4/TP8 group"
            )


def prepare_hccl_tp_groups(group: Group, *, nprocs_per_node: int) -> None:
    """Prepare an explicitly selected HCCL TP policy before warmup and capture.

    ProcessGroupManager owns communicator creation and lifetime. Without a
    policy the ordinary HCCL implementation is unchanged. All policy fields
    are required together so enabling transport cannot implicitly change dtype.
    """
    from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend

    names = (
        "TOKENSPEED_HCCL_TP_REDUCE_DTYPE",
        "TOKENSPEED_HCCL_TP_REDUCE_ALGORITHM",
        "TOKENSPEED_HCCL_TP_GATHER_ALGORITHM",
        "TOKENSPEED_HCCL_TP_AIV_MAX_BYTES",
    )
    values = [os.environ.get(name) for name in names]
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(f"Specify all HCCL TP policy settings: {', '.join(names)}")
    policy = HcclTpPolicy(values[0], values[1], values[2], int(values[3]))
    policy.validate_group(group, nprocs_per_node=nprocs_per_node)
    backend = get_global_backend()
    if not isinstance(backend, HcclBackend):
        raise TypeError("HCCL TP policy requires the HCCL communication backend")
    if policy.uses_aiv:
        import torch_npu

        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_op_expansion_mode": 3}
        pg_manager.init_process_group(
            group, backend="hccl", name=AIV_TP_GROUP_NAME, pg_options=options
        )
    backend.configure_tp(group, policy=policy)


class HcclBackend(CommBackend):
    """Collectives implemented by torch.distributed over HCCL."""

    def __init__(self):
        self._tp_policies: dict[Group, HcclTpPolicy] = {}

    def configure_tp(self, group: Group, *, policy: HcclTpPolicy) -> None:
        """Bind an immutable policy to an already prepared communicator."""
        if group in self._tp_policies and self._tp_policies[group] != policy:
            raise ValueError("Cannot change an initialized HCCL TP policy")
        if policy.uses_aiv:
            pg_manager.get_process_group("hccl", group, name=AIV_TP_GROUP_NAME)
        self._tp_policies[group] = policy

    @staticmethod
    def _process_group(group: Group):
        return pg_manager.get_process_group("hccl", group)

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            return super().all_reduce(tensor, group, op=op)
        if len(group) > 1:
            policy = self._tp_policies.get(group)
            if (
                policy is not None
                and tensor.dtype == torch.bfloat16
                and op in (None, dist.ReduceOp.SUM)
            ):
                reduced = (
                    tensor.float().contiguous()
                    if policy.reduce_dtype == "float32"
                    else tensor
                )
                process_group = (
                    pg_manager.get_process_group("hccl", group, name=AIV_TP_GROUP_NAME)
                    if policy.reduce_algorithm == "aiv"
                    and 0
                    < reduced.numel() * reduced.element_size()
                    <= policy.aiv_max_bytes
                    else self._process_group(group)
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=process_group)
                if reduced is not tensor:
                    tensor.copy_(reduced)
                return tensor
            if (
                tensor.dtype == torch.int64
                and op == dist.ReduceOp.MAX
                and 0 < tensor.numel() * tensor.element_size() <= 4096
            ):
                # CANN's AIV AllReduce does not support INT64. Gather the
                # unchanged bits as INT32, then select the exact local maximum.
                # Bound the gather to small payloads such as packed argmax keys.
                flat = tensor.reshape(-1)
                if flat.stride(0) != 1:
                    # contiguous() can retain a single-element view's stride.
                    flat = flat.clone(memory_format=torch.contiguous_format)
                gathered = tensor.new_empty((len(group), flat.numel()))
                self.all_gather_single(
                    gathered.view(torch.int32).view(-1),
                    flat.view(torch.int32),
                    group,
                )
                torch.amax(gathered, dim=0, out=flat)
                if flat.data_ptr() != tensor.data_ptr():
                    tensor.copy_(flat.view_as(tensor))
                return tensor
            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM if op is None else op,
                group=self._process_group(group),
            )
        return tensor

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        if len(group) == 1:
            return tensor
        if dim < 0:
            dim += tensor.dim()

        input_size = tensor.size()
        gathered = torch.empty(
            (input_size[0] * len(group),) + input_size[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        self.all_gather_single(gathered, tensor, group)
        return (
            gathered.reshape((len(group),) + input_size)
            .movedim(0, dim)
            .reshape(
                input_size[:dim]
                + (len(group) * input_size[dim],)
                + input_size[dim + 1 :]
            )
        )

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        if len(group) == 1:
            output.copy_(input)
            return
        policy = self._tp_policies.get(group)
        process_group = (
            pg_manager.get_process_group("hccl", group, name=AIV_TP_GROUP_NAME)
            if policy is not None
            and policy.gather_algorithm == "aiv"
            and input.dtype == torch.bfloat16
            and input.numel() > 0
            and output.numel() * output.element_size() <= policy.aiv_max_bytes
            else self._process_group(group)
        )
        dist.all_gather_into_tensor(output, input, group=process_group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        if len(group) == 1:
            return tensor
        output = torch.empty(
            (tensor.shape[0] // len(group),) + tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.reduce_scatter_tensor(output, tensor, group=self._process_group(group))
        return output

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: Group,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> None:
        if len(group) == 1:
            output.copy_(input)
            return
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self._process_group(group),
        )

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        max_tokens = max(scattered_num_tokens)
        if tensor.shape[0] < max_tokens:
            tensor = torch.cat(
                (
                    tensor,
                    tensor.new_zeros(max_tokens - tensor.shape[0], tensor.shape[-1]),
                )
            )

        gathered = tensor.new_empty(len(group) * max_tokens, tensor.shape[-1])
        self.all_gather_single(gathered, tensor.contiguous(), group)
        # Equal-sized shards already form the contiguous, rank-ordered output.
        if all(tokens == max_tokens for tokens in scattered_num_tokens):
            return gathered
        # With trailing idle ranks, live rows may already be a contiguous
        # prefix. Preserve the collective's padding; only trim its result.
        last_live_rank = max(
            (rank for rank, count in enumerate(scattered_num_tokens) if count),
            default=-1,
        )
        if all(count == max_tokens for count in scattered_num_tokens[:last_live_rank]):
            return gathered[: sum(scattered_num_tokens)]
        return torch.cat(
            [
                gathered[rank * max_tokens : rank * max_tokens + tokens]
                for rank, tokens in enumerate(scattered_num_tokens)
            ]
        )

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        """Reduce rank-major rows, padding only unequal token counts."""
        max_tokens = max(scattered_num_tokens)
        # Equal rank-major slices already have the collective's input layout.
        if all(tokens == max_tokens for tokens in scattered_num_tokens):
            return self.reduce_scatter(tensor.contiguous(), group)
        padded = tensor.new_zeros(
            len(group) * max_tokens,
            tensor.shape[-1],
        )
        offset = 0
        for rank, tokens in enumerate(scattered_num_tokens):
            padded[rank * max_tokens : rank * max_tokens + tokens].copy_(
                tensor[offset : offset + tokens]
            )
            offset += tokens

        output = tensor.new_empty(max_tokens, tensor.shape[-1])
        if len(group) == 1:
            output.copy_(padded)
        else:
            dist.reduce_scatter_tensor(
                output,
                padded.contiguous(),
                group=self._process_group(group),
            )
        rank = group.index(dist.get_rank())
        return output[: scattered_num_tokens[rank]].contiguous()

    def send(self, tensor: torch.Tensor, dst: int, group: Group) -> None:
        dist.send(tensor, group[dst], group=self._process_group(group))

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        group: Group,
    ) -> torch.Tensor:
        tensor = torch.empty(size, dtype=dtype, device=device)
        dist.recv(tensor, group[src], group=self._process_group(group))
        return tensor
