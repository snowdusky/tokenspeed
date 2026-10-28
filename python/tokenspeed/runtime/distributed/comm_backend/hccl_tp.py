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

"""Opt-in Ascend TP communication policy, prepared by the model at startup."""

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.mapping import Group
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


def prepare_hccl_tp_groups(group: Group, *, nprocs_per_node: int) -> bool:
    """Prepare an explicitly selected HCCL TP policy before warmup and capture.

    ProcessGroupManager owns communicator creation and lifetime. Without a
    policy the ordinary HCCL implementation is unchanged. All policy fields
    are required together so enabling transport cannot implicitly change dtype.
    """
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        get_global_backend,
        set_global_backend,
    )

    names = (
        "TOKENSPEED_HCCL_TP_REDUCE_DTYPE",
        "TOKENSPEED_HCCL_TP_REDUCE_ALGORITHM",
        "TOKENSPEED_HCCL_TP_GATHER_ALGORITHM",
        "TOKENSPEED_HCCL_TP_AIV_MAX_BYTES",
    )
    values = [os.environ.get(name) for name in names]
    if all(value is None for value in values):
        return False
    if any(value is None for value in values):
        raise ValueError(f"Specify all HCCL TP policy settings: {', '.join(names)}")
    # Rank emulation uses a single-rank communicator for logical TP groups.
    if pg_manager.get_device_process_group(group).size() != len(group):
        return False
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
    if type(backend) is HcclBackend:
        backend = HcclTpBackend()
    elif not isinstance(backend, HcclTpBackend):
        raise TypeError("Cannot replace another HCCL backend's communication policy")
    backend.configure_tp(group, policy=policy)
    set_global_backend(backend)
    return True


class HcclTpBackend(HcclBackend):
    """Specialize selected TP groups; inherit ordinary HCCL for all other calls."""

    def __init__(self):
        self._tp_policies: dict[Group, HcclTpPolicy] = {}

    def configure_tp(self, group: Group, *, policy: HcclTpPolicy) -> None:
        """Bind an immutable policy to an already prepared communicator."""
        if group in self._tp_policies and self._tp_policies[group] != policy:
            raise ValueError("Cannot change an initialized HCCL TP policy")
        if policy.uses_aiv:
            pg_manager.get_process_group("hccl", group, name=AIV_TP_GROUP_NAME)
        self._tp_policies[group] = policy

    def all_reduce(self, tensor, group: Group, op=None):
        if not isinstance(tensor, torch.Tensor) or len(group) == 1:
            return super().all_reduce(tensor, group, op=op)
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
                and 0 < reduced.numel() * reduced.element_size() <= policy.aiv_max_bytes
                else self._process_group(group)
            )
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=process_group)
            if reduced is not tensor:
                tensor.copy_(reduced)
            return tensor
        return super().all_reduce(tensor, group, op=op)

    def all_gather_single(self, output, input, group: Group) -> None:
        if len(group) == 1:
            return super().all_gather_single(output, input, group)
        policy = self._tp_policies.get(group)
        if (
            policy is not None
            and policy.gather_algorithm == "aiv"
            and input.dtype == torch.bfloat16
            and input.numel() > 0
            and output.numel() * output.element_size() <= policy.aiv_max_bytes
        ):
            process_group = pg_manager.get_process_group(
                "hccl", group, name=AIV_TP_GROUP_NAME
            )
            dist.all_gather_single(output, input, group=process_group)
            return
        super().all_gather_single(output, input, group)
