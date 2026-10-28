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

"""Verify the production result boundary with multiple NPU forwards in flight."""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from tokenspeed.runtime.execution.types import ModelExecutionResult


@pytest.mark.parametrize("pinned", [False, True])
def test_output_completion_preserves_each_generation(pinned):
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(device)
    world = dist.get_world_size() if dist.is_initialized() else 1
    stream = torch.npu.Stream()
    source = torch.zeros(256, dtype=torch.int32, device="npu")
    packed = torch.empty_like(source)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            packed.copy_(source)
            if world > 1:
                dist.all_reduce(packed)
    torch.npu.synchronize()

    def forward(generation):
        torch.npu.set_device(device)
        with torch.npu.stream(stream):
            source.fill_(generation)
            graph.replay()
            # Exercise both persistent graph buffers and temporary eager outputs.
            value = packed if generation % 2 == 0 else packed.clone()
            if pinned:
                host = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                host.copy_(value, non_blocking=True)
            else:
                host = value.to("cpu", non_blocking=True)
            event = torch.npu.Event()
            event.record()
            return ModelExecutionResult(output_tokens=host, copy_event=event)

    def check(generation, pending):
        result = pending.result()
        result.sync()
        torch.testing.assert_close(
            result.output_tokens,
            torch.full((256,), generation * world, dtype=torch.int32),
            atol=0,
            rtol=0,
        )

    try:
        # Like the Rust host, commit runs on a thread that did not initialize
        # the accelerator or submit the forward.
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            ThreadPoolExecutor(max_workers=1) as commit,
        ):
            pending = deque()
            for generation in range(1, 301):
                pending.append((generation, pool.submit(forward, generation)))
                if len(pending) >= 4:
                    commit.submit(check, *pending.popleft()).result()
            while pending:
                commit.submit(check, *pending.popleft()).result()
    finally:
        torch.npu.synchronize()
        graph.reset()
