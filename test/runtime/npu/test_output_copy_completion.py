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

"""Exercise asynchronous output completion with eager and captured producers."""

import queue
import threading

import pytest


def test_copy_stream_result_completion():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() < 6:
        pytest.skip("requires six visible NPUs for distinct producer/commit contexts")
    from tokenspeed.runtime.execution.copy_completion import CopyStreamPool
    from tokenspeed.runtime.execution.types import ModelExecutionResult

    torch.npu.set_device(5)
    torch.set_grad_enabled(False)
    a = torch.randn((1024, 1024), dtype=torch.float16, device="npu:5")
    b = torch.randn_like(a)
    work = torch.empty_like(a)
    source = torch.zeros((8, 6), dtype=torch.int32, device="npu:5")
    target = torch.zeros_like(source)
    lengths = torch.zeros((8,), dtype=torch.int32, device="npu:5")

    def forward():
        for _ in range(4):
            torch.mm(a, b, out=work)
        target.copy_(source)
        lengths.copy_(source[:, 0])

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    producer = torch.npu.Stream(device=5)
    producer.wait_stream(torch.npu.current_stream())
    pool = CopyStreamPool(torch.npu, torch.device("npu:5"))
    results = []
    for captured in (False, True):
        pending = queue.Queue(maxsize=3)

        def issue():
            try:
                torch.npu.set_device(5)
                torch.set_grad_enabled(False)
                with torch.npu.stream(producer):
                    for i in range(128):
                        source.fill_(1000 + i)
                        graph.replay() if captured else forward()
                        completion = pool.copy_after(producer, (target, lengths))
                        with completion:
                            tokens_cpu = target.to("cpu", non_blocking=True)
                            lengths_cpu = lengths.to("cpu", non_blocking=True)
                        pending.put(
                            (
                                i,
                                ModelExecutionResult(
                                    output_tokens=tokens_cpu,
                                    output_lengths=lengths_cpu,
                                    copy_event=completion,
                                ),
                            )
                        )
            except BaseException as exc:
                pending.put(exc)
            finally:
                pending.put(None)

        thread = threading.Thread(target=issue)
        thread.start()
        while True:
            item = pending.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            i, result = item
            # Exercise host commit with a different current NPU device.
            torch.npu.set_device(0)
            result.sync()
            assert result.output_tokens.eq(1000 + i).all().item(), (
                captured,
                i,
                "tokens",
            )
            assert result.output_lengths.eq(1000 + i).all().item(), (
                captured,
                i,
                "lengths",
            )
            results.append((captured, i))
        thread.join()
    assert len(results) == 256
