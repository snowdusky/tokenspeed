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

"""Collective alias compatibility using real Gloo operations on CPU."""

from functools import wraps

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel_npu import register_torch_compat


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    rendezvous = tmp_path_factory.mktemp("collective_compat") / "rendezvous"
    dist.init_process_group(
        "gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("existing_gather", [False, True])
@pytest.mark.parametrize("existing_scatter", [False, True])
@pytest.mark.parametrize("async_op", [False, True])
def test_collective_aliases_preserve_execution_and_work(
    monkeypatch, group, existing_gather, existing_scatter, async_op
):
    operations = (
        ("all_gather_single", "all_gather_into_tensor", existing_gather),
        ("reduce_scatter_single", "reduce_scatter_tensor", existing_scatter),
    )
    returned_work = {}

    def tracked(function, name):
        @wraps(function)
        def run(*args, **kwargs):
            work = function(*args, **kwargs)
            returned_work[name] = work
            return work

        return run

    expected = {}
    for name, legacy, existing in operations:
        legacy_function = getattr(dist, legacy)
        monkeypatch.setattr(dist, legacy, tracked(legacy_function, name))
        # A distinct existing implementation must survive compatibility setup.
        monkeypatch.setattr(dist, name, tracked(legacy_function, name), raising=False)
        expected[name] = getattr(dist, name if existing else legacy)
        if not existing:
            monkeypatch.delattr(dist, name)
    register_torch_compat()
    register_torch_compat()
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    for name, _, _ in operations:
        output = torch.empty_like(source)
        function = getattr(dist, name)
        assert function is expected[name]
        work = function(output, source, group=group, async_op=async_op)
        assert work is returned_work[name]
        if async_op:
            assert work is not None
            work.wait()
        else:
            assert work is None
        torch.testing.assert_close(output, source, atol=0, rtol=0)
