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


"""Mixed Cube/Vector packed-FP4 index scan for Ascend 910B.

Physical page equality is checked on device before sharing keys between two
query rows. Different requests, partial pages and replay-time page changes
retain independent scores. The native kernel preserves the existing BF16
rounding boundaries and rank-ordered shard reduction.
"""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _library():
    library = ctypes.CDLL(str(Path(__file__).with_name("libindex_scan_reduction.so")))
    library.launch_scan.argtypes = (
        [ctypes.c_void_p] * 7
        + [ctypes.c_int] * 7
        + [ctypes.c_int64]
        + [ctypes.c_void_p] * 3
    )
    library.launch_scan.restype = None
    return library


def scan_scores(q, weights, cache, table, visible, out, *, shards, parts, candidates):
    """Write live full-history scores into caller-owned FP32 output.

    Inputs are BF16 Q [T,32,128], BF16 weights [T,32], packed FP4 cache
    [pages,64,68], INT32 page table [T,P] and visible lengths [T].
    Parts is the number of disjoint historical partitions per query pair.
    Unsupported metadata is rejected before launch. Graph replay reads fresh
    page IDs and lengths; no host tensor reads or persistent request state.
    """
    n = q.shape[0]
    width = table.shape[1] * 64 if candidates is None else candidates.shape[1] * 8
    if (
        q.shape != (n, 32, 128)
        or weights.shape != (n, 32)
        or cache.ndim != 3
        or cache.shape[1:] != (64, 68)
        or table.shape[0] != n
        or visible.shape != (n,)
        or out.shape != (n, width)
    ):
        raise ValueError("Unsupported native index scan shape")
    if (
        q.dtype != torch.bfloat16
        or weights.dtype != torch.bfloat16
        or cache.dtype != torch.uint8
        or table.dtype != torch.int32
        or visible.dtype != torch.int32
        or out.dtype != torch.float32
    ):
        raise ValueError("Native index scan requires BF16 data and INT32 metadata")
    if candidates is not None and (
        candidates.shape[0] != n
        or candidates.dtype != torch.int32
        or candidates.device != q.device
    ):
        raise ValueError("Candidates must be INT32 [T,B] on the same device")
    candidates = None if candidates is None else candidates.contiguous()
    tensors = (q, weights, cache, table, visible, out)
    if q.device.type != "npu" or any(t.device != q.device for t in tensors):
        raise ValueError("Native index scan tensors must share an NPU device")
    if cache.stride(1) != 68 or cache.stride(2) != 1 or not out.is_contiguous():
        raise ValueError(
            "Index pages and output must be contiguous; the page stride may include other cache fields"
        )
    if shards not in (1, 2, 4, 8, 16, 32) or not 1 <= parts <= 24:
        raise ValueError("Unsupported index shard or partition count")
    if max(n, width, cache.shape[0]) >= 2**31:
        raise ValueError("Native index scan dimensions exceed INT32")
    if n == 0 or width == 0:
        return out
    library = _library()
    inputs = [
        q.contiguous(),
        weights.contiguous(),
        cache,
        table.contiguous(),
        visible.contiguous(),
        out,
    ]
    workspace = torch.empty(
        24 * 2 * (64 * 128 + 2 * 128 * 128), device=q.device, dtype=torch.bfloat16
    )
    tables = make_tables(q.device)
    inputs.append(workspace)
    library.launch_scan(
        *[ctypes.c_void_p(t.data_ptr()) for t in inputs],
        n,
        width,
        cache.shape[0],
        parts,
        shards,
        table.shape[1],
        int(candidates is not None),
        cache.stride(0),
        ctypes.c_void_p(tables.data_ptr()),
        ctypes.c_void_p(0 if candidates is None else candidates.data_ptr()),
        ctypes.c_void_p(torch.npu.current_stream(q.device).npu_stream),
    )
    return out


@lru_cache(maxsize=16)
def make_tables(device):
    codes = torch.arange(256, dtype=torch.int32)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = torch.cat((values, -values))
    pairs = (
        torch.stack((values[codes & 15], values[codes >> 4]), -1)
        .half()
        .view(torch.int32)
        .flatten()
    )
    scales = codes.byte().view(torch.float8_e8m0fnu).float().view(torch.int32)
    return torch.cat((pairs, scales)).to(device)


def scores(q, weights, cache, table, visible, candidates, width, shards):
    if q.shape[1] != 32 or weights.dtype != torch.bfloat16:
        raise ValueError("This prototype requires 32 heads and BF16 weights")
    output = torch.full(
        (q.shape[0], width), -torch.inf, dtype=torch.float32, device=q.device
    )
    parts = max(
        1, min(24, (24 + (q.shape[0] + 1) // 2 - 1) // max(1, (q.shape[0] + 1) // 2))
    )
    return scan_scores(
        q,
        weights,
        cache,
        table,
        visible,
        output,
        shards=shards,
        parts=parts,
        candidates=candidates,
    )
