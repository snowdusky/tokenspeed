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


from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def grouped_merge(
    P,
    S,
    Sink,
    O,
    L,
    heads: tl.constexpr,
    parts: tl.constexpr,
    PS: tl.constexpr,
    H: tl.constexpr,
    HEAD_CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * H + tl.arange(0, H)
    p = tl.arange(0, PS)
    d = tl.arange(0, 512)
    sums = tl.load(
        S + (row * parts + p[:, None]) * (2 * HEAD_CAPACITY) + head[None, :],
        (p[:, None] < parts) & (head[None, :] < heads),
        other=0,
    )
    maxima = tl.load(
        S
        + (row * parts + p[:, None]) * (2 * HEAD_CAPACITY)
        + HEAD_CAPACITY
        + head[None, :],
        (p[:, None] < parts) & (head[None, :] < heads),
        other=-float("inf"),
    )
    maxima = tl.where(sums > 0, maxima, -float("inf"))
    sink = tl.load(Sink + head, head < heads, other=-float("inf"))
    m = tl.maximum(tl.max(maxima, 0), sink)
    m = tl.where(m == -float("inf"), 0.0, m)
    w = tl.where(sums > 0, tl.exp(maxima - m[None, :]), 0.0)
    den = tl.sum(sums * w, 0) + tl.exp(sink - m)
    value = tl.load(
        P
        + ((row * parts + p[:, None, None]) * HEAD_CAPACITY + head[None, :, None]) * 512
        + d[None, None, :],
        (p[:, None, None] < parts) & (head[None, :, None] < heads),
        other=0.0,
    )
    y = tl.sum(value * w[:, :, None], 0) / tl.where(den[:, None] > 0, den[:, None], 1.0)
    tl.store(
        O + (row * heads + head[:, None]) * 512 + d[None, :], y, head[:, None] < heads
    )
    tl.store(
        L + row * heads + head,
        tl.where(den > 0, tl.log(den) + m, -float("inf")),
        head < heads,
    )
