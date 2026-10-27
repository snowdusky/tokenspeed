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

"""AscendC post mixing candidate, preserving sequential FP32 accumulation."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libmhc_post.so")))
    fn = lib.launch_mhc_post
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    fn.restype = None
    return lib, fn


def post_mapping(hidden_states, residual, post, comb, *, block):
    if block not in (512, 1024, 2048):
        raise ValueError("unsupported post block size")
    if residual.dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("post mixing requires BF16 residual and hidden states")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise ValueError("post mixing requires FP32 coefficients")
    if any(not x.is_contiguous() for x in (hidden_states, residual, post, comb)):
        raise ValueError("post mixing requires contiguous inputs")
    hidden = residual.shape[-1]
    rows = residual.numel() // (4 * hidden)
    if residual.shape[-2] != 4 or hidden % 16:
        raise ValueError("post mixing requires four streams and aligned hidden size")
    if (
        hidden_states.numel() != rows * hidden
        or post.numel() != rows * 4
        or comb.numel() != rows * 16
    ):
        raise ValueError("post mixing input sizes disagree")
    result = torch.empty_like(residual)
    if rows:
        _, fn = _library()
        fn(
            hidden_states.data_ptr(),
            residual.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            result.data_ptr(),
            rows,
            hidden,
            block,
            torch.npu.current_stream().npu_stream,
        )
    return result
