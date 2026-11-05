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

"""Sqrt-softplus router top-k on the Ascend vector unit.

Same contract as ``torch_sqrt_softplus_topk``: FP32 sqrt-softplus scores,
selection-only correction bias, renormalisation and routed scaling, int32 ids
and caller-selected weight dtype. Hash routing is not implemented here and the
caller must fall back.
"""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

MAX_EXPERTS = 512
MAX_TOP_K = 8

_IN_DTYPES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_OUT_DTYPES = {torch.float32: 0, torch.bfloat16: 1}


@lru_cache(None)
def _library():
    path = Path(__file__).with_name("librouter_topk.so")
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    lib.launch_router_topk.restype = None
    lib.launch_router_topk.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_int] * 3
        + [ctypes.c_float]
        + [ctypes.c_int] * 3
        + [ctypes.c_void_p]
    )
    return lib


def available() -> bool:
    """Return whether the compiled provider is present."""
    return _library() is not None


def supported(
    logits: torch.Tensor, top_k: int, correction_bias: torch.Tensor | None
) -> bool:
    """Return whether the native kernel can serve this geometry exactly."""
    if not available() or logits.ndim != 2 or not logits.is_contiguous():
        return False
    rows, experts = logits.shape
    if logits.dtype not in _IN_DTYPES:
        return False
    if not 0 < top_k <= min(experts, MAX_TOP_K) or experts > MAX_EXPERTS:
        return False
    if experts < 8 or experts % 8:
        return False
    del rows
    if correction_bias is not None and tuple(correction_bias.shape) != (experts,):
        return False
    return True


def router_topk_sqrt_softplus(
    logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select experts and weights; None when the caller must use a fallback."""
    lib = _library()
    if (
        lib is None
        or weights_dtype not in _OUT_DTYPES
        or not supported(logits, top_k, correction_bias)
    ):
        return None
    rows, experts = logits.shape
    ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    weights = torch.empty((rows, top_k), dtype=weights_dtype, device=logits.device)
    if rows == 0:
        return weights, ids
    bias_ptr = 0
    if correction_bias is not None:
        # The kernel ranks in FP32, so the bias is narrowed once per call.
        bias = correction_bias.to(dtype=torch.float32).contiguous()
        bias_ptr = bias.data_ptr()
    lib.launch_router_topk(
        logits.data_ptr(),
        bias_ptr,
        ids.data_ptr(),
        weights.data_ptr(),
        rows,
        experts,
        top_k,
        float(routed_scaling_factor),
        int(bool(renormalize)),
        _IN_DTYPES[logits.dtype],
        _OUT_DTYPES[weights_dtype],
        torch.npu.current_stream().npu_stream,
    )
    return weights, ids
