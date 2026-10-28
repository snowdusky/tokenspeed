"""BF16 SwiGLU with FP32 intermediates and one final BF16 rounding."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libshared_swiglu_general.so")))
    lib.launch_shared_activation.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    )
    lib.launch_shared_activation.restype = None
    return lib


def silu_and_mul(x, out=None, *, limit):
    if (
        x.ndim < 1
        or x.shape[-1] % 2
        or x.dtype != torch.bfloat16
        or x.device.type != "npu"
    ):
        raise ValueError("Expected BF16 NPU [..., gate|up]")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    h = x.shape[-1] // 2
    shape = (*x.shape[:-1], h)
    if out is not None and (
        out.shape != shape or out.dtype != x.dtype or out.device != x.device
    ):
        raise ValueError("Output shape, dtype and device must match the activation")
    result = (
        out
        if out is not None and out.is_contiguous()
        else torch.empty(shape, device=x.device, dtype=x.dtype)
    )
    if result.numel():
        flat = x.reshape(-1, 2 * h)
        if flat.stride(1) != 1:
            flat = flat.contiguous()
        if max(flat.shape[0], h, flat.stride(0)) >= 2**31:
            raise ValueError("Activation geometry exceeds INT32 launch parameters")
        library().launch_shared_activation(
            flat.data_ptr(),
            result.data_ptr(),
            flat.shape[0],
            h,
            flat.stride(0),
            0.0 if limit is None else float(limit),
            torch.npu.current_stream(x.device).npu_stream,
        )
    if out is not None and result is not out:
        out.copy_(result)
        return out
    return result
