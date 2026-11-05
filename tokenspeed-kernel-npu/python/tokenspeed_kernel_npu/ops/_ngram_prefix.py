import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _entry():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libngram_prefix.so")))
    fn = lib.launch_ngram_prefix
    fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    fn.restype = None
    return lib, fn


def resolve(
    snapshots,
    positions,
    reset,
    slots,
    cache_lengths,
    tail,
    needs_seed,
    prefix,
    bs,
    vocab,
):
    for tensor in (snapshots, positions, slots, tail, prefix):
        if tensor.dtype != torch.int64 or not tensor.is_contiguous():
            raise ValueError(
                "Engram prefix requires contiguous int64 history and slots"
            )
    if cache_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Engram cache lengths must be int32 or int64")
    if reset.dtype not in (torch.bool, torch.int64) or needs_seed.dtype != torch.bool:
        raise ValueError("Engram reset must be bool/int64 and seed flags must be bool")
    _, fn = _entry()
    fn(
        *(
            t.data_ptr()
            for t in (
                snapshots,
                positions,
                reset,
                slots,
                cache_lengths,
                tail,
                needs_seed,
                prefix,
            )
        ),
        bs,
        vocab,
        cache_lengths.element_size(),
        reset.element_size(),
        torch.npu.current_stream().npu_stream,
    )
