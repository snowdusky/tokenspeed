# Stop conditions

`apply_stop_conditions` validates tensors and dispatches the in-place update
through `metadata / apply_stop_conditions`. NVIDIA and AMD retain the portable Triton leaf. Ascend selects a device leaf
that clamps masked addresses for its DMA loads, including empty prefill outputs.
CPU retains the scalar reference. Device leaves accept the public function's nine tensors
followed by `num_extends`, `num_prefill_outputs`, `decode_width` and `padding_index`,
all passed positionally. Device leaves must preserve EOS, token budgets,
speculative width truncation, stopped requests, unfinished prefill and padding
semantics, without host tensor reads during graph capture or replay.

Engram preparation and commit keep the same frontier ownership on all devices.
Ascend's commit handles zero-input rows in its device leaf without changing
the CUDA/ROCm Triton implementation.
