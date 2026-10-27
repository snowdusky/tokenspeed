# DeepSeek V4.1 Flash on Ascend

This recipe uses eight Ascend devices, FlatQuant W4A4 experts, BF16 dense
layers, packed KV caches, host-resident FP8 Engram tables and DSpark. Select
TP1/DP8, TP2/DP4 or TP8/DP1 explicitly; all three use EP8.

The TP8 recipe explicitly selects FP32 accumulation for BF16 reductions and
AIV transport for reductions and gathers up to 2 MiB. The same policy supports
aligned, contiguous TP2 and TP4 groups within one node. Larger messages keep
the same accumulation dtype and use host-driven HCCL. The four
`TOKENSPEED_HCCL_TP_{REDUCE_DTYPE,REDUCE_ALGORITHM,GATHER_ALGORITHM,AIV_MAX_BYTES}`
settings are required together. They apply to the configured rank group,
including Dense and LM-head collectives when those layers share its ranks.
The V4.1 model prepares this policy through the existing
`prepare_communication_runtime` hook, before cache profiling and graph capture.
Other models do not install it. Without these settings, ordinary HCCL behavior
is unchanged.

The W4A4 path also requires the existing `flash_ops` FlatQuant extension,
which registers `torch.ops.custom.flat_quant`.

Activate the serving environment and source the CANN environment. Build the
Ascend C libraries and install the kernel package from this checkout:

```bash
python tokenspeed-kernel-npu/csrc/dsv41/build.py
python -m pip install --no-deps --no-build-isolation -e tokenspeed-kernel-npu
bash examples/deepseek_v41_ascend/serve.sh /path/to/model 8
python examples/deepseek_v41_ascend/benchmark.py --model /path/to/model \
  --endpoint 127.0.0.1:9350 --input-len 131072 --output-len 1000 \
  --dp-size 1 --concurrency 1 8 16 32 --rounds 3 --reuse-prefix \
  --out /path/to/result.json
```

Set `--dp-size` to `8 / TP` so requests cover all attention DP ranks. The
benchmark records streamed token arrivals and reports decode throughput only
inside the interval when every request is active. End-to-end throughput and
TTFT are reported separately. Reusing a prefix reduces prefill work; this
fixed counting workload is not an agentic accuracy benchmark.

The kernel package registers providers through the TokenSpeed plugin entry
point. No external experiment directory is needed on `PYTHONPATH`. Native
libraries are built locally; checkpoints, weight conversion, measurements and
compiled libraries do not belong in the source tree.

Host Engram mappings are prepared before graph capture and must remain alive
until all graphs using them are destroyed. Attention builds 16-head and
64-head capacity variants from one source; the model's local head count
selects the layout, while request counts remain runtime inputs.

DSpark context writes and proposal workspaces use registered Ascend kernels.
They preserve the reference normalization and RoPE rounding, followed by the
all-channel FP8 group32 round trip; the window remains BF16. The codec uses
integer operations instead of requiring native FP8 tensor conversion on 910.

## Expert load balancing

V4.1 FlatQuant W4A4 supports the standard expert placement and online EPLB
controller. MC2 selects a local-row load mask so idle and padded rows do not
inflate expert counts; other transports keep their existing recorder layout.
The model binds MC2's dedicated communicator before expert weight processing.
For cross-node MC2, use EP32, EP64 or EP128 with MoE TP1. Start
with one redundant expert slot per EP rank: `--ep-num-redundant-experts`
counts extra slots across the entire EP group, not per device. For EP64:

```text
--moe-backend ascend_v41_w4a4 --all2all-backend ascend_mc2
--ep-num-redundant-experts 64 --ep-dispatch-algorithm static
--expert-distribution-recorder-mode stat --enable-eplb
--eplb-algorithm deepseek --eplb-rebalance-num-iterations 1000
--eplb-rebalance-layers-per-chunk 1
```

These flags supplement a multi-node serving configuration; the eight-device
script above remains an EP8 recipe. Reserve memory for both the extra expert
slots and migration staging before sizing KV capacity. The existing execution
gate performs each layer's migration before publishing its new routing map.
Packed NZ weights and both scale representations move together without
changing captured tensor addresses. The target owns placement and counters;
DSpark's draft keeps its own original expert layout.

Compare EPLB on/off at equal EP size, input/output length and offered load.
Include counter and migration overhead in sustained throughput, and record
DSpark acceptance rather than assuming it stays constant as EP grows.
