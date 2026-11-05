#!/usr/bin/env bash
set -euo pipefail
# Usage: serve.sh MODEL_PATH TP_SIZE
model_path=${1:?Pass the prepared FlatQuant checkpoint path}
tp_size=${2:?Pass 1, 2, or 8 for the eight-device EP deployment}
case "$tp_size" in
    1|2) capture_sizes=(1 2 4 8 16); aiv=0 ;;
    8) capture_sizes=(1 2 4 8 16 32); aiv=1 ;;
    *) echo "Expected TP size 1, 2, or 8" >&2; exit 2 ;;
esac
repo_root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
export PYTHONPATH="$repo_root/python:$repo_root/tokenspeed-kernel/python:$repo_root/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
if (( aiv )); then
    export TOKENSPEED_HCCL_TP_REDUCE_DTYPE=float32
    export TOKENSPEED_HCCL_TP_REDUCE_ALGORITHM=aiv
    export TOKENSPEED_HCCL_TP_GATHER_ALGORITHM=aiv
    export TOKENSPEED_HCCL_TP_AIV_MAX_BYTES=2097152
fi
export HCCL_OP_EXPANSION_MODE=HOST HCCL_BUFFSIZE=200 HCCL_DETERMINISTIC=true
export OMP_NUM_THREADS=1
exec python -m smg_grpc_servicer.tokenspeed \
    --model "$model_path" --served-model-name v41-flatquant --host 0.0.0.0 --port 9350 \
    --dist-init-addr 127.0.0.1:22172 --device npu \
    --world-size 8 --nnodes 1 --node-rank 0 --nprocs-per-node 8 \
    --tensor-parallel-size "$tp_size" --data-parallel-size "$((8 / tp_size))" --expert-parallel-size 8 \
    --moe-tp-size 1 --dense-tp-size "$tp_size" --lm-head-tp-size "$tp_size" --engram-tp-size "$tp_size" \
    --engram-host-table --engram-host-table-layout shared --engram-host-table-dir /dev/shm \
    --deepseek-v41-indexer-query-parallel-size "$tp_size" \
    --attention-backend deepseek_v41 --moe-backend ascend_v41_w4a4 \
    --trust-remote-code --dtype bfloat16 --sampling-backend greedy \
    --max-model-len 133120 --max-total-tokens 5242880 --max-num-seqs 32 \
    --chunked-prefill-size 384 --max-cudagraph-capture-size "${capture_sizes[-1]}" \
    --cudagraph-capture-sizes "${capture_sizes[@]}" \
    --disable-weight-loader-prefetch-checkpoints --disable-kvstore --language-model-only \
    --disable-prefill-graph --disable-pdl --gpu-memory-utilization 0.90 --speculative-algorithm DSPARK
