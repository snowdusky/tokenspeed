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

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import statistics
import threading
import time
from pathlib import Path

import grpc
from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc as rpc
from smg_grpc_proto.generated import common_pb2
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2 as pb
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument(
    "--model", required=True, help="Local checkpoint including tokenizer and encoding/"
)
p.add_argument("--endpoint", default="127.0.0.1:18345")
p.add_argument("--input-len", type=int, required=True)
p.add_argument("--output-len", type=int, required=True)
p.add_argument("--concurrency", nargs="+", type=int, required=True)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--out", required=True)
p.add_argument("--dp-size", required=True, type=int)
p.add_argument("--skip-warmup", action="store_true")
p.add_argument("--reuse-prefix", action="store_true")
a = p.parse_args()
if a.dp_size < 1 or a.rounds < 1 or a.input_len < 32 or a.output_len < 2:
    p.error("positive DP size/rounds, input >= 32, and output >= 2 are required")
if any(c < 1 for c in a.concurrency):
    p.error("concurrency must be positive")
root = Path(a.model)
tok = AutoTokenizer.from_pretrained(
    str(root), local_files_only=True, trust_remote_code=True
)
spec = importlib.util.spec_from_file_location(
    "checkpoint_encoding", root / "encoding" / "encoding.py"
)
encoding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoding)
channel = grpc.insecure_channel(a.endpoint)
grpc.channel_ready_future(channel).result(timeout=120)
stub = rpc.TokenSpeedSchedulerStub(channel)
assert stub.HealthCheck(pb.HealthCheckRequest(), timeout=60).healthy
output = Path(a.out)
output.parent.mkdir(parents=True, exist_ok=True)
log = output.with_suffix(".jsonl").open("w")


def emit(record):
    line = json.dumps(record, ensure_ascii=False)
    log.write(line + "\n")
    log.flush()
    print(line, flush=True)


def request(i, c, round_id, nout, barrier):
    prompt = (
        "" if a.reuse_prefix else f"Case {i}: "
    ) + "Reference notes: the archive is stored in the north building. " * (
        a.input_len // 8 + 10
    )
    prompt += "\nCase 0: Count upward from one, separated by commas."
    ids = tok.encode(
        encoding.encode_messages(
            [{"role": "user", "content": prompt}], thinking_mode="chat"
        ),
        add_special_tokens=False,
    )
    assert len(ids) >= a.input_len and a.input_len >= 32
    ids = ids[: a.input_len - 24] + ids[-24:]
    req = pb.GenerateRequest(
        request_id=f"perf-c{c}-r{round_id}-i{i}-{time.monotonic_ns()}",
        data_parallel_rank=i % a.dp_size,
        tokenized=pb.TokenizedInput(input_ids=ids),
        sampling_params=pb.SamplingParams(
            temperature=0,
            top_p=1,
            max_new_tokens=nout,
            ignore_eos=True,
            n=1,
            skip_special_tokens=False,
            repetition_penalty=1,
        ),
        stream=True,
    )
    barrier.wait()
    start = time.monotonic()
    chunks = []
    complete = None
    output_ids = []
    for response in stub.Generate(req, timeout=7200):
        now = time.monotonic()
        if response.HasField("chunk") and response.chunk.token_ids:
            chunks.append((now, len(response.chunk.token_ids)))
            output_ids.extend(response.chunk.token_ids)
        if response.HasField("complete"):
            complete = response.complete
    end = time.monotonic()
    assert complete is not None and chunks
    count = sum(n for _, n in chunks)
    assert count == nout and complete.completion_tokens == nout, (count, nout, complete)
    if not a.reuse_prefix:
        assert complete.cached_tokens == 0, complete
    first, last = chunks[0][0], chunks[-1][0]
    denomin = count - chunks[0][1]
    return {
        "case": i,
        "concurrency": c,
        "round": round_id,
        "input_tokens": len(ids),
        "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "assigned_dp_rank": i % a.dp_size,
        "output_tokens": count,
        "cached_tokens": complete.cached_tokens,
        "output_ids": output_ids,
        "chunk_sizes": sorted(set(n for _, n in chunks)),
        "start": start,
        "end": end,
        "ttft_ms": 1000 * (first - start),
        "e2e_s": end - start,
        "tpot_ms": 1000 * (last - first) / denomin,
        "token_gap_ms": [
            1000 * (chunks[j][0] - chunks[j - 1][0]) / chunks[j][1]
            for j in range(1, len(chunks))
        ],
        "chunk_elapsed_s": [(t - start, n) for t, n in chunks],
    }


all_results = []
for c in a.concurrency:
    for r in range(0 if a.skip_warmup else -1, a.rounds):
        if not a.reuse_prefix or r == -1:
            flushed = stub.FlushCache(
                common_pb2.FlushCacheRequest(timeout_s=30), timeout=40
            )
            assert flushed.success, flushed
        barrier = threading.Barrier(c)
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            fs = [
                pool.submit(request, i, c, r, 16 if r == -1 else a.output_len, barrier)
                for i in range(c)
            ]
            results = [f.result() for f in fs]
        span = max(x["end"] for x in results) - min(x["start"] for x in results)
        summary = {
            "kind": "warmup" if r == -1 else "measurement",
            "concurrency": c,
            "round": r,
            "mean_tpot_ms": statistics.mean(x["tpot_ms"] for x in results),
            "mean_ttft_ms": statistics.mean(x["ttft_ms"] for x in results),
            "output_throughput_tps": sum(x["output_tokens"] for x in results) / span,
            "total_throughput_tps": sum(
                x["input_tokens"] + x["output_tokens"] for x in results
            )
            / span,
            "span_s": span,
        }
        lo = max(x["start"] + x["chunk_elapsed_s"][0][0] for x in results) + 0.1
        hi = min(x["start"] + x["chunk_elapsed_s"][-1][0] for x in results) - 0.1
        steady = {"valid": False, "reason": "no common all-active decode window"}
        if hi > lo:
            per = []
            arrivals = 0
            for row in results:
                points = []
                cumulative = 0
                for elapsed, tokens in row["chunk_elapsed_s"]:
                    cumulative += tokens
                    points.append((row["start"] + elapsed, cumulative))
                inside = [(t, n) for t, n in points if lo <= t <= hi]
                if len(inside) < 2:
                    break
                per.append(
                    1000
                    * (inside[-1][0] - inside[0][0])
                    / (inside[-1][1] - inside[0][1])
                )
                arrivals += max((n for t, n in points if t <= hi), default=0) - max(
                    (n for t, n in points if t <= lo), default=0
                )
            if len(per) == c:
                steady = {
                    "valid": True,
                    "concurrency": c,
                    "duration_s": hi - lo,
                    "tpot_p50_ms": statistics.median(per),
                    "per_request_tpot_ms": per,
                    "output_throughput_tps": arrivals / (hi - lo),
                }
        summary["steady_decode"] = steady
        summary["cached_input_tokens"] = sum(x["cached_tokens"] for x in results)
        emit(summary)
        for result in results:
            log.write(json.dumps({"kind": "request", **result}) + "\n")
        log.flush()
        if r >= 0:
            all_results.append({"summary": summary, "requests": results})
output.write_text(
    json.dumps({"args": vars(a), "results": all_results}, indent=2) + "\n"
)
print("PERF_COMPLETE", flush=True)
