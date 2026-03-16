"""
Per-iteration timing: continuous batching vs chunked prefill.

Collects wall-clock time for every scheduler.run() call and plots them
as scatter plots (iteration id on x-axis, time in ms on y-axis).

Workload (same as profile_lognormal.py):
  - 100 requests
  - Input  ~ LogNormal(mu=6, sigma=0.7), clipped to [1, 512] tokens
  - Output ~ Uniform[1, 512]
  - Token budget (chunked): 512
  - Request batch (continuous): 512

Memory note
-----------
Each engine's default KV pool is replaced with a right-sized one
(MAX_PAGES_PROFILING). Only one engine is live at a time.
"""

import sys
import time
import random
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continous_engine import Engine as ContEngine, DistKVPool as ContKVPool
from continous_scheduler import Scheduler as ContScheduler, InputRequest

from chunked_engine import Engine as ChunkEngine, DistKVPool as ChunkKVPool
from chunked_scheduler import Scheduler as ChunkScheduler

# ── Parameters ─────────────────────────────────────────────────────────────────
SEED              = 42
NUM_REQUESTS      = 100
TOKEN_BUDGET      = 512
CONT_BATCH_SZ     = 512
MAX_INPUT_TOKENS  = 512
MAX_OUTPUT_TOKENS = 512
MAX_PAGES_PROFILING = 8000   # ≈ 4 GB; default engines use 10–15 GB

BASE_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z "
) * 20


def make_workload(tokenizer, seed: int = SEED):
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    words  = BASE_WORDS.split()
    requests = []
    for _ in range(NUM_REQUESTS):
        raw_len = int(round(np_rng.lognormal(mean=6.0, sigma=0.7)))
        in_len  = max(1, min(raw_len, MAX_INPUT_TOKENS))
        out_len = rng.randint(1, MAX_OUTPUT_TOKENS)
        prompt  = " ".join(rng.choices(words, k=in_len))
        requests.append(InputRequest(prompt, output_len=out_len))
    return requests


def _shrink_pool(engine, KVPoolClass, max_pages: int):
    old = engine.pool
    del old.k_datas, old.v_datas
    engine.pool = KVPoolClass(
        num_layers=engine.layers,
        num_kv_heads=engine.num_kv_heads,
        head_dim=engine.head_dim,
        capacity=max_pages,
        page_size=engine.page_size,
    )
    engine.max_pages = max_pages
    torch.cuda.empty_cache()


def _warmup(engine):
    mod   = sys.modules[type(engine).__module__]
    R     = mod.Request
    dummy = R(req_id=-99, prompt_ids=torch.tensor([1], dtype=torch.long), target_len=1)
    if hasattr(dummy, "scheduling_pf_tokens"):
        dummy.scheduling_pf_tokens = dummy.prompt_token_ids
        dummy.last_chunk = True
    engine.run([dummy], num_decode_req=0)
    if -99 in engine.kv_cache_map:
        engine.kv_cache_map[-99].release()
        del engine.kv_cache_map[-99]
    torch.cuda.synchronize()


# ── Collection helpers ─────────────────────────────────────────────────────────
def collect_iteration_times_continuous(workload):
    print("Running continuous batching …")
    engine    = ContEngine()
    _shrink_pool(engine, ContKVPool, MAX_PAGES_PROFILING)
    scheduler = ContScheduler(engine, req_batch_size=CONT_BATCH_SZ)
    for req in workload:
        scheduler.add_req(req)

    _warmup(engine)

    times_ms = []
    while not scheduler.finished():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scheduler.run()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    print(f"  Continuous: {len(times_ms)} iterations, "
          f"mean={np.mean(times_ms):.1f} ms, max={max(times_ms):.1f} ms")

    del scheduler, engine
    torch.cuda.empty_cache()
    return times_ms


def collect_iteration_times_chunked(workload):
    print("Running chunked prefill …")
    engine    = ChunkEngine()
    _shrink_pool(engine, ChunkKVPool, MAX_PAGES_PROFILING)
    scheduler = ChunkScheduler(engine, token_batch_size=TOKEN_BUDGET)
    for req in workload:
        scheduler.add_req(req)

    _warmup(engine)

    times_ms = []
    while not scheduler.finished():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scheduler.run()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    print(f"  Chunked:    {len(times_ms)} iterations, "
          f"mean={np.mean(times_ms):.1f} ms, max={max(times_ms):.1f} ms")

    del scheduler, engine
    torch.cuda.empty_cache()
    return times_ms


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_scatter(cont_times, chunk_times, out_path="iteration_times_scatter.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: both series on the same axes (unclipped – shows full range)
    ax = axes[0]
    ax.scatter(range(len(cont_times)),  cont_times,
               s=5, alpha=0.6, color="steelblue",  label="Continuous batching")
    ax.scatter(range(len(chunk_times)), chunk_times,
               s=5, alpha=0.6, color="darkorange", label="Chunked prefill")
    ax.set_xlabel("Iteration ID")
    ax.set_ylabel("Iteration time (ms)")
    ax.set_title("Per-Iteration Time (full range)")
    ax.legend(loc="upper right", markerscale=3)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Right: zoomed to p99 to suppress outliers
    ax2 = axes[1]
    ax2.scatter(range(len(cont_times)),  cont_times,
                s=5, alpha=0.6, color="steelblue",  label="Continuous batching")
    ax2.scatter(range(len(chunk_times)), chunk_times,
                s=5, alpha=0.6, color="darkorange", label="Chunked prefill")
    ax2.set_xlabel("Iteration ID")
    ax2.set_ylabel("Iteration time (ms)")
    ax2.set_title("Per-Iteration Time (clipped at p99)")
    ax2.legend(loc="upper right", markerscale=3)
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    p99 = max(
        np.percentile(cont_times,  99),
        np.percentile(chunk_times, 99),
    )
    ax2.set_ylim(0, p99 * 1.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Scatter plot saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/local1/cse554/models/meta-llama/Llama-3.2-1B"
    )
    workload = make_workload(tokenizer)
    del tokenizer

    cont_times  = collect_iteration_times_continuous(list(workload))
    chunk_times = collect_iteration_times_chunked(list(workload))

    # Persist raw data
    with open("iteration_times.json", "w") as f:
        json.dump({"continuous_ms": cont_times, "chunked_ms": chunk_times}, f, indent=2)
    print("Raw data saved → iteration_times.json")

    plot_scatter(cont_times, chunk_times)

    # Summary table
    print(f"\n{'─'*55}")
    print(f"{'Metric':<30} {'Continuous':>12} {'Chunked':>10}")
    print(f"{'─'*55}")
    for label, arr in [("Continuous", cont_times), ("Chunked", chunk_times)]:
        a = np.array(arr)
        print(f"  {label}")
        print(f"    Iterations   : {len(arr)}")
        print(f"    Mean   (ms)  : {a.mean():.2f}")
        print(f"    Median (ms)  : {np.median(a):.2f}")
        print(f"    P99    (ms)  : {np.percentile(a, 99):.2f}")
        print(f"    Max    (ms)  : {a.max():.2f}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
