import argparse
import json
import random
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continous_engine import Engine as ContEngine, DistKVPool as ContKVPool
from continous_scheduler import Scheduler as ContScheduler, InputRequest as ContRequest

from chunked_engine import Engine as ChunkEngine, DistKVPool as ChunkKVPool
from chunked_scheduler import Scheduler as ChunkScheduler, InputRequest as ChunkRequest

SEED = 42
NUM_REQUESTS = 100
TOKEN_BUDGET = 512
CONT_BATCH_SZ = 512
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 512
MAX_PAGES_PROFILING = 8000
BASE_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z "
) * 20


def make_workload(seed: int = SEED):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    words = BASE_WORDS.split()
    cont_requests: list[ContRequest] = []
    chunk_requests: list[ChunkRequest] = []
    for _ in range(NUM_REQUESTS):
        raw_len = int(round(np_rng.lognormal(mean=6.0, sigma=0.7)))
        in_len = max(1, min(raw_len, MAX_INPUT_TOKENS))
        out_len = rng.randint(1, MAX_OUTPUT_TOKENS)
        prompt = " ".join(rng.choices(words, k=in_len))
        cont_requests.append(ContRequest(prompt, output_len=out_len))
        chunk_requests.append(ChunkRequest(prompt, output_len=out_len))
    return cont_requests, chunk_requests


def shrink_pool(
    engine: ContEngine | ChunkEngine,
    KVPoolClass: type[ContKVPool | ChunkKVPool],
    max_pages: int,
):
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


def warmup(engine: ContEngine | ChunkEngine):
    mod = sys.modules[type(engine).__module__]
    request_cls = mod.Request
    dummy = request_cls(req_id=-99, prompt_ids=torch.tensor([1], dtype=torch.long), target_len=1)
    if hasattr(dummy, "scheduling_pf_tokens"):
        dummy.scheduling_pf_tokens = dummy.prompt_token_ids
        dummy.last_chunk = True
    engine.run([dummy], num_decode_req=0)
    if -99 in engine.kv_cache_map:
        engine.kv_cache_map[-99].release()
        del engine.kv_cache_map[-99]
    torch.cuda.synchronize()


def benchmark_continuous(workload: list[ContRequest], collect_iteration_times: bool = False):
    engine = ContEngine()
    shrink_pool(engine, ContKVPool, MAX_PAGES_PROFILING)
    scheduler = ContScheduler(engine, req_batch_size=CONT_BATCH_SZ)
    for req in workload:
        scheduler.add_req(req)
    warmup(engine)

    iter_times_ms: list[float] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not scheduler.finished():
        if collect_iteration_times:
            torch.cuda.synchronize()
            it_start = time.perf_counter()
            scheduler.run()
            torch.cuda.synchronize()
            iter_times_ms.append((time.perf_counter() - it_start) * 1000.0)
        else:
            scheduler.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[Continuous]      {len(scheduler.completed)} requests  |  {elapsed:.3f} s")

    del scheduler, engine
    torch.cuda.empty_cache()
    return elapsed, iter_times_ms


def benchmark_chunked(workload: list[ChunkRequest], collect_iteration_times: bool = False):
    engine = ChunkEngine()
    shrink_pool(engine, ChunkKVPool, MAX_PAGES_PROFILING)
    scheduler = ChunkScheduler(engine, token_batch_size=TOKEN_BUDGET)
    for req in workload:
        scheduler.add_req(req)
    warmup(engine)

    iter_times_ms: list[float] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not scheduler.finished():
        if collect_iteration_times:
            torch.cuda.synchronize()
            it_start = time.perf_counter()
            scheduler.run()
            torch.cuda.synchronize()
            iter_times_ms.append((time.perf_counter() - it_start) * 1000.0)
        else:
            scheduler.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"[Chunked prefill] {len(scheduler.completed)} requests  |  {elapsed:.3f} s")

    del scheduler, engine
    torch.cuda.empty_cache()
    return elapsed, iter_times_ms


def plot_scatter(cont_times, chunk_times, out_path="iteration_times_scatter.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(range(len(cont_times)), cont_times, s=5, alpha=0.6, color="steelblue", label="Continuous batching")
    ax.scatter(range(len(chunk_times)), chunk_times, s=5, alpha=0.6, color="darkorange", label="Chunked prefill")
    ax.set_xlabel("Iteration ID")
    ax.set_ylabel("Iteration time (ms)")
    ax.set_title("Per-Iteration Time (full range)")
    ax.legend(loc="upper right", markerscale=3)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax2 = axes[1]
    ax2.scatter(range(len(cont_times)), cont_times, s=5, alpha=0.6, color="steelblue", label="Continuous batching")
    ax2.scatter(range(len(chunk_times)), chunk_times, s=5, alpha=0.6, color="darkorange", label="Chunked prefill")
    ax2.set_xlabel("Iteration ID")
    ax2.set_ylabel("Iteration time (ms)")
    ax2.set_title("Per-Iteration Time (clipped at p99)")
    ax2.legend(loc="upper right", markerscale=3)
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    p99 = max(np.percentile(cont_times, 99), np.percentile(chunk_times, 99))
    ax2.set_ylim(0, p99 * 1.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Scatter plot saved → {out_path}")


def print_iteration_summary(name: str, times_ms: list[float]):
    arr = np.array(times_ms)
    print(f"  {name}")
    print(f"    Iterations   : {len(arr)}")
    print(f"    Mean   (ms)  : {arr.mean():.2f}")
    print(f"    Median (ms)  : {np.median(arr):.2f}")
    print(f"    P99    (ms)  : {np.percentile(arr, 99):.2f}")
    print(f"    Max    (ms)  : {arr.max():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Profile continuous batching vs chunked prefill")
    parser.add_argument("--with-scatter", action="store_true", help="Collect and plot per-iteration timing scatter")
    parser.add_argument("--scatter-json", default="iteration_times.json")
    parser.add_argument("--scatter-png", default="iteration_times_scatter.png")
    args = parser.parse_args()

    cont_workload, chunk_workload = make_workload()

    in_lens = [len(r.input_str.split()) for r in cont_workload]
    out_lens = [r.output_len for r in cont_workload]
    print(
        f"  Input  words: mean={sum(in_lens)/len(in_lens):.1f}, "
        f"min={min(in_lens)}, max={max(in_lens)}"
    )
    print(
        f"  Output len  : mean={sum(out_lens)/len(out_lens):.1f}, "
        f"min={min(out_lens)}, max={max(out_lens)}"
    )
    print(f"  KV pool size: {MAX_PAGES_PROFILING} pages per engine\n")

    t_cont, cont_times = benchmark_continuous(list(cont_workload), collect_iteration_times=args.with_scatter)
    t_chunk, chunk_times = benchmark_chunked(list(chunk_workload), collect_iteration_times=args.with_scatter)

    print(f"\n{'─'*50}")
    print(f"Continuous batching time : {t_cont:.3f} s")
    print(f"Chunked prefill time     : {t_chunk:.3f} s")
    ratio = t_cont / t_chunk if t_chunk > 0 else float("inf")
    if ratio >= 1.0:
        print(f"Chunked is {ratio:.2f}x faster than continuous")
    else:
        print(f"Continuous is {1/ratio:.2f}x faster than chunked")
    print(f"{'─'*50}")

    if args.with_scatter:
        with open(args.scatter_json, "w") as f:
            json.dump({"continuous_ms": cont_times, "chunked_ms": chunk_times}, f, indent=2)
        print(f"Raw data saved → {args.scatter_json}")
        plot_scatter(cont_times, chunk_times, args.scatter_png)

        print(f"\n{'─'*55}")
        print(f"{'Metric':<30} {'Continuous':>12} {'Chunked':>10}")
        print(f"{'─'*55}")
        print_iteration_summary("Continuous", cont_times)
        print_iteration_summary("Chunked", chunk_times)
        print(f"{'─'*55}")


if __name__ == "__main__":
    main()
