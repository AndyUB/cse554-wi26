"""
End-to-end profiling: chunked prefill vs continuous batching.

Workload:
  - 100 requests
  - Input length  ~ LogNormal(mu=6, sigma=0.7), clipped to [1, 512] tokens
  - Output length ~ Uniform[1, 512]
  - Token batch size (chunked): 512
  - Request batch size (continuous): 512

Memory note
-----------
The default engine KV pools (20k–30k pages ≈ 10–15 GB) are replaced with a
right-sized pool (MAX_PAGES_PROFILING pages) immediately after construction.
Only one engine is live at a time; torch.cuda.empty_cache() is called between
runs to avoid OOM.
"""

import sys
import time
import random
import numpy as np
import torch

from continous_engine import Engine as ContEngine, DistKVPool as ContKVPool
from continous_scheduler import Scheduler as ContScheduler, InputRequest

from chunked_engine import Engine as ChunkEngine, DistKVPool as ChunkKVPool
from chunked_scheduler import Scheduler as ChunkScheduler

# ── Workload parameters ────────────────────────────────────────────────────────
SEED              = 42
NUM_REQUESTS      = 100
TOKEN_BUDGET      = 512   # max tokens/iter for chunked prefill
CONT_BATCH_SZ     = 512   # max concurrent requests for continuous batching
MAX_INPUT_TOKENS  = 512   # lognormal clip upper bound
MAX_OUTPUT_TOKENS = 512

# Right-sized KV pool: 100 reqs × (512+512) tokens / 16 page_size = 6400 pages
# Add 20 % headroom → 8000 pages ≈ 4 GB  (vs default 10–15 GB)
MAX_PAGES_PROFILING = 8000

BASE_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z "
) * 20


def make_workload(tokenizer, seed: int = SEED):
    """Return InputRequest list with lognormal input / uniform output lengths."""
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
    """Replace engine's default (large) KV pool with a smaller one."""
    old = engine.pool
    del old.k_datas, old.v_datas   # free GPU tensors immediately
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
    # chunked engine requires scheduling_pf_tokens to be set by the scheduler;
    # set it manually here since we are bypassing the scheduler for warmup.
    if hasattr(dummy, "scheduling_pf_tokens"):
        dummy.scheduling_pf_tokens = dummy.prompt_token_ids
        dummy.last_chunk = True
    engine.run([dummy], num_decode_req=0)
    if -99 in engine.kv_cache_map:
        engine.kv_cache_map[-99].release()
        del engine.kv_cache_map[-99]
    torch.cuda.synchronize()


# ── Benchmark helpers ──────────────────────────────────────────────────────────
def benchmark_continuous(workload):
    print("Loading continuous batching engine …")
    engine = ContEngine()
    _shrink_pool(engine, ContKVPool, MAX_PAGES_PROFILING)

    scheduler = ContScheduler(engine, req_batch_size=CONT_BATCH_SZ)
    for req in workload:
        scheduler.add_req(req)

    _warmup(engine)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not scheduler.finished():
        scheduler.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"[Continuous]      {len(scheduler.completed)} requests  |  {elapsed:.3f} s")

    # Explicit cleanup before next engine is created
    del scheduler, engine
    torch.cuda.empty_cache()
    return elapsed


def benchmark_chunked(workload):
    print("Loading chunked prefill engine …")
    engine = ChunkEngine()
    _shrink_pool(engine, ChunkKVPool, MAX_PAGES_PROFILING)

    scheduler = ChunkScheduler(engine, token_batch_size=TOKEN_BUDGET)
    for req in workload:
        scheduler.add_req(req)

    _warmup(engine)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not scheduler.finished():
        scheduler.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"[Chunked prefill] {len(scheduler.completed)} requests  |  {elapsed:.3f} s")

    del scheduler, engine
    torch.cuda.empty_cache()
    return elapsed


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Building workload …")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/local1/cse554/models/meta-llama/Llama-3.2-1B"
    )
    workload = make_workload(tokenizer)
    del tokenizer   # free before loading model weights

    in_lens  = [len(r.input_str.split()) for r in workload]
    out_lens = [r.output_len for r in workload]
    print(f"  Input  words: mean={sum(in_lens)/len(in_lens):.1f}, "
          f"min={min(in_lens)}, max={max(in_lens)}")
    print(f"  Output len  : mean={sum(out_lens)/len(out_lens):.1f}, "
          f"min={min(out_lens)}, max={max(out_lens)}")
    print(f"  KV pool size: {MAX_PAGES_PROFILING} pages per engine\n")

    t_cont  = benchmark_continuous(list(workload))
    t_chunk = benchmark_chunked(list(workload))

    print(f"\n{'─'*50}")
    print(f"Continuous batching time : {t_cont:.3f} s")
    print(f"Chunked prefill time     : {t_chunk:.3f} s")
    ratio = t_cont / t_chunk if t_chunk > 0 else float("inf")
    if ratio >= 1.0:
        print(f"Chunked is {ratio:.2f}x faster than continuous")
    else:
        print(f"Continuous is {1/ratio:.2f}x faster than chunked")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
