import sys
import time
import random
import numpy as np
import torch

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
    R = mod.Request
    dummy = R(req_id=-99, prompt_ids=torch.tensor([1], dtype=torch.long), target_len=1)
    if hasattr(dummy, "scheduling_pf_tokens"):
        dummy.scheduling_pf_tokens = dummy.prompt_token_ids
        dummy.last_chunk = True
    engine.run([dummy], num_decode_req=0)
    if -99 in engine.kv_cache_map:
        engine.kv_cache_map[-99].release()
        del engine.kv_cache_map[-99]
    torch.cuda.synchronize()


def benchmark_continuous(workload: list[ContRequest]) -> float:
    engine = ContEngine()
    shrink_pool(engine, ContKVPool, MAX_PAGES_PROFILING)
    scheduler = ContScheduler(engine, req_batch_size=CONT_BATCH_SZ)
    for req in workload:
        scheduler.add_req(req)
    warmup(engine)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not scheduler.finished():
        scheduler.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[Continuous]      {len(scheduler.completed)} requests  |  {elapsed:.3f} s")

    del scheduler, engine
    torch.cuda.empty_cache()
    return elapsed


def benchmark_chunked(workload: list[ChunkRequest]) -> float:
    engine = ChunkEngine()
    shrink_pool(engine, ChunkKVPool, MAX_PAGES_PROFILING)
    scheduler = ChunkScheduler(engine, token_batch_size=TOKEN_BUDGET)
    for req in workload:
        scheduler.add_req(req)
    warmup(engine)

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


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/local1/cse554/models/meta-llama/Llama-3.2-1B"
    )
    cont_workload, chunk_workload = make_workload(tokenizer)
    del tokenizer

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

    t_cont = benchmark_continuous(list(cont_workload))
    t_chunk = benchmark_chunked(list(chunk_workload))

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
