import time
import json
import numpy as np
import torch

from chunked_engine import Engine, DistKVPool, Request
from chunked_scheduler import Scheduler

INPUT_LEN = 512
OUTPUT_LEN = 512
NUM_REQUESTS = 100
TOKEN_BUDGET = 512
MAX_PAGES = 8000


def shrink_pool(engine: Engine, max_pages: int):
    old = engine.pool
    del old.k_datas, old.v_datas
    engine.pool = DistKVPool(
        num_layers=engine.layers,
        num_kv_heads=engine.num_kv_heads,
        head_dim=engine.head_dim,
        capacity=max_pages,
        page_size=engine.page_size,
    )
    engine.max_pages = max_pages
    torch.cuda.empty_cache()


def make_workload(tokenizer):
    tok_id = tokenizer.encode("the")[0]
    requests: list[FakeInputRequest] = []
    for _ in range(NUM_REQUESTS):
        prompt_ids = torch.full((INPUT_LEN,), tok_id, dtype=torch.long)
        req = FakeInputRequest(prompt_ids, OUTPUT_LEN)
        requests.append(req)
    return requests


class FakeInputRequest:
    def __init__(self, prompt_ids: torch.Tensor, output_len: int):
        self.prompt_ids = prompt_ids
        self.output_len = output_len


def warmup(engine):
    dummy = Request(
        req_id=-99, prompt_ids=torch.tensor([1], dtype=torch.long), target_len=1
    )
    dummy.scheduling_pf_tokens = dummy.prompt_token_ids
    dummy.last_chunk = True
    engine.run([dummy], num_decode_req=0)
    if -99 in engine.kv_cache_map:
        engine.kv_cache_map[-99].release()
        del engine.kv_cache_map[-99]
    torch.cuda.synchronize()


def run_chunked(workload):
    engine = Engine()
    shrink_pool(engine, MAX_PAGES)
    scheduler = Scheduler(engine, token_batch_size=TOKEN_BUDGET)

    uid = 0
    raw_requests = []
    for item in workload:
        req = Request(
            req_id=uid, prompt_ids=item.prompt_ids, target_len=item.output_len
        )
        raw_requests.append(req)
        uid += 1
    scheduler.pending_prefill = raw_requests
    warmup(engine)

    iter_times_ms = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    while not scheduler.finished():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scheduler.run()
        torch.cuda.synchronize()
        iter_times_ms.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    total_input_tokens = NUM_REQUESTS * INPUT_LEN
    total_output_tokens = sum(
        req.current_length - req.prompt_length for req in scheduler.completed
    )
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / elapsed

    print(f"  Requests completed : {len(scheduler.completed)}")
    print(f"  Total input tokens : {total_input_tokens}")
    print(f"  Total output tokens: {total_output_tokens}")
    print(f"  Elapsed time       : {elapsed:.3f} s")
    print(f"  Throughput         : {throughput:.1f} tokens/s")

    del scheduler, engine
    torch.cuda.empty_cache()

    return {
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "total_tokens": total_tokens,
        "num_completed": NUM_REQUESTS,
        "iter_times_ms": iter_times_ms,
    }


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/local1/cse554/models/meta-llama/Llama-3.2-1B"
    )
    workload = make_workload(tokenizer)
    del tokenizer

    print(f"Chunked-prefill benchmark")
    print(
        f"  input_len={INPUT_LEN}, output_len={OUTPUT_LEN}, "
        f"num_requests={NUM_REQUESTS}, token_budget={TOKEN_BUDGET}\n"
    )

    results = run_chunked(workload)

    with open("profile_chunked_fixed_itertimes.json", "w") as f:
        json.dump({"iter_times_ms": results["iter_times_ms"]}, f)

    print("\n=== Chunked-prefill benchmark results ===")
    print(f"  Elapsed time  : {results['elapsed_s']:.3f} s")
    print(f"  Throughput    : {results['throughput_tok_s']:.1f} tokens/s")
    iters = np.array(results["iter_times_ms"])
    print(f"  Iterations    : {len(iters)}")
    print(f"  Mean iter     : {iters.mean():.2f} ms")
    print(f"  Median iter   : {np.median(iters):.2f} ms")
    print(f"  Max iter      : {iters.max():.2f} ms")


if __name__ == "__main__":
    main()
