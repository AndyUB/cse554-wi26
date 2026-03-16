"""
Profile naive scheduling vs continuous batching for LLM serving.

Settings:
  - batch size: 10
  - input length:  Uniform[1, 10]
  - output length: Uniform[1, 128]
  - total requests: 100
"""

import time
import random
import torch
from continous_engine import Engine, Request
from continous_scheduler import Scheduler, InputRequest

# ──────────────────────────────────────────────
# Reproducible request workload
# ──────────────────────────────────────────────
SEED = 42
NUM_REQUESTS = 100
BATCH_SIZE = 10
BASE_PROMPT = "Today is a rainy day"   # short, predictable prompt


def make_workload(engine: Engine, seed: int = SEED):
    """Return a list of InputRequest with uniform input/output lengths."""
    rng = random.Random(seed)
    requests = []
    for _ in range(NUM_REQUESTS):
        # input length in [1, 10] tokens → repeat the base word that many times
        in_len = rng.randint(1, 10)
        out_len = rng.randint(1, 128)
        # Build a prompt whose tokenised length is approximately in_len
        prompt = " ".join(BASE_PROMPT.split()[:in_len]) or BASE_PROMPT.split()[0]
        requests.append(InputRequest(prompt, output_len=out_len))
    return requests


# ──────────────────────────────────────────────────────────────────────────────
# Naive scheduler
#   - fill a batch of BATCH_SIZE requests
#   - run until ALL of them finish
#   - only then admit the next batch
# ──────────────────────────────────────────────────────────────────────────────
class NaiveScheduler:
    def __init__(self, engine: Engine, batch_size: int):
        self.engine = engine
        self.batch_size = batch_size
        self._pending: list[InputRequest] = []
        self._active: list[Request] = []   # current batch (all in decode phase after prefill)
        self._completed: list[Request] = []
        self._uid = 0

    def add_req(self, req: InputRequest):
        self._pending.append(req)

    def finished(self) -> bool:
        return not self._pending and not self._active

    def _tokenise(self, input_req: InputRequest) -> Request:
        ids = self.engine.tokenizer.encode(
            input_req.input_str, return_tensors="pt"
        ).squeeze(0)
        r = Request(req_id=self._uid, prompt_ids=ids, target_len=input_req.output_len)
        self._uid += 1
        return r

    def run(self):
        # If the active batch is empty, load the next BATCH_SIZE requests
        if not self._active:
            batch_in = self._pending[:self.batch_size]
            self._pending = self._pending[self.batch_size:]
            self._active = [self._tokenise(r) for r in batch_in]
            # Prefill step: all requests are new (num_decode_req=0)
            new_tokens = self.engine.run(self._active, num_decode_req=0)
            for i, req in enumerate(self._active):
                req.output_token_ids = torch.cat(
                    [req.output_token_ids, new_tokens[i].unsqueeze(0)]
                )
            return  # first step done (prefill)

        # Decode step: all active requests are in decode phase
        new_tokens = self.engine.run(self._active, num_decode_req=len(self._active))
        for i, req in enumerate(self._active):
            req.output_token_ids = torch.cat(
                [req.output_token_ids, new_tokens[i].unsqueeze(0)]
            )

        # Check for finished requests
        still_going = []
        for req in self._active:
            tokens_generated = req.current_length - req.prompt_length
            if tokens_generated >= req.output_length:
                self.engine.kv_cache_map[req.request_id].release()
                del self.engine.kv_cache_map[req.request_id]
                self._completed.append(req)
            else:
                still_going.append(req)
        self._active = still_going


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_naive(engine: Engine, workload: list[InputRequest]) -> float:
    scheduler = NaiveScheduler(engine, batch_size=BATCH_SIZE)
    for req in workload:
        scheduler.add_req(req)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while not scheduler.finished():
        scheduler.run()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[Naive]      Completed {len(scheduler._completed)} requests in {elapsed:.3f}s")
    return elapsed


def benchmark_continuous(engine: Engine, workload: list[InputRequest]) -> float:
    scheduler = Scheduler(engine, req_batch_size=BATCH_SIZE)
    for req in workload:
        scheduler.add_req(req)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while not scheduler.finished():
        scheduler.run()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[Continuous] Completed {len(scheduler.completed)} requests in {elapsed:.3f}s")
    return elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading model …")
    engine = Engine()
    print("Model loaded.\n")

    # Warm-up: one tiny forward pass so CUDA is fully initialised
    dummy_req = Request(req_id=-1,
                        prompt_ids=torch.tensor([1], dtype=torch.long),
                        target_len=1)
    engine.run([dummy_req], num_decode_req=0)
    if -1 in engine.kv_cache_map:
        engine.kv_cache_map[-1].release()
        del engine.kv_cache_map[-1]
    torch.cuda.synchronize()
    print("Warm-up done.\n")

    workload = make_workload(engine)

    print(f"Benchmarking with {NUM_REQUESTS} requests, batch_size={BATCH_SIZE}")
    print(f"Input length ~ Uniform[1,10], Output length ~ Uniform[1,128]\n")

    t_naive = benchmark_naive(engine, list(workload))

    # Reset KV cache state between runs
    for kv in engine.kv_cache_map.values():
        kv.release()
    engine.kv_cache_map.clear()
    engine.pool._free_pages = set(range(engine.max_pages))

    t_cont = benchmark_continuous(engine, list(workload))

    print(f"\n{'─'*45}")
    print(f"Naive scheduling time     : {t_naive:.3f} s")
    print(f"Continuous batching time  : {t_cont:.3f} s")
    print(f"Speedup (naive / cont.)   : {t_naive / t_cont:.2f}x")
    print(f"{'─'*45}")


if __name__ == "__main__":
    main()
