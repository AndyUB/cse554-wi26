import argparse
import json
import time
from pathlib import Path

import torch

from chunked_engine import DistKVPool, Engine, Request
from chunked_scheduler import Scheduler

MODEL_PATH = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
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
    token_id = tokenizer.encode("the", add_special_tokens=False)[0]
    requests: list[Request] = []
    for uid in range(NUM_REQUESTS):
        prompt_ids = torch.full((INPUT_LEN,), token_id, dtype=torch.long)
        requests.append(
            Request(req_id=uid, prompt_ids=prompt_ids, target_len=OUTPUT_LEN)
        )
    return requests


def warmup(engine: Engine):
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


def summarize_timing(engine_timing_ms: dict[str, float], wall_time_s: float):
    bucket_to_output = {
        "attention": "attention",
        "ffn": "ffn",
        "norm": "norm",
        "other": "other_ops",
    }

    gpu_total_s = engine_timing_ms["total"] / 1e3
    breakdown = {}
    for src_key, out_key in bucket_to_output.items():
        sec = engine_timing_ms[src_key] / 1e3
        breakdown[out_key] = {
            "gpu_time_s": sec,
            "pct_of_gpu": (sec / gpu_total_s * 100.0) if gpu_total_s > 0 else 0.0,
            "pct_of_wall": (sec / wall_time_s * 100.0) if wall_time_s > 0 else 0.0,
        }

    overhead_s = max(0.0, wall_time_s - gpu_total_s)
    breakdown["overheads"] = {
        "wall_time_s": overhead_s,
        "pct_of_wall": (overhead_s / wall_time_s * 100.0) if wall_time_s > 0 else 0.0,
    }

    return {
        "wall_time_s": wall_time_s,
        "gpu_total_s": gpu_total_s,
        "breakdown": breakdown,
    }


def run_chunked_breakdown(output_json: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    requests = make_workload(tokenizer)
    del tokenizer

    engine = Engine(enable_timing=True)
    # shrink_pool(engine, MAX_PAGES)
    scheduler = Scheduler(engine, token_batch_size=TOKEN_BUDGET)
    scheduler.pending_prefill = list(requests)

    warmup(engine)
    engine.reset_timing()

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

    total_input = NUM_REQUESTS * INPUT_LEN
    total_output = sum(
        req.current_length - req.prompt_length for req in scheduler.completed
    )
    total_tokens = total_input + total_output

    engine_timing_ms = engine.get_timing_totals_ms()
    summary = summarize_timing(engine_timing_ms, elapsed)
    summary["backend"] = "chunked"
    summary["timing_source"] = "cuda_events_in_engine"
    summary["engine_timing_ms"] = engine_timing_ms
    summary["num_completed"] = len(scheduler.completed)
    summary["total_input_tokens"] = total_input
    summary["total_output_tokens"] = total_output
    summary["total_tokens"] = total_tokens
    summary["throughput_tok_s"] = total_tokens / elapsed
    summary["iter_times_ms"] = iter_times_ms

    output_json.write_text(json.dumps(summary, indent=2))
    print(f"Breakdown saved: {output_json}")
    return summary


def print_summary(summary: dict):
    b = summary["breakdown"]
    print("\n=== Chunked breakdown (direct timed) ===")
    print(f"Wall time: {summary['wall_time_s']:.3f} s")
    print(f"GPU total (timed in engine): {summary['gpu_total_s']:.3f} s")
    for key in ["attention", "ffn", "norm", "other_ops"]:
        print(
            f"  {key:<14} {b[key]['gpu_time_s']:.3f} s "
            f"({b[key]['pct_of_gpu']:.1f}% GPU, {b[key]['pct_of_wall']:.1f}% wall)"
        )
    print(
        f"  {'overheads':<14} {b['overheads']['wall_time_s']:.3f} s "
        f"({b['overheads']['pct_of_wall']:.1f}% wall)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate direct GPU-time breakdown for chunked engine"
    )
    parser.add_argument("--output-json", default="profile_chunked_breakdown.json")
    args = parser.parse_args()

    summary = run_chunked_breakdown(Path(args.output_json))
    print_summary(summary)


if __name__ == "__main__":
    main()
