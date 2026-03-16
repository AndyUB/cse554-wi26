import argparse
import csv
import json
import os
import subprocess
import sys
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
        req = Request(req_id=uid, prompt_ids=prompt_ids, target_len=OUTPUT_LEN)
        requests.append(req)
    return requests


def warmup(engine: Engine):
    dummy = Request(req_id=-99, prompt_ids=torch.tensor([1], dtype=torch.long), target_len=1)
    dummy.scheduling_pf_tokens = dummy.prompt_token_ids
    dummy.last_chunk = True
    engine.run([dummy], num_decode_req=0)
    if -99 in engine.kv_cache_map:
        engine.kv_cache_map[-99].release()
        del engine.kv_cache_map[-99]
    torch.cuda.synchronize()


def run_chunked_internal(output_json: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    requests = make_workload(tokenizer)
    del tokenizer

    engine = Engine()
    shrink_pool(engine, MAX_PAGES)
    scheduler = Scheduler(engine, token_batch_size=TOKEN_BUDGET)
    scheduler.pending_prefill = list(requests)
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

    total_input = NUM_REQUESTS * INPUT_LEN
    total_output = sum(req.current_length - req.prompt_length for req in scheduler.completed)
    total_tokens = total_input + total_output

    result = {
        "backend": "chunked",
        "elapsed_s": elapsed,
        "throughput_tok_s": total_tokens / elapsed,
        "num_completed": len(scheduler.completed),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "iter_times_ms": iter_times_ms,
    }
    output_json.write_text(json.dumps(result, indent=2))
    print(f"Chunked raw metrics saved: {output_json}")


def run_subprocess(cmd: list[str], env: dict[str, str] | None = None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_nsys_profile(command: list[str], report_prefix: Path):
    run_subprocess(
        [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cuda-memory-usage=true",
            "-o",
            str(report_prefix),
            *command,
        ]
    )
    return report_prefix.with_suffix(".nsys-rep")


def export_gpukernsum_csv(report_file: Path, output_prefix: Path):
    run_subprocess(
        [
            "nsys",
            "stats",
            "--report",
            "gpukernsum",
            "--format",
            "csv",
            "--output",
            str(output_prefix),
            str(report_file),
        ]
    )
    csv_path = output_prefix.parent / f"{output_prefix.name}_gpukernsum.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected kernel summary CSV not found: {csv_path}")
    return csv_path


def _get_ns(row: dict[str, str]):
    for key in row:
        low = key.lower()
        if "total" in low and "ns" in low:
            return float(row[key])
        if "sum" in low and "ns" in low:
            return float(row[key])
    return 0.0


def _get_name(row: dict[str, str]):
    for key in row:
        if "name" in key.lower():
            return row[key]
    return "unknown"


def classify_kernel(name: str):
    n = name.lower()
    if any(k in n for k in ["attn", "attention", "flash", "softmax", "qk", "kv", "paged"]):
        return "attention"
    if any(k in n for k in ["gemm", "matmul", "cublas", "cutlass", "silu", "swiglu", "mlp"]):
        return "ffn"
    if any(k in n for k in ["norm", "rms", "layernorm"]):
        return "norm"
    return "other_kernels"


def summarize_breakdown(csv_path: Path, wall_time_s: float):
    totals_ns = {"attention": 0.0, "ffn": 0.0, "norm": 0.0, "other_kernels": 0.0}
    with csv_path.open() as f:
        rows = csv.DictReader(f)
        for row in rows:
            name = _get_name(row)
            t_ns = _get_ns(row)
            bucket = classify_kernel(name)
            totals_ns[bucket] += t_ns

    gpu_total_s = sum(totals_ns.values()) / 1e9
    breakdown = {}
    for key, t_ns in totals_ns.items():
        sec = t_ns / 1e9
        breakdown[key] = {
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


def load_wall_time(metrics_json: Path):
    data = json.loads(metrics_json.read_text())
    if "elapsed_s" in data:
        return float(data["elapsed_s"])
    if "elapsed_time" in data:
        return float(data["elapsed_time"])
    raise ValueError(f"Cannot find elapsed time in {metrics_json}")


def run_backend_profile(backend: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "chunked":
        raw_json = out_dir / "profile_chunked_results.json"
        target_cmd = [sys.executable, __file__, "--mode", "internal-chunked", "--output-json", str(raw_json)]
    elif backend == "vllm":
        raw_json = out_dir / "profile_vllm_results.json"
        target_cmd = [
            sys.executable,
            "-m",
            "vllm.benchmarks.throughput",
            "--backend",
            "vllm",
            "--model",
            MODEL_PATH,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(INPUT_LEN),
            "--random-output-len",
            str(OUTPUT_LEN),
            "--num-prompts",
            str(NUM_REQUESTS),
            "--output-json",
            str(raw_json),
            "--disable-detokenize",
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "float16",
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    report_prefix = out_dir / f"nsys_{backend}"
    report_file = run_nsys_profile(target_cmd, report_prefix)
    csv_path = export_gpukernsum_csv(report_file, out_dir / f"nsys_{backend}")

    wall_time_s = load_wall_time(raw_json)
    summary = summarize_breakdown(csv_path, wall_time_s)
    summary["backend"] = backend
    summary["raw_metrics_json"] = str(raw_json)
    summary["kernel_csv"] = str(csv_path)
    summary["nsys_report"] = str(report_file)

    out_json = out_dir / f"profile_{backend}_breakdown.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Breakdown saved: {out_json}")
    return out_json, summary


def print_summary(name: str, summary: dict):
    b = summary["breakdown"]
    print(f"\n=== {name} breakdown ===")
    print(f"Wall time: {summary['wall_time_s']:.3f} s")
    print(f"GPU total: {summary['gpu_total_s']:.3f} s")
    for key in ["attention", "ffn", "norm", "other_kernels"]:
        print(
            f"  {key:<14} {b[key]['gpu_time_s']:.3f} s "
            f"({b[key]['pct_of_gpu']:.1f}% GPU, {b[key]['pct_of_wall']:.1f}% wall)"
        )
    print(
        f"  {'overheads':<14} {b['overheads']['wall_time_s']:.3f} s "
        f"({b['overheads']['pct_of_wall']:.1f}% wall)"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate kernel-time breakdown for chunked prefill and/or vLLM")
    parser.add_argument("--backend", choices=["chunked", "vllm", "both"], default="both")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--mode", choices=["driver", "internal-chunked"], default="driver")
    parser.add_argument("--output-json", default="profile_chunked_results.json")
    args = parser.parse_args()

    if args.mode == "internal-chunked":
        run_chunked_internal(Path(args.output_json))
        return

    out_dir = Path(args.out_dir)
    if args.backend == "both":
        _, chunked_summary = run_backend_profile("chunked", out_dir)
        _, vllm_summary = run_backend_profile("vllm", out_dir)
        print_summary("Chunked", chunked_summary)
        print_summary("vLLM", vllm_summary)
    else:
        _, summary = run_backend_profile(args.backend, out_dir)
        print_summary(args.backend, summary)


if __name__ == "__main__":
    main()
