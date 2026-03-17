"""Profile vLLM Llama-3.2-1B with per-component CUDA-event timing.

Architecture:
  This script is both the launcher and the worker.
  - When run normally it re-invokes itself as a subprocess with
    VLLM_LLAMA_TIMING_WORKER=1, which runs the actual benchmark and writes
    raw timing to a JSON file, then exits cleanly (triggering atexit).
  - The parent reads the JSON and prints the summary.

Requires enforce_eager=True (no CUDA graphs).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_PATH   = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
INPUT_LEN    = 512
OUTPUT_LEN   = 512
NUM_REQUESTS = 100

SCRIPT_DIR   = Path(__file__).parent
RAW_JSON     = SCRIPT_DIR / "profile_vllm_breakdown_raw.json"
OUTPUT_JSON  = SCRIPT_DIR / "profile_vllm_breakdown.json"
START_FLAG   = SCRIPT_DIR / "timing_start.flag"
VLLM_ENV     = "/local1/groups/554g13/cse554-winter-2026/vllm_env"


# ---------------------------------------------------------------------------
# Worker — runs inside a dedicated subprocess so atexit fires on clean exit
# ---------------------------------------------------------------------------
def _worker() -> None:
    """Full benchmark run; called when VLLM_LLAMA_TIMING_WORKER=1."""
    import torch
    from vllm import LLM, SamplingParams

    print("Loading model (enforce_eager=True, no CUDA graphs)...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    params = SamplingParams(
        max_tokens=OUTPUT_LEN,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    prompts = ["the " * INPUT_LEN] * NUM_REQUESTS

    # Warmup — timing is NOT active yet (flag file doesn't exist)
    print("Warming up (2 requests)...")
    llm.generate(prompts[:2], params)
    torch.cuda.synchronize()

    # Create the flag file: EngineCore picks it up at the next model forward()
    # entry, resets accumulated warmup data, and begins clean timing.
    start_flag = Path(os.environ["VLLM_LLAMA_TIMING_START_FLAG"])
    start_flag.write_text("start")

    print(f"Running {NUM_REQUESTS} requests "
          f"(input={INPUT_LEN}, output={OUTPUT_LEN})...")
    t_start = time.perf_counter()
    llm.generate(prompts, params)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_start
    print(f"Wall time: {wall_s:.3f} s")
    start_flag.unlink(missing_ok=True)

    # Write wall time alongside timing so the parent can read both
    wall_file = RAW_JSON.with_suffix(".wall")
    wall_file.write_text(str(wall_s))
    # atexit in llama.py will write RAW_JSON when this process exits


# ---------------------------------------------------------------------------
# Parent — launches worker, waits, reads results
# ---------------------------------------------------------------------------
def _parent() -> None:
    RAW_JSON.unlink(missing_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"]                    = f"{VLLM_ENV}:{env.get('PYTHONPATH', '')}"
    env["VLLM_LLAMA_TIMING"]             = "1"
    env["VLLM_LLAMA_TIMING_OUT"]         = str(RAW_JSON)
    env["VLLM_LLAMA_TIMING_START_FLAG"]  = str(START_FLAG)
    env["VLLM_LLAMA_TIMING_WORKER"]      = "1"
    START_FLAG.unlink(missing_ok=True)   # ensure no stale flag from prior run

    print("Launching benchmark worker subprocess...")
    ret = subprocess.run(
        [sys.executable, __file__],
        env=env,
    )
    if ret.returncode != 0:
        print(f"Worker exited with code {ret.returncode}")
        return

    if not RAW_JSON.exists():
        print(f"ERROR: timing file not written: {RAW_JSON}")
        return

    totals_ms = json.loads(RAW_JSON.read_text())
    wall_file = RAW_JSON.with_suffix(".wall")
    wall_s = float(wall_file.read_text()) if wall_file.exists() else 0.0

    summary = _summarize(totals_ms, wall_s)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Breakdown saved: {OUTPUT_JSON}")
    _print_summary(summary)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize(totals_ms: dict, wall_s: float) -> dict:
    gpu_total_s = totals_ms["total"] / 1e3
    breakdown: dict = {}
    for key in ["attention", "ffn", "norm"]:
        s = totals_ms[key] / 1e3
        breakdown[key] = {
            "gpu_time_s":  s,
            "pct_of_gpu":  s / gpu_total_s * 100 if gpu_total_s > 0 else 0.0,
            "pct_of_wall": s / wall_s       * 100 if wall_s       > 0 else 0.0,
        }
    overhead_s = max(0.0, wall_s - gpu_total_s)
    breakdown["overheads"] = {
        "wall_time_s": overhead_s,
        "pct_of_wall": overhead_s / wall_s * 100 if wall_s > 0 else 0.0,
    }
    return {
        "backend": "vllm_eager",
        "timing_source": "cuda_events_in_llama_decoder_layer",
        "wall_time_s": wall_s,
        "gpu_total_s": gpu_total_s,
        "breakdown": breakdown,
        "engine_timing_ms": totals_ms,
    }


def _print_summary(summary: dict) -> None:
    b = summary["breakdown"]
    print("\n=== vLLM breakdown (CUDA events, enforce_eager) ===")
    print(f"Wall time:         {summary['wall_time_s']:.3f} s")
    print(f"GPU total (timed): {summary['gpu_total_s']:.3f} s")
    for key in ["attention", "ffn", "norm"]:
        print(
            f"  {key:<14} {b[key]['gpu_time_s']:.3f} s "
            f"({b[key]['pct_of_gpu']:.1f}% GPU, {b[key]['pct_of_wall']:.1f}% wall)"
        )
    print(
        f"  {'overheads':<14} {b['overheads']['wall_time_s']:.3f} s "
        f"({b['overheads']['pct_of_wall']:.1f}% wall)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if os.environ.get("VLLM_LLAMA_TIMING_WORKER"):
        _worker()
    else:
        _parent()
