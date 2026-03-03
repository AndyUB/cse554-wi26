from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from flashinfer_pipeline import Engine, Request


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_SIZE = 32_000


def _make_requests(batch: int, prompt_len: int, output_len: int) -> List[Request]:
    return [
        Request(i, torch.randint(100, VOCAB_SIZE, (prompt_len,)), output_len)
        for i in range(batch)
    ]


def _cuda_sync() -> None:
    torch.cuda.synchronize()


def _time_event_ms() -> Tuple[torch.cuda.Event, torch.cuda.Event]:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def time_prefill_ms(
    engine: Engine,
    batch: int,
    prompt_len: int,
    warmup: int = 2,
    iters: int = 5,
) -> float:
    """Average wall-time (ms) for a single prefill pass."""
    times: List[float] = []
    for i in range(warmup + iters):
        reqs = _make_requests(batch, prompt_len, 1)
        _cuda_sync()
        s, e = _time_event_ms()
        s.record()
        engine.run(reqs, num_decode_req=0)
        e.record()
        _cuda_sync()
        engine.reset()
        if i >= warmup:
            times.append(s.elapsed_time(e))
    return sum(times) / len(times)


def time_decode_total_ms(
    engine: Engine,
    batch: int,
    prompt_len: int,
    decode_len: int,
) -> float:
    """Total time (ms) for decode_len decode steps (after one prefill)."""
    reqs = _make_requests(batch, prompt_len, decode_len)

    outputs = engine.run(reqs, num_decode_req=0)
    for i, req in enumerate(reqs):
        req.output_token_ids = torch.cat([req.output_token_ids, outputs[i : i + 1]])

    _cuda_sync()
    s, e = _time_event_ms()
    s.record()
    for _ in range(decode_len):
        dec_out = engine.run(reqs, num_decode_req=len(reqs))
        for i, req in enumerate(reqs):
            req.output_token_ids = torch.cat([req.output_token_ids, dec_out[i : i + 1]])
    e.record()
    _cuda_sync()

    engine.reset()
    return s.elapsed_time(e)


def profile_last_decode_ms(
    engine: Engine,
    batch: int,
    prompt_len: int,
    decode_len: int,
) -> Dict[str, float]:
    """Run decode_len steps then profile the last decode step per operation."""
    reqs = _make_requests(batch, prompt_len, decode_len)

    outputs = engine.run(reqs, num_decode_req=0)
    for i, req in enumerate(reqs):
        req.output_token_ids = torch.cat([req.output_token_ids, outputs[i : i + 1]])

    for _ in range(decode_len - 1):
        dec_out = engine.run(reqs, num_decode_req=len(reqs))
        for i, req in enumerate(reqs):
            req.output_token_ids = torch.cat([req.output_token_ids, dec_out[i : i + 1]])

    _cuda_sync()
    _, op_times = engine.run(reqs, num_decode_req=len(reqs), profile_ops=True)

    engine.reset()
    return op_times


def profile_prefill_ms(
    engine: Engine,
    batch: int,
    prompt_len: int,
    warmup: int = 1,
) -> Dict[str, float]:
    """Profile per-op times for a single prefill pass."""
    all_times: Optional[Dict[str, float]] = None

    for i in range(warmup + 1):
        reqs = _make_requests(batch, prompt_len, 1)
        _cuda_sync()
        _, op_times = engine.run(reqs, num_decode_req=0, profile_ops=True)
        engine.reset()
        if i == warmup:
            all_times = op_times

    return all_times


def experiment_q2_1(engine: Engine) -> None:
    print("\n=== Q2.1: decode-length sweep (batch=32, prefill=256) ===")
    batch = 32
    prompt_len = 256
    decode_lens = [2**i for i in range(5, 11)]  # 32 .. 1024

    time_rows: List[Dict] = []
    op_rows: List[Dict] = []

    for dl in decode_lens:
        print(f"  decode_len={dl} ...", end=" ", flush=True)
        t_pre = time_prefill_ms(engine, batch, prompt_len, warmup=2, iters=3)
        t_dec = time_decode_total_ms(engine, batch, prompt_len, dl)
        print(f"prefill={t_pre:.1f}ms  decode_total={t_dec:.1f}ms")
        time_rows.append(
            {
                "decode_len": dl,
                "log2_decode_len": int(math.log2(dl)),
                "prefill_ms": t_pre,
                "decode_total_ms": t_dec,
            }
        )
        op_t = profile_last_decode_ms(engine, batch, prompt_len, dl)
        row: Dict = {"decode_len": dl, "log2_decode_len": int(math.log2(dl))}
        row.update(op_t)
        op_rows.append(row)
        print(f"    last-step ops: {op_t}")

    _write_csv(RESULTS_DIR / "q2_1_time_vs_decode_len.csv", time_rows)
    _write_csv(RESULTS_DIR / "q2_1_last_decode_ops.csv", op_rows)

    x = [r["log2_decode_len"] for r in time_rows]
    fig, ax = plt.subplots()
    ax.plot(x, [r["prefill_ms"] for r in time_rows], "o-", label="Prefill")
    ax.plot(x, [r["decode_total_ms"] for r in time_rows], "s-", label="Decode total")
    ax.set_xlabel("log₂(decode length)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Q2.1 End-to-end time (batch=32, prefill=256)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_1_time_vs_decode_len.png", dpi=150)
    plt.close(fig)

    op_keys = [
        k for k in op_rows[0].keys() if k not in ("decode_len", "log2_decode_len")
    ]
    x_labels = [str(r["decode_len"]) for r in op_rows]
    fig, ax = plt.subplots()
    bottoms = [0.0] * len(op_rows)
    for op in op_keys:
        vals = [r.get(op, 0.0) for r in op_rows]
        ax.bar(x_labels, vals, bottom=bottoms, label=op)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xlabel("Decode length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Q2.1 Last decode step op breakdown (batch=32)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_1_last_decode_ops.png", dpi=150)
    plt.close(fig)

    print("  Saved Q2.1 plots and CSVs.")


def experiment_q2_2(engine: Engine) -> None:
    print("\n=== Q2.2: prefill-length sweep (batch=1) ===")
    batch = 1
    prefill_lens = [2**i for i in range(8, 15)]  # 256 .. 16384

    rows: List[Dict] = []
    for pl in prefill_lens:
        print(f"  prefill_len={pl} ...", end=" ", flush=True)
        op_t = profile_prefill_ms(engine, batch, pl, warmup=1)
        print(op_t)
        row: Dict = {"prefill_len": pl, "log2_prefill_len": int(math.log2(pl))}
        row.update(op_t)
        rows.append(row)

    _write_csv(RESULTS_DIR / "q2_2_prefill_breakdown.csv", rows)

    x = [r["log2_prefill_len"] for r in rows]
    op_keys = [
        k for k in rows[0].keys() if k not in ("prefill_len", "log2_prefill_len")
    ]
    fig, ax = plt.subplots()
    bottoms = [0.0] * len(rows)
    for op in op_keys:
        vals = [r.get(op, 0.0) for r in rows]
        ax.bar(x, vals, bottom=bottoms, label=op, width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xlabel("log₂(prefill length)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Q2.2 Prefill operation breakdown (batch=1)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_2_prefill_breakdown.png", dpi=150)
    plt.close(fig)

    totals = [sum(r.get(op, 0.0) for op in op_keys) for r in rows]
    fig, ax = plt.subplots()
    ax.plot(x, totals, "o-")
    ax.set_xlabel("log₂(prefill length)")
    ax.set_ylabel("Total prefill time (ms)")
    ax.set_title("Q2.2 Prefill time vs length (batch=1)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_2_prefill_total.png", dpi=150)
    plt.close(fig)

    print("  Saved Q2.2 plots and CSVs.")


def experiment_q2_3(engine: Engine) -> None:
    print("\n=== Q2.3: batch-size sweep (prefill=128, decode=128) ===")
    prompt_len = 128
    decode_len = 128
    batch_sizes = [2**i for i in range(0, 9)]  # 1 .. 256

    rows: List[Dict] = []
    for batch in batch_sizes:
        print(f"  batch={batch} ...", end=" ", flush=True)

        t_pre = time_prefill_ms(engine, batch, prompt_len, warmup=2, iters=3)
        t_dec = time_decode_total_ms(engine, batch, prompt_len, decode_len)
        total_ms = t_pre + t_dec
        total_toks = batch * (prompt_len + decode_len)
        throughput = total_toks / (total_ms * 1e-3)  # tokens/s

        print(
            f"prefill={t_pre:.1f}ms  decode={t_dec:.1f}ms  throughput={throughput:.0f} tok/s"
        )
        rows.append(
            {
                "batch": batch,
                "log2_batch": int(math.log2(batch)),
                "prefill_ms": t_pre,
                "decode_total_ms": t_dec,
                "total_ms": total_ms,
                "throughput_toks_per_s": throughput,
            }
        )

    _write_csv(RESULTS_DIR / "q2_3_batch_sweep.csv", rows)

    x = [r["log2_batch"] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(x, [r["prefill_ms"] for r in rows], "o-", label="Prefill")
    ax.plot(x, [r["decode_total_ms"] for r in rows], "s-", label="Decode total")
    ax.plot(x, [r["total_ms"] for r in rows], "^-", label="Total")
    ax.set_xlabel("log₂(batch size)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Q2.3 End-to-end time vs batch size (prefill=128, decode=128)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_3_time_vs_batch.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, [r["throughput_toks_per_s"] for r in rows], "o-")
    ax.set_xlabel("log₂(batch size)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Q2.3 Throughput vs batch size (prefill=128, decode=128)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_3_throughput_vs_batch.png", dpi=150)
    plt.close(fig)

    print("  Saved Q2.3 plots and CSVs.")


if __name__ == "__main__":
    print("Loading engine ...")
    engine = Engine()
    print("Engine ready.\n")

    experiment_q2_1(engine)
    experiment_q2_2(engine)
    experiment_q2_3(engine)

    print(f"\nAll results written to: {RESULTS_DIR}")
