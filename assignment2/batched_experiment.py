import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from different_prefill import Engine as DifferentPrefillEngine
from uniform_prefill import Engine as UniformPrefillEngine


def build_prompt(tokenizer, target_len: int, seed_text: str):
    ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id or 0]

    while len(ids) < target_len:
        ids.extend(ids)
    ids = ids[:target_len]

    return tokenizer.decode(ids, skip_special_tokens=True)


def benchmark_uniform(engine, prompts, output_len: int):
    if hasattr(engine, "reset_cache"):
        engine.reset_cache()

    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.generate_batched(prompts, rounds=output_len)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def benchmark_different(engine, prompts, output_len: int):
    if hasattr(engine, "reset_cache"):
        engine.reset_cache()

    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.generate_batched(prompts, rounds=output_len)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-batch-pow", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("assignment2/results/part2")
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    uniform_engine = UniformPrefillEngine()
    different_engine = DifferentPrefillEngine()

    base_prompt = build_prompt(
        uniform_engine.tokenizer,
        target_len=args.input_len,
        seed_text="The University of Washington is located in Seattle. ",
    )

    rows = []
    batch_sizes = [2**i for i in range(args.max_batch_pow + 1)]
    for batch_size in batch_sizes:
        uniform_prompts = [base_prompt for _ in range(batch_size)]
        different_prompts = [
            build_prompt(
                different_engine.tokenizer,
                target_len=args.input_len,
                seed_text=f"Prompt {i}: artificial profiling text. ",
            )
            for i in range(batch_size)
        ]

        uniform_times = [
            benchmark_uniform(
                uniform_engine, uniform_prompts, output_len=args.output_len
            )
            for _ in range(args.repeats)
        ]
        different_times = [
            benchmark_different(
                different_engine, different_prompts, output_len=args.output_len
            )
            for _ in range(args.repeats)
        ]

        uniform_time = sum(uniform_times) / len(uniform_times)
        different_time = sum(different_times) / len(different_times)

        total_tokens = batch_size * args.output_len
        uniform_tps = total_tokens / uniform_time
        different_tps = total_tokens / different_time

        rows.append(
            {
                "batch_size": batch_size,
                "uniform_time_s": uniform_time,
                "uniform_tps": uniform_tps,
                "different_time_s": different_time,
                "different_tps": different_tps,
            }
        )

        print(
            f"batch={batch_size:2d} | "
            f"uniform: {uniform_time:.4f}s, {uniform_tps:.1f} tok/s | "
            f"different: {different_time:.4f}s, {different_tps:.1f} tok/s"
        )

    csv_path = args.out_dir / "part2_batch_profile.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "uniform_time_s",
                "uniform_tps",
                "different_time_s",
                "different_tps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["batch_size"] for r in rows],
        [r["uniform_time_s"] for r in rows],
        marker="o",
        label="Uniform prefill",
    )
    plt.plot(
        [r["batch_size"] for r in rows],
        [r["different_time_s"] for r in rows],
        marker="o",
        label="Different prefill",
    )
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Generation time (s)")
    plt.title("Part 2: Generation time vs batch size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    time_plot_path = args.out_dir / "part2_generation_time_vs_batch.png"
    plt.tight_layout()
    plt.savefig(time_plot_path, dpi=150)

    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["batch_size"] for r in rows],
        [r["uniform_tps"] for r in rows],
        marker="o",
        label="Uniform prefill",
    )
    plt.plot(
        [r["batch_size"] for r in rows],
        [r["different_tps"] for r in rows],
        marker="o",
        label="Different prefill",
    )
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Part 2: Throughput vs batch size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    tps_plot_path = args.out_dir / "part2_throughput_vs_batch.png"
    plt.tight_layout()
    plt.savefig(tps_plot_path, dpi=150)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {time_plot_path}")
    print(f"Saved plot: {tps_plot_path}")


if __name__ == "__main__":
    main()
