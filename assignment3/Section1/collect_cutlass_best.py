"""Collect best CUTLASS profiler results for Assignment 3 Section 1 Q2.

This script parses CUTLASS profiler CSV outputs and keeps the best TFLOPs per
(M, N, K) across all profiled kernels and split-K settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "m",
    "n",
    "k",
    "runtime",
    "split_k_slices",
    "Operation",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more CUTLASS profiler CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cutlass_gemm_perf.csv"),
        help="Output CSV path with best per-shape CUTLASS performance.",
    )
    return parser.parse_args()


def read_cutlass_csv(path: Path) -> pd.DataFrame:
    """Read one profiler CSV and return normalized columns needed for analysis."""
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.rename(columns={"m": "M", "n": "N", "k": "K", "runtime": "time_ms"})
    df["tflops"] = (2.0 * df["M"] * df["N"] * df["K"]) / (df["time_ms"] * 1e-3) / 1e12
    return df[["M", "N", "K", "time_ms", "split_k_slices", "Operation", "tflops"]]


def combine_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    """Combine multiple CUTLASS CSV files into one DataFrame."""
    frames = [read_cutlass_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def select_best_kernels(df: pd.DataFrame) -> pd.DataFrame:
    """Select the best TFLOPs row for each GEMM problem shape."""
    best_idx = df.groupby(["M", "N", "K"])["tflops"].idxmax()
    best = df.loc[best_idx].copy()
    best.insert(3, "library", "cutlass")
    return best.sort_values(["N", "K", "M"]).reset_index(drop=True)


def main() -> None:
    """Entry point for collecting best CUTLASS GEMM results."""
    args = parse_args()
    combined = combine_inputs(args.inputs)
    best = select_best_kernels(combined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(args.output, index=False)
    print(f"Wrote {len(best)} best-shape rows to {args.output}")


if __name__ == "__main__":
    main()
