"""Plot CUBLAS and CUTLASS GEMM TFLOPs for Assignment 3 Section 1 Q2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("gemm_perf.csv"),
        help="Merged CSV with columns M,N,K,library,tflops.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where PNG files will be written.",
    )
    return parser.parse_args()


def plot_shape(df: pd.DataFrame, n: int, k: int, output_dir: Path) -> None:
    """Plot one (N, K) shape with M on x-axis and TFLOPs on y-axis."""
    shape_df = df[(df["N"] == n) & (df["K"] == k)].sort_values("M")

    plt.figure(figsize=(8, 5))
    for library, style in (("cublas", "-o"), ("cutlass", "-s")):
        library_df = shape_df[shape_df["library"] == library]
        if library_df.empty:
            continue
        plt.plot(
            library_df["M"],
            library_df["tflops"],
            style,
            linewidth=2,
            markersize=5,
            label=library.upper(),
        )

    plt.title(f"GEMM Performance (N={n}, K={k})")
    plt.xlabel("M")
    plt.ylabel("TFLOPs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"gemm_n{n}_k{k}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Entry point for plotting per-shape GEMM performance curves."""
    args = parse_args()
    df = pd.read_csv(args.input)

    required = {"M", "N", "K", "library", "tflops"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shapes = df[["N", "K"]].drop_duplicates().sort_values(["N", "K"])

    for _, row in shapes.iterrows():
        plot_shape(df, int(row["N"]), int(row["K"]), args.output_dir)


if __name__ == "__main__":
    main()
