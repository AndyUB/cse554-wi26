import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS = ["llama3_1b", "llama3_3b", "llama3_8b"]
MODEL_LABELS = {
    "llama3_1b": "LLaMA3-1B",
    "llama3_3b": "LLaMA3-3B",
    "llama3_8b": "LLaMA3-8B",
}


def _subplot_compare(
    df: pd.DataFrame,
    x_col: str,
    y_a: str,
    y_b: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(MODELS):
        part = df[df["model"] == model].sort_values(x_col)
        axs[i].plot(
            part[x_col], part[y_a], marker="o", linestyle="-", label="PyTorch SDPA"
        )
        axs[i].plot(
            part[x_col], part[y_b], marker="x", linestyle="--", label="FlashInfer"
        )
        axs[i].set_title(MODEL_LABELS[model])
        axs[i].set_xlabel(x_label)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()
    axs[0].set_ylabel(y_label)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def _subplot_flashinfer_only(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(MODELS):
        part = df[df["model"] == model].sort_values(x_col)
        axs[i].plot(
            part[x_col], part[y_col], marker="o", linestyle="-", label="FlashInfer"
        )
        axs[i].set_title(MODEL_LABELS[model])
        axs[i].set_xlabel(x_label)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()
    axs[0].set_ylabel(y_label)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def main(results_dir: Path, out_dir: Path) -> None:
    prefill_p = pd.read_csv(results_dir / "prefill_vs_p.csv")
    prefill_batch = pd.read_csv(results_dir / "prefill_vs_batch.csv")
    decode_c = pd.read_csv(results_dir / "decode_vs_c.csv")
    decode_batch = pd.read_csv(results_dir / "decode_vs_batch.csv")
    decode_page = pd.read_csv(results_dir / "decode_vs_page_size.csv")

    _subplot_compare(
        prefill_p,
        x_col="log2_p",
        y_a="sdpa_tflops",
        y_b="flashinfer_tflops",
        x_label="log2(p)",
        y_label="Compute Utilization (TFLOPs)",
        title="Prefill Attention Compute Utilization vs Prompt Length (Batch=1)",
        out_path=out_dir / "prefill_vs_p.png",
    )

    _subplot_compare(
        prefill_batch,
        x_col="log2_batch",
        y_a="sdpa_tflops",
        y_b="flashinfer_tflops",
        x_label="log2(batch size)",
        y_label="Compute Utilization (TFLOPs)",
        title="Prefill Attention Compute Utilization vs Batch Size (p=1024)",
        out_path=out_dir / "prefill_vs_batch.png",
    )

    _subplot_compare(
        decode_c,
        x_col="log2_c",
        y_a="sdpa_gbps",
        y_b="flashinfer_gbps",
        x_label="log2(c)",
        y_label="Memory Bandwidth Utilization (GB/s)",
        title="Decode Attention Memory Bandwidth vs Context Length (Batch=1)",
        out_path=out_dir / "decode_vs_c.png",
    )

    _subplot_compare(
        decode_batch,
        x_col="log2_batch",
        y_a="sdpa_gbps",
        y_b="flashinfer_gbps",
        x_label="log2(batch size)",
        y_label="Memory Bandwidth Utilization (GB/s)",
        title="Decode Attention Memory Bandwidth vs Batch Size (c=1024)",
        out_path=out_dir / "decode_vs_batch.png",
    )

    _subplot_flashinfer_only(
        decode_page,
        x_col="page_size",
        y_col="flashinfer_gbps",
        x_label="Page Size",
        y_label="Memory Bandwidth Utilization (GB/s)",
        title="Decode Attention (FlashInfer) Bandwidth vs Page Size (Batch=128, c=1024)",
        out_path=out_dir / "decode_vs_page_size.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Section2 attention benchmark figures from CSV outputs."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("assignment3/Section2/results")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("assignment3/Section2/figures")
    )
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
    print(f"Saved figures to {args.output_dir}")
