"""Parse nsys cuda_gpu_kern_sum CSV and print an attention/FFN/norm/other breakdown."""

import csv
import sys
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Kernel categorisation rules (checked in order; first match wins)
# ---------------------------------------------------------------------------
# Patterns are matched against the lower-cased, demangled kernel name.

ATTENTION_PATTERNS = [
    "flashinfer",
    "flash_infer",
    "batch_prefill",
    "batch_decode",
    "paged_kv",
    "fmha",
    "flash_attn",
    "apply_rope",
    "rotary",
    "softmax_lse",
]

NORM_PATTERNS = [
    "rms_norm",
    "rmsnorm",
    "layer_norm",
    "layernorm",
    "fused_add_rms",
    "vectorized_rms",
    # torch.compile/inductor fused RMSNorm: triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_*
    "pow_rsqrt",
]

# cuBLAS / CUTLASS GEMM kernels that implement the linear projections in FFN
# and attention QKV + output projections.  Also include Triton/inductor fused
# FFN activation kernels (SiLU+mul produced by torch.compile).
GEMM_PATTERNS = [
    "gemm",
    "cutlass",
    "tensorop",
    "_sgemm",
    "_hgemm",
    "_dgemm",
    "sm80_xmma",
    "sm86_xmma",
    "sm89_xmma",
    "sm90_xmma",
    "volta_",
    "ampere_",
    "turing_",
    "xmma_gemm",
    "splitk_gemm",
    # torch.compile fused SiLU+mul activation in FFN
    "mul_silu",
    "silu_and_mul",
]


def categorize(name: str) -> str:
    n = name.lower()
    for p in ATTENTION_PATTERNS:
        if p in n:
            return "attention"
    for p in NORM_PATTERNS:
        if p in n:
            return "norm"
    for p in GEMM_PATTERNS:
        if p in n:
            return "gemm"
    return "other"


def main(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    bucket_ns: dict[str, float] = defaultdict(float)
    top_kernels: dict[str, list] = defaultdict(list)  # bucket -> [(ns, name)]

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column name may vary slightly; try both common spellings
            name = row.get("Name") or row.get("name") or ""
            total_ns_str = (
                row.get("Total Time (ns)")
                or row.get("Total Time")
                or row.get("total_time_ns")
                or "0"
            )
            try:
                total_ns = float(total_ns_str.replace(",", ""))
            except ValueError:
                continue

            bucket = categorize(name)
            bucket_ns[bucket] += total_ns
            top_kernels[bucket].append((total_ns, name))

    grand_total_ns = sum(bucket_ns.values())
    if grand_total_ns == 0:
        print("No kernel data found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Map GEMM bucket to "ffn" label for display (GEMMs are the FFN / proj ops)
    display_labels = {
        "attention": "attention",
        "gemm": "ffn (gemm)",
        "norm": "norm",
        "other": "other",
    }

    print("\n=== vLLM GPU kernel breakdown (nsys cuda_gpu_kern_sum) ===")
    print(f"{'Category':<18} {'GPU Time (s)':>14} {'% of GPU':>10}")
    print("-" * 46)
    for bucket in ["attention", "gemm", "norm", "other"]:
        ns = bucket_ns.get(bucket, 0.0)
        label = display_labels[bucket]
        print(f"  {label:<16} {ns/1e9:>14.3f} {ns/grand_total_ns*100:>9.1f}%")
    print("-" * 46)
    print(f"  {'total':<16} {grand_total_ns/1e9:>14.3f} {'100.0%':>10}")

    print("\nTop-3 kernels per category:")
    for bucket in ["attention", "gemm", "norm", "other"]:
        label = display_labels[bucket]
        entries = sorted(top_kernels.get(bucket, []), reverse=True)[:3]
        print(f"\n  [{label}]")
        for ns, name in entries:
            short = name if len(name) <= 90 else name[:87] + "..."
            print(f"    {ns/1e9:.4f} s  {short}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cuda_gpu_kern_sum.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
