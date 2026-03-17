"""
Parse vllm_nsys_events.tsv (nsys per-kernel event trace) and compute
Attention / FFN / other / cuda_graph breakdown.

Column layout (tab-separated, header on line 1):
  Name  Start  Duration  GPU  Context

Kernel classification by name
----------------------------------------------------------------------
ATTENTION
  - flashinfer::* (BatchPrefill, BatchDecode, MergeStates, ...)
  - vllm::reshape_and_cache_flash_kernel  (KV cache write, part of attn)
  - triton_poi_fused_1                    (rotary embedding Q/K)
  - triton_poi_fused_3                    (rotary embedding Q/K, next variant)

FFN  (linear projections + activation)
  - turing_fp16_*gemm* / turing_fp16_*xmma* (cuBLAS GEMM)
  - cutlass::Kernel*                         (CUTLASS GEMM)
  - triton_poi_fused_mul_silu_slice_*        (FFN SiLU activation)

NOTE: GEMMs include both attention projections (QKV, O) and FFN
projections (gate_up, down).  All have the same kernel name so they
cannot be separated purely by name.  We assign all GEMMs to the FFN
bucket because FFN projections dominate GEMM time ~5:1 by parameter
count (gate_up=16384x2048, down=2048x8192 vs QKV=3072x2048, O=2048x2048
for Llama-3.2-1B).

OTHER  (norms, misc, memcpy, ...)
  - triton_red_fused_*  (fused RMSNorm kernels from inductor)
  - Memcpy / memset
  - everything else not matched above

CUDA_GRAPH  (separate category — cannot attribute to attention or FFN)
  - "Graph XXXX (GraphExec YYYY)"
  These are CUDAGraphMode.FULL executions: the entire model forward
  (all 16 decoder layers, including both attention AND FFN) is captured
  in one graph and replayed for pure-decode steps.  The 7 distinct graph
  IDs correspond to 7 different padded batch sizes.  Since each graph
  replay contains both attention and FFN, it is not possible to split
  them into those categories from the kernel name alone.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def parse_duration(s: str) -> float:
    """'9.309 ms' → 0.009309, '186.815 μs' → 1.86815e-4, '53.8173s' → time in s."""
    s = s.strip().replace(",", "")
    m = re.match(r"([0-9.]+)\s*([a-zμ]*)", s)
    if not m:
        return 0.0
    v, u = float(m.group(1)), m.group(2).lower()
    if u in ("s", ""):
        return v
    if u == "ms":
        return v * 1e-3
    if u in ("μs", "us", "\u03bcs"):
        return v * 1e-6
    if u == "ns":
        return v * 1e-9
    return v


# ---------------------------------------------------------------------------
# Kernel categorisation
# ---------------------------------------------------------------------------

ATTENTION_PATTERNS = [
    r"flashinfer",
    r"vllm::reshape_and_cache",
    r"^triton_poi_fused_1$",
    r"^triton_poi_fused_3$",
]

FFN_PATTERNS = [
    # cuBLAS Turing tensor-op GEMMs
    r"turing_fp16.*gemm",
    r"turing_fp16.*xmma",
    # CUTLASS
    r"cutlass",
    # Triton FFN activation (SiLU+mul+slice fused kernel from torch.compile)
    r"triton_poi_fused_mul_silu",
]

CUDA_GRAPH_PATTERN = re.compile(r"^Graph\s+\d+\s+\(GraphExec\s+\d+\)", re.IGNORECASE)


def categorize(name: str) -> str:
    n = name.strip()

    if CUDA_GRAPH_PATTERN.match(n):
        return "cuda_graph"

    nl = n.lower()
    for pat in ATTENTION_PATTERNS:
        if re.search(pat, nl):
            return "attention"
    for pat in FFN_PATTERNS:
        if re.search(pat, nl):
            return "ffn"

    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tsv_path: str) -> None:
    path = Path(tsv_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    bucket_s: dict[str, float] = defaultdict(float)
    bucket_count: dict[str, int] = defaultdict(int)
    graph_ids: set[str] = set()
    unmatched: dict[str, float] = defaultdict(float)

    with open(path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name     = parts[0]
            dur_str  = parts[2]
            dur_s    = parse_duration(dur_str)

            bucket = categorize(name)
            bucket_s[bucket]     += dur_s
            bucket_count[bucket] += 1

            if bucket == "cuda_graph":
                m = re.match(r"^(Graph\s+\d+\s+\(GraphExec\s+\d+\))", name)
                if m:
                    graph_ids.add(m.group(1))
            elif bucket == "other":
                unmatched[name] += dur_s

    grand      = sum(bucket_s.values())
    excl_graph = grand - bucket_s.get("cuda_graph", 0.0)

    print("\n=== vLLM GPU kernel breakdown (nsys events, non-eager run) ===")
    print(f"{'Category':<14} {'Time (s)':>10} {'% of total':>12} {'% ex-graph':>12}  Count")
    print("-" * 64)
    for bucket in ["attention", "ffn", "other", "cuda_graph"]:
        s = bucket_s.get(bucket, 0.0)
        n = bucket_count.get(bucket, 0)
        pct_total = s / grand * 100 if grand > 0 else 0.0
        pct_ex    = s / excl_graph * 100 if excl_graph > 0 and bucket != "cuda_graph" else 0.0
        ex_str    = f"{pct_ex:>11.1f}%" if bucket != "cuda_graph" else "         N/A"
        print(f"  {bucket:<12} {s:>10.3f} {pct_total:>11.1f}% {ex_str}  {n:>6}")
    print("-" * 64)
    print(f"  {'total':<12} {grand:>10.3f} {'100.0%':>12}")

    print(f"\nCUDA graph IDs seen ({len(graph_ids)} distinct):")
    for gid in sorted(graph_ids):
        print(f"  {gid}")

    print("\nNote: CUDA graphs are CUDAGraphMode.FULL — each replays the full")
    print("  model forward (all decoder layers, both attention + FFN + norms).")
    print("  They appear only in pure-decode steps; prefill runs without graphs.")
    print("\nNote: All GEMM kernels (turing_fp16_*gemm, cutlass) are bucketed")
    print("  into FFN. They include attention QKV/O projections too, but FFN")
    print("  projections dominate compute ~5:1 by parameter count.")

    print("\nTop 'other' kernel contributors:")
    top_other = sorted(unmatched.items(), key=lambda x: -x[1])[:10]
    for nm, s in top_other:
        short = nm if len(nm) <= 80 else nm[:77] + "..."
        print(f"  {s:.4f} s  {short}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <vllm_nsys_events.tsv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
