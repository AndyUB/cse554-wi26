"""
Parse vllm_nsys_ranges.tsv (nsys NVTX range summary) and compute
attention / FFN / norm / other breakdown without double-counting.

Hierarchy in vLLM layerwise NVTX:
  LlamaForCausalLM
    .model.layers.N                  ← decoder layer
      .self_attn                     ← attention (parent of qkv_proj, attn, o_proj, ...)
        .qkv_proj / .rotary_emb / .attn / .o_proj
      .mlp                           ← FFN (parent of gate_up_proj, act_fn, down_proj)
        .gate_up_proj / .act_fn / .down_proj
      .input_layernorm
      .post_attention_layernorm
    .model.norm                      ← final norm
    .logits_processor

Strategy: use the "middle" level — sum .self_attn rows for attention, .mlp rows for FFN,
.layernorm rows for norm.  These ranges are non-overlapping at the same depth level and
each fully contains its children, so no double-counting occurs.

The same module can appear on multiple rows with different input shapes (prefill vs decode
with different batch sizes); we sum ALL such rows.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def parse_time(s: str) -> float:
    """Parse a time string like '344.329 ms', '63.398 μs', '8.365 s' → seconds."""
    s = s.strip()
    # Remove thousands separators that may appear
    s = s.replace(",", "")
    # Match number + unit
    m = re.match(r"([0-9.]+)\s*([a-zμ]+)", s)
    if not m:
        return 0.0
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit in ("s",):
        return value
    if unit in ("ms",):
        return value * 1e-3
    if unit in ("μs", "us", "\u03bcs"):
        return value * 1e-6
    if unit in ("ns",):
        return value * 1e-9
    return value  # fallback: assume seconds


# ---------------------------------------------------------------------------
# Module path classification
# ---------------------------------------------------------------------------

def classify(module_path: str) -> str:
    """
    Classify a module path into a bucket.

    We use ONLY the "direct parent" level:
      - ends with .self_attn           → attention
      - ends with .mlp                 → ffn
      - ends with layernorm            → norm
      - LlamaForCausalLM (top level)   → skip (would double-count everything)
      - .model.layers.N (layer level)  → skip (would double-count attn+ffn+norm)
      - everything else                → other
    """
    # Normalise
    p = module_path.strip()

    # Skip the very top-level model wrapper (contains everything)
    if re.match(r"^LlamaForCausalLM$", p):
        return "skip"

    # Skip the decoder-layer wrapper (contains attn + ffn + norms)
    if re.match(r"^LlamaForCausalLM\.model\.layers\.\d+$", p):
        return "skip"

    # Attention: the self_attn module itself (not its children)
    if re.match(r"^LlamaForCausalLM\.model\.layers\.\d+\.self_attn$", p):
        return "attention"

    # FFN: the mlp module itself (not its children)
    if re.match(r"^LlamaForCausalLM\.model\.layers\.\d+\.mlp$", p):
        return "ffn"

    # Layer norms
    if re.search(r"layernorm|layer_norm", p, re.IGNORECASE):
        return "norm"
    if re.match(r"^LlamaForCausalLM\.model\.norm$", p):
        return "norm"

    # Sub-components of self_attn and mlp — skip to avoid double-counting
    if re.match(r"^LlamaForCausalLM\.model\.layers\.\d+\.self_attn\.", p):
        return "skip"
    if re.match(r"^LlamaForCausalLM\.model\.layers\.\d+\.mlp\.", p):
        return "skip"

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
    skipped_s: float = 0.0

    with open(path, newline="", encoding="utf-8") as f:
        header = f.readline()   # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue

            total_time_str = parts[1]   # column 2: "Total Time"
            range_str      = parts[9]   # column 10: "Range" (the NVTX marker text)

            total_s = parse_time(total_time_str)

            # Extract module path from range JSON-like string
            # Format: :{'Module': 'LlamaForCausalLM.model...', ...}
            m = re.search(r"'Module':\s*'([^']+)'", range_str)
            if not m:
                bucket_s["other"] += total_s
                continue

            module_path = m.group(1)
            bucket = classify(module_path)

            if bucket == "skip":
                skipped_s += total_s
            else:
                bucket_s[bucket] += total_s

    grand = sum(v for k, v in bucket_s.items())

    print("\n=== vLLM NVTX range breakdown (eager mode, wall-clock) ===")
    print(f"{'Category':<12} {'Time (s)':>10} {'% of counted':>14}")
    print("-" * 40)
    for bucket in ["attention", "ffn", "norm", "other"]:
        s = bucket_s.get(bucket, 0.0)
        pct = s / grand * 100 if grand > 0 else 0.0
        print(f"  {bucket:<10} {s:>10.3f} {pct:>13.1f}%")
    print("-" * 40)
    print(f"  {'total':<10} {grand:>10.3f} {'100.0%':>14}")
    print(f"\n(skipped {skipped_s:.3f} s from parent wrappers that would double-count)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <vllm_nsys_ranges.tsv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
