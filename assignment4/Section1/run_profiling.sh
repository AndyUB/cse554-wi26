#!/usr/bin/env bash
# Profile all scheduling strategies on GPU 6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=6

echo "============================================"
echo " GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================"

echo ""
echo "--- [1/3] Naive vs Continuous (uniform input/output) ---"
python "$SCRIPT_DIR/profile_comparison.py" > "$SCRIPT_DIR/profile_comparison_results.txt" 2>&1
echo "Done. Results -> profile_comparison_results.txt"

echo ""
echo "--- [2/3] Chunked Prefill vs Continuous (lognormal input) ---"
python "$SCRIPT_DIR/profile_lognormal.py" > "$SCRIPT_DIR/profile_lognormal_results.txt" 2>&1
echo "Done. Results -> profile_lognormal_results.txt"

echo ""
echo "--- [3/3] Per-iteration timing scatter plot ---"
python "$SCRIPT_DIR/profile_iteration_scatter.py" > "$SCRIPT_DIR/profile_iteration_scatter_results.txt" 2>&1
echo "Done. Results -> profile_iteration_scatter_results.txt"
echo "      Plot   -> iteration_times_scatter.png"
echo "      Data   -> iteration_times.json"

echo ""
echo "All profiling complete."
