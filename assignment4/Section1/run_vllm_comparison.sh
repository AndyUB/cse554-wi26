#!/usr/bin/env bash
# Compare vLLM vs our chunked-prefill on Llama-3.2-1B
# input=512 tokens, output=512 tokens, 100 requests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ENV="/local1/groups/554g13/cse554-winter-2026/vllm_env"

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="${VLLM_ENV}:${SCRIPT_DIR}/../..${PYTHONPATH:+:$PYTHONPATH}"

# Redirect all caches away from the home directory (disk quota exceeded)
CACHE_BASE="/local1/groups/554g13/cse554-winter-2026/.cache"
mkdir -p "${CACHE_BASE}"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export VLLM_CONFIG_ROOT="${CACHE_BASE}/vllm_config"
export FLASHINFER_WORKSPACE_BASE="${CACHE_BASE}"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/inductor"
export HF_HOME="${CACHE_BASE}/huggingface"
export TORCH_HOME="${CACHE_BASE}/torch"

echo "================================================"
echo " vLLM vs Chunked-Prefill Comparison"
echo " GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " input=512, output=512, 100 requests"
echo "================================================"
echo ""

echo "--- [1/3] vLLM benchmark + breakdown ---"
cd "$SCRIPT_DIR"
GENERATE_BREAKDOWN=1 bash profile_vllm.sh 2>&1 | tee profile_vllm_stdout.txt
echo ""

echo "--- [2/3] Our chunked-prefill benchmark + breakdown ---"
python profile_chunked_breakdown.py --backend chunked --out-dir "$SCRIPT_DIR" 2>&1 | tee profile_chunked_fixed_stdout.txt
echo ""

echo "--- [3/3] Combined breakdown comparison ---"
python profile_chunked_breakdown.py --backend both --out-dir "$SCRIPT_DIR" 2>&1 | tee profile_breakdown_stdout.txt
echo ""

# ── Side-by-side summary ───────────────────────────────────────────────────────
echo "================================================"
echo " Summary"
echo "================================================"

VLLM_TPUT=$(python -c "
import json
try:
    d = json.load(open('profile_vllm_results.json'))
    total_tok = 100 * (512 + 512)
    elapsed = d.get('elapsed_time', None)
    if elapsed:
        print(f'{total_tok/elapsed:.1f}')
    else:
        print('N/A')
except Exception:
    print('N/A')
" 2>/dev/null)

VLLM_ELAPSED=$(python -c "
import json
try:
    d = json.load(open('profile_vllm_results.json'))
    print(f\"{float(d.get('elapsed_time', 0.0)):.3f}\")
except Exception:
    print('N/A')
" 2>/dev/null)

CHUNK_ELAPSED=$(python -c "
import json
try:
    d = json.load(open('profile_chunked_results.json'))
    print(f\"{float(d['elapsed_s']):.3f}\")
except Exception:
    print('N/A')
" 2>/dev/null)

echo ""
echo "  vLLM elapsed time       : ${VLLM_ELAPSED} s"
echo "  vLLM throughput (tok/s) : ${VLLM_TPUT}"
echo ""
echo "  Chunked-prefill elapsed : ${CHUNK_ELAPSED} s"
echo ""
echo "See profile_vllm_results.json, profile_chunked_results.json," \
     "profile_vllm_breakdown.json, and profile_chunked_breakdown.json for full data."
echo "================================================"
