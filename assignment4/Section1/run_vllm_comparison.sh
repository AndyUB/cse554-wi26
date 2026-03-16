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

echo "--- [1/2] vLLM benchmark ---"
cd "$SCRIPT_DIR"
python profile_vllm.py 2>&1 | tee profile_vllm_stdout.txt
echo ""

echo "--- [2/2] Our chunked-prefill benchmark ---"
python profile_chunked_fixed.py 2>&1 | tee profile_chunked_fixed_stdout.txt
echo ""

# ── Side-by-side summary ───────────────────────────────────────────────────────
echo "================================================"
echo " Summary"
echo "================================================"

VLLM_TPUT=$(python -c "
import json, sys
try:
    d = json.load(open('profile_vllm_results.json'))
    # vLLM reports 'throughput' in requests/s; also report tokens/s
    tput_req = d.get('throughput', 0)
    # total tokens = num_prompts * (input + output)
    total_tok = 100 * (512 + 512)
    elapsed   = d.get('elapsed_time', None)
    if elapsed:
        print(f'{total_tok/elapsed:.1f}')
    else:
        print('N/A')
except Exception as e:
    print('N/A')
" 2>/dev/null)

VLLM_ELAPSED=$(python -c "
import json
try:
    d = json.load(open('profile_vllm_results.json'))
    print(f\"{d.get('elapsed_time', 'N/A'):.3f}\")
except:
    print('N/A')
" 2>/dev/null)

CHUNK_ELAPSED=$(python -c "
import json
try:
    d = json.load(open('profile_chunked_fixed_itertimes.json'))
    import numpy as np
    iters = np.array(d['iter_times_ms'])
    print(f'{iters.sum()/1000:.3f}')
except:
    print('N/A')
" 2>/dev/null)

echo ""
echo "  vLLM elapsed time       : ${VLLM_ELAPSED} s"
echo "  vLLM throughput (tok/s) : ${VLLM_TPUT}"
echo ""
echo "  Chunked-prefill elapsed : ${CHUNK_ELAPSED} s"
echo ""
echo "See profile_vllm_results.json and profile_chunked_fixed_itertimes.json for full data."
echo "================================================"
