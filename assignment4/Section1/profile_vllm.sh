#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ENV="${VLLM_ENV:-/local1/groups/554g13/cse554-winter-2026/vllm_env}"
MODEL_PATH="${MODEL_PATH:-/local1/cse554/models/meta-llama/Llama-3.2-1B}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/profile_vllm_results.json}"
NSYS_OUT="${NSYS_OUT:-${SCRIPT_DIR}/vllm_nsys_profile}"
NSYS_STATS_CSV="${NSYS_STATS_CSV:-${SCRIPT_DIR}/vllm_nsys_kern_sum}"

export PYTHONPATH="${VLLM_ENV}:${SCRIPT_DIR}/../..${PYTHONPATH:+:$PYTHONPATH}"
export PATH="${VLLM_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

echo "Running vLLM benchmark under nsys"
echo "  model=${MODEL_PATH}"
echo "  input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, num_prompts=${NUM_PROMPTS}"
echo "  nsys output: ${NSYS_OUT}.nsys-rep"

nsys profile \
  --output "${NSYS_OUT}" \
  --trace cuda,nvtx \
  --force-overwrite true \
  -- \
  vllm bench throughput \
    --backend vllm \
    --model "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --output-json "${OUTPUT_JSON}" \
    --disable-detokenize \
    --tensor-parallel-size 1 \
    --dtype float16

python - <<'PY' "${OUTPUT_JSON}"
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
print("\n=== vLLM benchmark results ===")
for k, v in data.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
PY

echo ""
echo "Extracting kernel breakdown from nsys profile..."
nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output "${NSYS_STATS_CSV}" \
  --force-overwrite true \
  "${NSYS_OUT}.nsys-rep"

python "${SCRIPT_DIR}/parse_vllm_nsys.py" "${NSYS_STATS_CSV}_cuda_gpu_kern_sum.csv"
