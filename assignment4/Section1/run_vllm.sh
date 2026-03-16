#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ENV="${VLLM_ENV:-/local1/groups/554g13/cse554-winter-2026/vllm_env}"
MODEL_PATH="${MODEL_PATH:-/local1/cse554/models/meta-llama/Llama-3.2-1B}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/profile_vllm_results.json}"
OUTPUT_DIR="${SCRIPT_DIR}/torch_profile"
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${VLLM_ENV}:${SCRIPT_DIR}/../..${PYTHONPATH:+:$PYTHONPATH}"

echo "Running vLLM benchmark"
echo "  model=${MODEL_PATH}"
echo "  input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, num_prompts=${NUM_PROMPTS}"

export CUDA_VISIBLE_DEVICES=6

PYTHONPATH=/local1/groups/554g13/cse554-winter-2026/vllm_env /local1/groups/554g13/cse554-winter-2026/vllm_env/bin/vllm bench throughput \
  --model "${MODEL_PATH}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --output-json "${OUTPUT_JSON}" \
  --profile \
  --profiler-config.profiler=torch --profiler-config.torch_profiler_dir="${OUTPUT_DIR}"\
  --dtype float16

# python - <<'PY' "${OUTPUT_JSON}"
# import json
# import sys

# path = sys.argv[1]
# with open(path) as f:
#     data = json.load(f)
# print("\n=== vLLM benchmark results ===")
# for k, v in data.items():
#     if isinstance(v, float):
#         print(f"  {k}: {v:.4f}")
#     else:
#         print(f"  {k}: {v}")
# PY
