"""
Profile vLLM offline throughput with Llama-3.2-1B.

Settings:
  - input length  : 512 tokens (fixed, random dataset)
  - output length : 512 tokens (fixed)
  - num prompts   : 100
  - backend       : vllm

This script invokes vllm.benchmarks.throughput.main() programmatically
and saves results to profile_vllm_results.json.
"""

import sys
import argparse
import json

VLLM_ENV = "/local1/groups/554g13/cse554-winter-2026/vllm_env"
if VLLM_ENV not in sys.path:
    sys.path.insert(0, VLLM_ENV)

MODEL_PATH  = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
INPUT_LEN   = 512
OUTPUT_LEN  = 512
NUM_PROMPTS = 100
OUTPUT_JSON = "profile_vllm_results.json"

def main():
    from vllm.benchmarks.throughput import add_cli_args, main as vllm_bench_main

    parser = argparse.ArgumentParser(description="vLLM throughput benchmark")
    add_cli_args(parser)

    args = parser.parse_args([
        "--backend",          "vllm",
        "--model",            MODEL_PATH,
        "--dataset-name",     "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len",str(OUTPUT_LEN),
        "--num-prompts",      str(NUM_PROMPTS),
        "--output-json",      OUTPUT_JSON,
        "--disable-detokenize",
        # keep it on a single GPU – the shell script sets CUDA_VISIBLE_DEVICES=6
        "--tensor-parallel-size", "1",
        "--dtype",            "float16",
    ])

    print(f"Running vLLM benchmark: model={MODEL_PATH}")
    print(f"  input_len={INPUT_LEN}, output_len={OUTPUT_LEN}, num_prompts={NUM_PROMPTS}\n")

    vllm_bench_main(args)

    # Pretty-print the saved JSON summary
    try:
        with open(OUTPUT_JSON) as f:
            result = json.load(f)
        print("\n=== vLLM benchmark results ===")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
