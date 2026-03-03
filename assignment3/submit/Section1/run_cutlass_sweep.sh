#!/usr/bin/env bash
set -euo pipefail

PROF=../cutlass/build/tools/profiler/cutlass_profiler

OUT_BASE=cutlass_raw
rm -f ${OUT_BASE}.gemm.csv

M_RANGE="128:2048:128"
CASES=("512 512" "4096 4096" "14336 4096" "4096 1024" "1024 4096")
SPLITS=(1 2 4 8)

for nk in "${CASES[@]}"; do
  read -r N K <<< "${nk}"

  for sk in "${SPLITS[@]}"; do
    echo "Running N=${N} K=${K} split_k=${sk}"

    ${PROF} \
      --operation=Gemm \
      --providers=cutlass \
      --verification-enabled=false \
      --profiling-iterations=100 \
      --m=${M_RANGE} --n=${N} --k=${K} \
      --split_k_mode=serial --split_k_slices=${sk} \
      --op_class=tensorop --accum=f32 \
      --A=f16:column --B=f16:column --C=f16:column \
      --output=${OUT_BASE}.csv --append=true \
      --tags=library:cutlass,split_k:${sk},N:${N},K:${K}
  done
done
