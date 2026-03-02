# Section 1, Q2 Workflow

This document lists the exact commands to run for Section 1, Q2 experiments.

## 1) Build and run CUBLAS profiler

```bash
cd /workspace/cse554-wi26/assignment3/Section1
nvcc -O3 -std=c++17 profile_cublas_gemm.cu -lcublas -o profile_cublas_gemm
./profile_cublas_gemm cublas_gemm_perf.csv
```

## 2) Build CUTLASS profiler

```bash
cd /workspace/cse554-wi26
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_NVCC_ARCHS=75
cmake --build . -j 32
```

> Adjust architecture flags (`75`) if your GPU is different.

## 3) Run CUTLASS profiler for each split-K setting and shape

```bash
cd /workspace/cse554-wi26/cutlass/build/tools/profiler

for split_k in 1 2 4 8; do
  for shape in "512 512" "4096 4096" "14336 4096" "4096 1024" "1024 4096"; do
    read -r n k <<< "${shape}"
    ./cutlass_profiler \
      --kernels=sgemm \
      --A=f16:column \
      --B=f16:column \
      --C=f16:column \
      --accum=f16 \
      --m=128:2048:128 \
      --n=${n} \
      --k=${k} \
      --split_k_mode=serial \
      --split_k_slices=${split_k} \
      --profiling-iterations=100 \
      --output=cutlass_n${n}_k${k}_splitk_${split_k}.csv
  done
done
```

### If CSV files only contain headers

1. Make sure you run the command as a real multiline shell command (or from a `.sh` file).
   If `\n` appears literally in your terminal command, it is treated as text, not a newline.
2. Check whether kernels matched your filters:

```bash
head -n 5 cutlass_n512_k512_splitk_1.csv
wc -l cutlass_n512_k512_splitk_1.csv
```

If `wc -l` is `1`, no kernel configuration matched. The command above uses
`B=f16:col` and explicit `D=f16:row` because these are broadly supported GEMM
layout constraints for FP16 tensor-op kernels.

## 4) Collect best CUTLASS performance per shape

```bash
cd /workspace/cse554-wi26/assignment3/Section1
./run_cutlass_q2.sh /workspace/cse554-wi26/cutlass/build/tools/profiler

# then collect the best results
python collect_cutlass_best.py \
  /workspace/cse554-wi26/cutlass/build/tools/profiler/cutlass_n*_k*_splitk_*.csv \
  --output cutlass_gemm_perf.csv
```

## 5) Merge CUBLAS + CUTLASS CSVs and plot

```bash
cd /workspace/cse554-wi26/assignment3/Section1
python - <<'PY'
import pandas as pd

cublas = pd.read_csv('cublas_gemm_perf.csv')
cutlass = pd.read_csv('cutlass_gemm_perf.csv')
merged = pd.concat([cublas[['M','N','K','library','tflops']], cutlass[['M','N','K','library','tflops']]], ignore_index=True)
merged.to_csv('gemm_perf.csv', index=False)
print('Wrote gemm_perf.csv')
PY

python plot_gemm.py --input gemm_perf.csv --output-dir plots
```

This generates one plot per `(N, K)` pair under `assignment3/Section1/plots`.
