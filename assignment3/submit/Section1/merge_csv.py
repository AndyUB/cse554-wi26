import pandas as pd

cutlass = pd.read_csv("cutlass_perf_full.csv")
cublas  = pd.read_csv("cublas_perf.csv")   # must match schema

gemm_perf = pd.concat([cutlass, cublas], ignore_index=True)
gemm_perf.to_csv("gemm_perf.csv", index=False)
print("Wrote gemm_perf.csv")