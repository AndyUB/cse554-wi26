import pandas as pd

# CUTLASS raw output
cut = pd.read_csv("keep.cutlass_raw.gemm.csv")

# CUTLASS CSV usually has columns like: m,n,k,Provider,Operation,Runtime,GFLOPs,...
# We want best GFLOPs for each (m,n,k) over all kernels and split_k values.
# The exact column name can be "GFLOPs" or "GFLOP/s" depending on version; handle both.
gflop_col = None
for c in ["GFLOPs", "GFLOP/s", "gflop/s", "gflops"]:
    if c in cut.columns:
        gflop_col = c
        break
if gflop_col is None:
    raise ValueError(f"Cannot find GFLOPs column. Columns are: {cut.columns.tolist()}")

# best = (cut
#         .groupby(["m","n","k"], as_index=False)[gflop_col]
#         .max()
#        )
# get index of max GFLOPs per (m,n,k)
idx = cut.groupby(["m", "n", "k"])[gflop_col].idxmax()

best = cut.loc[idx].reset_index(drop=True)

best["batch_size"] = best["m"]              # align with your plot script
best["N"] = best["n"]
best["K"] = best["k"]
best["library"] = "cutlass"
best["tflops"] = best[gflop_col] / 1000.0   # GFLOP/s -> TFLOP/s

# cutlass_perf = best[["batch_size","N","K","library","tflops"]]
cutlass_perf = best
cutlass_perf.to_csv("cutlass_perf_full.csv", index=False)
print("Wrote cutlass_perf_full.csv")

