M = list(range(128, 2048 + 128, 128))
N_K = [(512, 512), (4096, 4096), (14336, 4096), (4096, 1024), (1024, 4096)]

print(M)
print(N_K)


def compute_op_intensity(m, n, k):
    intensity = (m * n * k) / (m * n + n * k + m * k)
    return intensity


for m in M:
    for n, k in N_K:
        intensity = compute_op_intensity(m, n, k)
        print(f"m: {m}, n: {n}, k: {k}, operational intensity: {intensity:.2f}")
