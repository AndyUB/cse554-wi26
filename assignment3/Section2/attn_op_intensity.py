cfg = {
    "1B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "3B": {"num_qo_heads": 24, "num_kv_heads": 8, "head_dim": 128},
    "8B": {"num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}


def prefill_intensity(model: str, p: int) -> float:
    config = cfg[model]
    num_qo_heads = config["num_qo_heads"]
    num_kv_heads = config["num_kv_heads"]

    intensity = num_qo_heads * p / (num_qo_heads + num_kv_heads)
    return intensity


def decode_intensity(model: str, c: int) -> float:
    config = cfg[model]
    num_qo_heads = config["num_qo_heads"]
    num_kv_heads = config["num_kv_heads"]

    intensity = num_qo_heads * c / (num_qo_heads + num_kv_heads * c)
    return intensity


vals = [2**i for i in range(7, 16)]
models = list(cfg.keys())

for val in vals:
    print(f"p={val}:")
    for model in models:
        intensity = prefill_intensity(model, val)
        print(f"  {model} prefill intensity: {intensity:.2f}")

for val in vals:
    print(f"c={val}:")
    for model in models:
        intensity = decode_intensity(model, val)
        print(f"  {model} decode intensity: {intensity:.2f}")
