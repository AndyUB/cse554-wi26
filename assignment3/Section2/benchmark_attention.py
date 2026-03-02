import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    import flashinfer
except ImportError:
    flashinfer = None


MODELS: Dict[str, Dict[str, int]] = {
    "llama3_1b": {"hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8},
    "llama3_3b": {"hidden_size": 3072, "num_attention_heads": 24, "num_key_value_heads": 8},
    "llama3_8b": {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8},
}


@dataclass
class BenchCfg:
    warmup: int = 10
    iters: int = 50
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _cuda_sync()
    return start.elapsed_time(end) / iters


def _expand_kv_for_sdpa(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    # x: [B, H_kv, S, D] -> [B, H_q, S, D]
    repeat_factor = num_q_heads // x.shape[1]
    return x.repeat_interleave(repeat_factor, dim=1)


def _flashinfer_call(module_name: str, fn_candidates: List[str], *args, **kwargs):
    if flashinfer is None:
        raise RuntimeError("flashinfer is not installed. Run: uv pip install flashinfer-python==0.5.3")
    module = getattr(flashinfer, module_name)
    for name in fn_candidates:
        if hasattr(module, name):
            return getattr(module, name)(*args, **kwargs)
    raise RuntimeError(f"No supported FlashInfer function found in flashinfer.{module_name}: {fn_candidates}")


def bench_prefill_once(model: Dict[str, int], batch: int, p: int, cfg: BenchCfg) -> Tuple[float, float]:
    h_q = model["num_attention_heads"]
    h_kv = model["num_key_value_heads"]
    d = model["hidden_size"] // h_q

    q = torch.randn(batch, h_q, p, d, device=cfg.device, dtype=cfg.dtype)
    k = torch.randn(batch, h_kv, p, d, device=cfg.device, dtype=cfg.dtype)
    v = torch.randn(batch, h_kv, p, d, device=cfg.device, dtype=cfg.dtype)

    k_sdpa = _expand_kv_for_sdpa(k, h_q)
    v_sdpa = _expand_kv_for_sdpa(v, h_q)

    t_sdpa_ms = _time_cuda(lambda: F.scaled_dot_product_attention(q, k_sdpa, v_sdpa, is_causal=True), cfg.warmup, cfg.iters)

    def _run_flashinfer_prefill():
        # flashinfer prefill expects [qo_len, num_qo_heads, head_dim] and [kv_len, num_kv_heads, head_dim]
        # Run per batch for compatibility across API revisions.
        for b in range(batch):
            _flashinfer_call(
                "prefill",
                ["single_prefill_with_kv_cache", "single_prefill_with_kv_cache_return_lse"],
                q[b].transpose(0, 1),
                k[b].transpose(0, 1),
                v[b].transpose(0, 1),
                causal=True,
            )

    t_flashinfer_ms = _time_cuda(_run_flashinfer_prefill, cfg.warmup, cfg.iters)

    flops = 4.0 * batch * h_q * p * p * d
    sdpa_tflops = flops / (t_sdpa_ms * 1e-3) / 1e12
    flashinfer_tflops = flops / (t_flashinfer_ms * 1e-3) / 1e12
    return sdpa_tflops, flashinfer_tflops


def bench_decode_once(model: Dict[str, int], batch: int, c: int, cfg: BenchCfg, page_size: int = 1) -> Tuple[float, float]:
    h_q = model["num_attention_heads"]
    h_kv = model["num_key_value_heads"]
    d = model["hidden_size"] // h_q
    elem_size = torch.tensor([], dtype=cfg.dtype).element_size()

    q = torch.randn(batch, h_q, 1, d, device=cfg.device, dtype=cfg.dtype)
    k_cache = torch.randn(batch, h_kv, c, d, device=cfg.device, dtype=cfg.dtype)
    v_cache = torch.randn(batch, h_kv, c, d, device=cfg.device, dtype=cfg.dtype)

    k_sdpa = _expand_kv_for_sdpa(k_cache, h_q)
    v_sdpa = _expand_kv_for_sdpa(v_cache, h_q)

    t_sdpa_ms = _time_cuda(lambda: F.scaled_dot_product_attention(q, k_sdpa, v_sdpa, is_causal=False), cfg.warmup, cfg.iters)

    def _run_flashinfer_decode():
        for b in range(batch):
            _flashinfer_call(
                "decode",
                ["single_decode_with_kv_cache", "single_decode_with_kv_cache_return_lse"],
                q[b, :, 0, :],
                k_cache[b].transpose(0, 1),
                v_cache[b].transpose(0, 1),
                kv_layout="NHD",
                pos_encoding_mode="NONE",
            )

    t_flashinfer_ms = _time_cuda(_run_flashinfer_decode, cfg.warmup, cfg.iters)

    bytes_moved = 2.0 * batch * c * h_kv * d * elem_size / max(page_size, 1)
    sdpa_gbps = bytes_moved / (t_sdpa_ms * 1e-3) / 1e9
    flashinfer_gbps = bytes_moved / (t_flashinfer_ms * 1e-3) / 1e9
    return sdpa_gbps, flashinfer_gbps


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_all(cfg: BenchCfg, out_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run these benchmarks.")

    p_values = [2 ** i for i in range(7, 16)]
    b_values = [2 ** i for i in range(0, 7)]
    c_values = [2 ** i for i in range(7, 16)]
    page_sizes = [1, 2, 4, 8, 16]

    prefill_p_rows = []
    prefill_batch_rows = []
    decode_c_rows = []
    decode_batch_rows = []
    decode_page_rows = []

    for model_name, model in MODELS.items():
        for p in p_values:
            sdpa, fi = bench_prefill_once(model, batch=1, p=p, cfg=cfg)
            prefill_p_rows.append({"model": model_name, "p": p, "log2_p": int(math.log2(p)), "sdpa_tflops": sdpa, "flashinfer_tflops": fi})

        for b in b_values:
            sdpa, fi = bench_prefill_once(model, batch=b, p=1024, cfg=cfg)
            prefill_batch_rows.append({"model": model_name, "batch": b, "log2_batch": int(math.log2(b)), "sdpa_tflops": sdpa, "flashinfer_tflops": fi})

        for c in c_values:
            sdpa, fi = bench_decode_once(model, batch=1, c=c, cfg=cfg)
            decode_c_rows.append({"model": model_name, "c": c, "log2_c": int(math.log2(c)), "sdpa_gbps": sdpa, "flashinfer_gbps": fi})

        for b in b_values:
            sdpa, fi = bench_decode_once(model, batch=b, c=1024, cfg=cfg)
            decode_batch_rows.append({"model": model_name, "batch": b, "log2_batch": int(math.log2(b)), "sdpa_gbps": sdpa, "flashinfer_gbps": fi})

        for page_size in page_sizes:
            _, fi = bench_decode_once(model, batch=128, c=1024, cfg=cfg, page_size=page_size)
            decode_page_rows.append({"model": model_name, "page_size": page_size, "flashinfer_gbps": fi})

    _write_csv(out_dir / "prefill_vs_p.csv", prefill_p_rows)
    _write_csv(out_dir / "prefill_vs_batch.csv", prefill_batch_rows)
    _write_csv(out_dir / "decode_vs_c.csv", decode_c_rows)
    _write_csv(out_dir / "decode_vs_batch.csv", decode_batch_rows)
    _write_csv(out_dir / "decode_vs_page_size.csv", decode_page_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark torch SDPA vs FlashInfer for Assignment3 Section2 Q2.")
    parser.add_argument("--output-dir", type=Path, default=Path("assignment3/Section2/results"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    cfg = BenchCfg(warmup=args.warmup, iters=args.iters)
    run_all(cfg, args.output_dir)
    print(f"Wrote benchmark CSVs to: {args.output_dir}")
