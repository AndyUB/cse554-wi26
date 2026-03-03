import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import flashinfer


MODELS: Dict[str, Dict[str, int]] = {
    "llama3_1b": {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
    "llama3_3b": {
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
    },
    "llama3_8b": {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
}


@dataclass
class BenchCfg:
    warmup: int = 10
    iters: int = 50
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    cuda_sync()
    return start.elapsed_time(end) / iters


def safe_time_cuda(fn, warmup: int, iters: int) -> float:
    try:
        return time_cuda(fn, warmup, iters)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return float("nan")


def bench_prefill_once(
    model: Dict[str, int], batch: int, p: int, cfg: BenchCfg
) -> Tuple[float, float]:
    h_q = model["num_attention_heads"]
    h_kv = model["num_key_value_heads"]
    d = model["hidden_size"] // h_q

    q = torch.randn(batch, h_q, p, d, device=cfg.device, dtype=cfg.dtype)
    k = torch.randn(batch, h_kv, p, d, device=cfg.device, dtype=cfg.dtype)
    v = torch.randn(batch, h_kv, p, d, device=cfg.device, dtype=cfg.dtype)

    t_sdpa_ms = safe_time_cuda(
        lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        ),
        cfg.warmup,
        cfg.iters,
    )

    q_fi = q.permute(0, 2, 1, 3).reshape(batch * p, h_q, d).contiguous()
    k_fi = k.permute(0, 2, 1, 3).reshape(batch * p, h_kv, d).contiguous()
    v_fi = v.permute(0, 2, 1, 3).reshape(batch * p, h_kv, d).contiguous()
    qo_indptr = torch.arange(
        0, (batch + 1) * p, p, dtype=torch.int32, device=cfg.device
    )
    kv_indptr = qo_indptr

    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=cfg.device)
    prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace)
    prefill_wrapper.plan(qo_indptr, kv_indptr, h_q, h_kv, d, causal=True)

    t_flashinfer_ms = time_cuda(
        lambda: prefill_wrapper.run(q_fi, k_fi, v_fi),
        cfg.warmup,
        cfg.iters,
    )

    flops = 4.0 * batch * h_q * p * p * d
    sdpa_tflops = flops / (t_sdpa_ms * 1e-3) / 1e12
    flashinfer_tflops = flops / (t_flashinfer_ms * 1e-3) / 1e12
    return sdpa_tflops, flashinfer_tflops


def bench_decode_once(
    model: Dict[str, int], batch: int, c: int, cfg: BenchCfg, page_size: int = 1
) -> Tuple[float, float]:
    h_q = model["num_attention_heads"]
    h_kv = model["num_key_value_heads"]
    d = model["hidden_size"] // h_q
    elem_size = torch.tensor([], dtype=cfg.dtype).element_size()

    q = torch.randn(batch, h_q, 1, d, device=cfg.device, dtype=cfg.dtype)
    k_cache = torch.randn(batch, h_kv, c, d, device=cfg.device, dtype=cfg.dtype)
    v_cache = torch.randn(batch, h_kv, c, d, device=cfg.device, dtype=cfg.dtype)

    t_sdpa_ms = safe_time_cuda(
        lambda: F.scaled_dot_product_attention(
            q, k_cache, v_cache, is_causal=False, enable_gqa=True
        ),
        cfg.warmup,
        cfg.iters,
    )

    pages_per_seq = (c + page_size - 1) // page_size
    last_pg = c - (pages_per_seq - 1) * page_size
    k_paged = k_cache.permute(0, 2, 1, 3).reshape(
        batch, pages_per_seq, page_size, h_kv, d
    )
    v_paged = v_cache.permute(0, 2, 1, 3).reshape(
        batch, pages_per_seq, page_size, h_kv, d
    )
    kv_paged = (
        torch.stack([k_paged, v_paged], dim=2)
        .reshape(batch * pages_per_seq, 2, page_size, h_kv, d)
        .contiguous()
    )

    q_fi = q[:, :, 0, :].contiguous()
    indptr = torch.arange(
        0,
        (batch + 1) * pages_per_seq,
        pages_per_seq,
        dtype=torch.int32,
        device=cfg.device,
    )
    indices = torch.arange(batch * pages_per_seq, dtype=torch.int32, device=cfg.device)
    last_page_len = torch.full((batch,), last_pg, dtype=torch.int32, device=cfg.device)

    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=cfg.device)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace)
    decode_wrapper.plan(
        indptr,
        indices,
        last_page_len,
        h_q,
        h_kv,
        d,
        page_size,
        pos_encoding_mode="NONE",
        data_type=cfg.dtype,
    )

    t_flashinfer_ms = time_cuda(
        lambda: decode_wrapper.run(q_fi, kv_paged),
        cfg.warmup,
        cfg.iters,
    )

    bytes_moved = 2.0 * batch * c * h_kv * d * elem_size
    sdpa_gbps = bytes_moved / (t_sdpa_ms * 1e-3) / 1e9
    flashinfer_gbps = bytes_moved / (t_flashinfer_ms * 1e-3) / 1e9
    return sdpa_gbps, flashinfer_gbps


def write_csv(path: Path, rows: list[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_all(cfg: BenchCfg, out_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run these benchmarks.")

    p_values = [2**i for i in range(7, 16)]
    b_values = [2**i for i in range(0, 7)]
    c_values = [2**i for i in range(7, 16)]
    page_sizes = [1, 2, 4, 8, 16]

    prefill_p_rows = []
    prefill_batch_rows = []
    decode_c_rows = []
    decode_batch_rows = []
    decode_page_rows = []

    for model_name, model in MODELS.items():
        for p in p_values:
            sdpa, fi = bench_prefill_once(model, batch=1, p=p, cfg=cfg)
            torch.cuda.empty_cache()
            prefill_p_rows.append(
                {
                    "model": model_name,
                    "p": p,
                    "log2_p": int(math.log2(p)),
                    "sdpa_tflops": sdpa,
                    "flashinfer_tflops": fi,
                }
            )

        for b in b_values:
            sdpa, fi = bench_prefill_once(model, batch=b, p=1024, cfg=cfg)
            torch.cuda.empty_cache()
            prefill_batch_rows.append(
                {
                    "model": model_name,
                    "batch": b,
                    "log2_batch": int(math.log2(b)),
                    "sdpa_tflops": sdpa,
                    "flashinfer_tflops": fi,
                }
            )

        for c in c_values:
            sdpa, fi = bench_decode_once(model, batch=1, c=c, cfg=cfg)
            torch.cuda.empty_cache()
            decode_c_rows.append(
                {
                    "model": model_name,
                    "c": c,
                    "log2_c": int(math.log2(c)),
                    "sdpa_gbps": sdpa,
                    "flashinfer_gbps": fi,
                }
            )

        for b in b_values:
            sdpa, fi = bench_decode_once(model, batch=b, c=1024, cfg=cfg)
            torch.cuda.empty_cache()
            decode_batch_rows.append(
                {
                    "model": model_name,
                    "batch": b,
                    "log2_batch": int(math.log2(b)),
                    "sdpa_gbps": sdpa,
                    "flashinfer_gbps": fi,
                }
            )

        for page_size in page_sizes:
            _, fi = bench_decode_once(
                model, batch=128, c=1024, cfg=cfg, page_size=page_size
            )
            torch.cuda.empty_cache()
            decode_page_rows.append(
                {"model": model_name, "page_size": page_size, "flashinfer_gbps": fi}
            )

    write_csv(out_dir / "prefill_vs_p.csv", prefill_p_rows)
    write_csv(out_dir / "prefill_vs_batch.csv", prefill_batch_rows)
    write_csv(out_dir / "decode_vs_c.csv", decode_c_rows)
    write_csv(out_dir / "decode_vs_batch.csv", decode_batch_rows)
    write_csv(out_dir / "decode_vs_page_size.csv", decode_page_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark torch SDPA vs FlashInfer for Assignment3 Section2 Q2."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("assignment3/Section2/results")
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    cfg = BenchCfg(warmup=args.warmup, iters=args.iters)
    run_all(cfg, args.output_dir)
    print(f"Wrote benchmark CSVs to: {args.output_dir}")
