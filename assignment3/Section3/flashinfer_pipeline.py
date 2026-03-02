from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import flashinfer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
#  Project utilities (local module)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper import WeightManager, extract_model_weights  # noqa: E402


# ---------------------------------------------------------------------------
#  Low-level data structures: paged KV-cache & per-request view
# ---------------------------------------------------------------------------
class DistKVPool:
    """Global *paged* KV-cache ("HND" = head-page-dim layout).

    Backing tensors: k_datas / v_datas of shape
        (num_layers, capacity, num_kv_heads, page_size, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity: int,
        page_size: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size

        # Simple free-list allocator
        self._free_pages: set[int] = set(range(capacity))

        kv_shape = (num_layers, capacity, num_kv_heads, page_size, head_dim)
        self.k_datas = torch.empty(kv_shape, dtype=torch.float16, device="cuda")
        self.v_datas = torch.empty_like(self.k_datas)

    @property
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    def alloc_page(self) -> int:
        return self._free_pages.pop()

    def free_page(self, idx: int) -> None:
        assert idx not in self._free_pages, "double-free detected"
        self._free_pages.add(idx)


class DistKVCache:
    """Light-weight *view* of a request's KV pages (no real storage)."""

    def __init__(self, pool: DistKVPool) -> None:
        self._pool = pool
        self._indices: list[int] = []
        self._seqlen: int = 0
        self.page_size = pool.page_size

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def indices(self) -> list[int]:
        return self._indices

    @property
    def last_page_offset(self) -> int:
        """Tokens used in the last page (in [1, page_size])."""
        if self._seqlen == 0:
            return 0
        remainder = self._seqlen % self.page_size
        return self.page_size if remainder == 0 else remainder

    def allocate_tokens(self, num_tokens: int) -> None:
        """Grow the cache to hold *num_tokens* additional tokens."""
        assert num_tokens > 0
        room_in_last = (self.page_size - self.last_page_offset) % self.page_size
        remaining = max(0, num_tokens - room_in_last)
        pages_needed = (remaining + self.page_size - 1) // self.page_size
        for _ in range(pages_needed):
            self._indices.append(self._pool.alloc_page())
        self._seqlen += num_tokens

    def release(self) -> None:
        for idx in self._indices:
            self._pool.free_page(idx)
        self._indices.clear()
        self._seqlen = 0


# ---------------------------------------------------------------------------
#  Helpers to convert a list of DistKVCache into FlashInfer ragged metadata
# ---------------------------------------------------------------------------

def build_kv_metadata(
    kvs: List[DistKVCache],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (indptr, indices, last_page_len) as int32 CUDA tensors."""
    kv_indptr: List[int] = [0]
    kv_indices: List[int] = []
    kv_last_page_len: List[int] = []

    for kv in kvs:
        kv_indices.extend(kv.indices)
        kv_indptr.append(kv_indptr[-1] + len(kv.indices))
        kv_last_page_len.append(kv.last_page_offset)

    device = "cuda"
    return (
        torch.tensor(kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indices, dtype=torch.int32, device=device),
        torch.tensor(kv_last_page_len, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
#  Simple request wrapper (prompt + generation buffer)
# ---------------------------------------------------------------------------
class Request:
    def __init__(self, req_id: int, prompt_ids: torch.Tensor, target_len: int) -> None:
        self.request_id = req_id
        self.prompt_token_ids = prompt_ids       # (prompt_len,)
        self.output_length = target_len
        self.output_token_ids = prompt_ids.clone()

    @property
    def prompt_length(self) -> int:
        return self.prompt_token_ids.size(0)

    @property
    def current_length(self) -> int:
        return self.output_token_ids.size(0)


# ---------------------------------------------------------------------------
#  Generation engine
# ---------------------------------------------------------------------------
class Engine:
    """Llama-3.2-1B engine using FlashInfer paged attention + RoPE kernels."""

    def __init__(self) -> None:
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.head_dim = 64
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.layers = 16

        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path)

        wm = WeightManager()
        wm.load_from_safe_tensor(self.weight_path)
        self.weights = extract_model_weights(wm.weight_map, self.layers)

        self.page_size = 16
        self.max_pages = 20_000
        self.pool = DistKVPool(
            num_layers=self.layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            capacity=self.max_pages,
            page_size=self.page_size,
        )

        self.kv_cache_map: Dict[int, DistKVCache] = {}

        workspace_bytes = 128 << 20  # 128 MiB
        self._fi_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._fi_workspace, "HND"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_workspace, "HND", use_tensor_cores=True
        )

    # ------------------------------------------------------------------
    #  Reset: release all KV caches (useful between benchmark runs)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        for kv in self.kv_cache_map.values():
            kv.release()
        self.kv_cache_map.clear()

    # ------------------------------------------------------------------
    #  One step (mixed prefill + decode)
    # ------------------------------------------------------------------
    def run(
        self,
        requests: List[Request],
        num_decode_req: int = 0,
        profile_ops: bool = False,
    ):
        """Run one transformer step.

        Parameters
        ----------
        requests : List[Request]
            Decode requests first (indices 0..num_decode_req-1), then prefill.
        num_decode_req : int
            Number of decode requests at the head of ``requests``.
        profile_ops : bool
            If True, return ``(tokens, op_times_dict)`` where op_times_dict
            gives per-category milliseconds summed over all layers.
        """
        # Collect CUDA events for profiling (no sync during forward pass)
        _events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        def _evt_start() -> torch.cuda.Event:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        def _evt_end(name: str, start: torch.cuda.Event) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            _events.append((name, start, end))

        with torch.inference_mode():
            # ----------------------------------------------------------------
            # 1) Build ragged input tensor and its CSR indptr
            # ----------------------------------------------------------------
            pieces: List[torch.Tensor] = []
            indptr: List[int] = [0]

            for idx, req in enumerate(requests):
                if idx < num_decode_req:
                    pieces.append(req.output_token_ids[-1:])
                    indptr.append(indptr[-1] + 1)
                else:
                    pieces.append(req.prompt_token_ids)
                    indptr.append(indptr[-1] + req.prompt_length)

            input_tensor = torch.cat(pieces).to("cuda")
            indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device="cuda")
            total_tokens = input_tensor.shape[0]
            num_decode_tokens = int(indptr_tensor[num_decode_req])
            num_prefill = len(requests) - num_decode_req

            # ----------------------------------------------------------------
            # 2) Create KV caches for new prefill requests
            # ----------------------------------------------------------------
            for idx, req in enumerate(requests):
                if idx >= num_decode_req:
                    self.kv_cache_map[req.request_id] = DistKVCache(self.pool)

            seq_lens_before: List[int] = [
                self.kv_cache_map[r.request_id].seqlen for r in requests
            ]
            seq_lens_before_t = torch.tensor(
                seq_lens_before, dtype=torch.int32, device="cuda"
            )

            # ----------------------------------------------------------------
            # 3) Allocate pages for all requests
            # ----------------------------------------------------------------
            for idx, req in enumerate(requests):
                kv = self.kv_cache_map[req.request_id]
                tokens = 1 if idx < num_decode_req else req.prompt_length
                kv.allocate_tokens(tokens)

            seq_lens_after: List[int] = [
                self.kv_cache_map[r.request_id].seqlen for r in requests
            ]
            seq_lens_after_t = torch.tensor(
                seq_lens_after, dtype=torch.int32, device="cuda"
            )

            # Global KV metadata (all requests, post-allocation)
            kv_indptr, kv_indices, kv_last_page_len = build_kv_metadata(
                [self.kv_cache_map[r.request_id] for r in requests]
            )

            # ----------------------------------------------------------------
            # 4) Plan FlashInfer wrappers (once, before the layer loop)
            # ----------------------------------------------------------------
            if num_prefill > 0:
                pre_kv_indptr, pre_kv_indices, pre_kv_last_page_len = build_kv_metadata(
                    [self.kv_cache_map[r.request_id] for r in requests[num_decode_req:]]
                )
                # qo_indptr starts at num_decode_tokens (absolute offset in q)
                self.prefill_wrapper.plan(
                    indptr_tensor[num_decode_req:],
                    pre_kv_indptr,
                    pre_kv_indices,
                    pre_kv_last_page_len,
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                    causal=True,
                )

            if num_decode_req > 0:
                dec_kv_indptr, dec_kv_indices, dec_kv_last_page_len = build_kv_metadata(
                    [self.kv_cache_map[r.request_id] for r in requests[:num_decode_req]]
                )
                self.decode_wrapper.plan(
                    dec_kv_indptr,
                    dec_kv_indices,
                    dec_kv_last_page_len,
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                    pos_encoding_mode="NONE",
                    data_type=torch.float16,
                )

            # Precompute (batch_index, position) for each token — used by
            # append_paged_kv_cache in every layer.
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                indptr_tensor,
                seq_lens_after_t,
                total_tokens,
            )

            # ----------------------------------------------------------------
            # 5) Forward pass through transformer layers
            # ----------------------------------------------------------------
            hidden: torch.Tensor = self.weights["embedding"][input_tensor]

            for layer in range(self.layers):
                kv_data = (
                    self.pool.k_datas[layer],
                    self.pool.v_datas[layer],
                )

                # === Self-attention sub-layer ================================
                if profile_ops:
                    s = _evt_start()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]

                k = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_k_proj_weight"][layer].T)
                    .view(-1, self.num_kv_heads, self.head_dim)
                )
                v = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_v_proj_weight"][layer].T)
                    .view(-1, self.num_kv_heads, self.head_dim)
                )
                q = (
                    ln_attn_in
                    .matmul(self.weights["self_attn_q_proj_weight"][layer].T)
                    .view(-1, self.num_qo_heads, self.head_dim)
                )
                if profile_ops:
                    _evt_end("norm_qkv", s)

                # ---- Rotary positional embedding ----------------------------
                # offsets = seq_lens_before: position 0 for new prefills,
                # current seqlen for decode tokens (so new token hits the right pos)
                if profile_ops:
                    s = _evt_start()
                flashinfer.apply_rope_inplace(
                    q,
                    k,
                    indptr_tensor,
                    seq_lens_before_t,
                    interleave=False,
                    rope_theta=500_000.0,
                )
                if profile_ops:
                    _evt_end("rope", s)

                # ---- Append new K/V to paged cache --------------------------
                if profile_ops:
                    s = _evt_start()
                flashinfer.append_paged_kv_cache(
                    k,
                    v,
                    batch_indices,
                    positions,
                    kv_data,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                    "HND",
                )
                if profile_ops:
                    _evt_end("kv_append", s)

                # ---- Attention compute --------------------------------------
                if profile_ops:
                    s = _evt_start()

                if num_prefill > 0:
                    # qo_indptr doesn't start at 0: output rows for decode
                    # positions will be zeros, filled in below.
                    attn_out = self.prefill_wrapper.run(q, kv_data)
                else:
                    attn_out = torch.zeros_like(q)

                if num_decode_req > 0:
                    decode_q = q[:num_decode_tokens]  # [num_decode_req, H_q, D]
                    decode_out = self.decode_wrapper.run(decode_q, kv_data)
                    attn_out[:num_decode_tokens] = decode_out

                if profile_ops:
                    _evt_end("attention", s)

                # Reshape [total_tokens, H_q, D] -> [total_tokens, hidden_dim]
                attn_out = attn_out.reshape(total_tokens, -1)

                # ---- O-projection + residual --------------------------------
                if profile_ops:
                    s = _evt_start()
                hidden = attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden
                if profile_ops:
                    _evt_end("o_proj", s)

                # === FFN sub-layer ==========================================
                if profile_ops:
                    s = _evt_start()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]

                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                hidden = (
                    (up * torch.nn.functional.silu(gate))
                    .matmul(self.weights["down_proj_weight"][layer].T)
                    + hidden
                )
                if profile_ops:
                    _evt_end("ffn", s)

            # ----------------------------------------------------------------
            # 6) Final LM head
            # ----------------------------------------------------------------
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (
                (hidden / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
            ).matmul(self.weights["lm_head_weight"].T)

            sample_ids = torch.argmax(logits, dim=-1)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            tokens_out = sample_ids[last_token_indices].cpu()

            if not profile_ops:
                return tokens_out

            # Sync and accumulate timing per operation category
            torch.cuda.synchronize()
            op_times: Dict[str, float] = {}
            for name, start_e, end_e in _events:
                op_times[name] = op_times.get(name, 0.0) + start_e.elapsed_time(end_e)
            return tokens_out, op_times

    # ------------------------------------------------------------------
    #  Full batched generation loop (prefill + iterative decode)
    # ------------------------------------------------------------------
    def generate_batched(self, prompts: List[str], rounds: int = 20) -> List[str]:
        print(f">>> starting batched generation ({rounds} rounds)")

        requests: List[Request] = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, rounds))

        # Prefill
        prefill_outputs = self.run(requests, num_decode_req=0)
        print("prefill pass finished - appending first generated token ...")
        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )

        # Iterative decode
        for _ in range(rounds - 1):
            decode_outputs = self.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )

        self.reset()
        return [
            self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True)
            for r in requests
        ]


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_prompts = (
        ["Hi, who are you?"] * 10
        + ["The University of Washington is located in"] * 10
    )

    engine = Engine()
    generated_texts = engine.generate_batched(sample_prompts, rounds=30)

    for idx, text in enumerate(generated_texts):
        print(f"[request {idx:02d}] {text}\n")
