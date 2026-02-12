import sys
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


class Engine:
    def __init__(self):
        self.weight_path = "/local1/cse554/models/meta-llama/Llama-3.2-1B"
        self.head_dim = 64
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.layers = 16

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path,
            local_files_only=True,
        )

        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)

        self.kv_cache = {}

    def reset_cache(self):
        self.kv_cache = {}

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
        return x / rms

    def _run_layers_prefill(
        self, padded_input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_tensor = padded_input_ids.to(device="cuda", dtype=torch.long)
        attention_mask = attention_mask.to(device="cuda", dtype=torch.bool)
        hidden_state = self.weights["embedding"][input_tensor]  # [B, T, H]

        bsz, seq_len = input_tensor.shape
        lengths = attention_mask.sum(dim=1)

        for current_layer in range(self.layers):
            normalized_x = self._rms_norm(hidden_state)
            x = (
                normalized_x.to(torch.float16)
                * self.weights["layernormAttn_weight"][current_layer]
            )

            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            for batch_idx in range(bsz):
                apply_rope(q[batch_idx], output=q[batch_idx], head_dim=self.head_dim, offset=0)
                apply_rope(k[batch_idx], output=k[batch_idx], head_dim=self.head_dim, offset=0)

            sub_q = q.view(bsz, seq_len, self.num_qo_heads, self.head_dim)
            sub_k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            sub_v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

            group_size = self.num_qo_heads // self.num_kv_heads
            sub_k_expand = sub_k.repeat_interleave(group_size, dim=2)
            sub_v_expand = sub_v.repeat_interleave(group_size, dim=2)

            sub_q_t = sub_q.permute(0, 2, 1, 3)
            sub_k_t = sub_k_expand.permute(0, 2, 1, 3)
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * (
                1.0 / (self.head_dim**0.5)
            )

            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device)
            ).view(1, 1, seq_len, seq_len)
            key_valid_mask = attention_mask.view(bsz, 1, 1, seq_len)
            query_valid_mask = attention_mask.view(bsz, 1, seq_len, 1)

            scores = scores.masked_fill(~causal_mask, float("-inf"))
            scores = scores.masked_fill(~key_valid_mask, float("-inf"))
            scores = scores.masked_fill(~query_valid_mask, 0.0)

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = attn_weights * query_valid_mask

            v_t = sub_v_expand.permute(0, 2, 1, 3)
            attn_output = torch.matmul(attn_weights, v_t)
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(
                bsz, seq_len, self.num_qo_heads * self.head_dim
            )

            prefill_output = (
                attn_output.matmul(self.weights["o_proj_weight"][current_layer].t())
                + hidden_state
            )

            normalized_x = self._rms_norm(prefill_output)
            layernormFFN_output = (
                normalized_x.to(torch.float16)
                * self.weights["layernormFFN_weight"][current_layer]
            )
            up_proj_output = layernormFFN_output.matmul(
                self.weights["up_proj_weight"][current_layer].t()
            )
            gate_proj_output = layernormFFN_output.matmul(
                self.weights["gate_proj_weight"][current_layer].t()
            )
            activation_output = up_proj_output * torch.nn.functional.silu(
                gate_proj_output
            )
            hidden_state = (
                activation_output.matmul(
                    self.weights["down_proj_weight"][current_layer].t()
                )
                + prefill_output
            )

            if current_layer not in self.kv_cache:
                self.kv_cache[current_layer] = {"k": {}, "v": {}}

            for sample_idx in range(bsz):
                length = int(lengths[sample_idx].item())
                self.kv_cache[current_layer]["k"][sample_idx] = sub_k[
                    sample_idx, :length
                ].clone()
                self.kv_cache[current_layer]["v"][sample_idx] = sub_v[
                    sample_idx, :length
                ].clone()

        normalized_x = self._rms_norm(hidden_state)
        model_output = (
            normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        )
        logits = model_output.matmul(self.weights["lm_head_weight"].t())

        last_token_indices = (lengths - 1).to(dtype=torch.int64)
        batch_indices = torch.arange(bsz, device=logits.device)
        last_logits = logits[batch_indices, last_token_indices]
        return torch.argmax(last_logits, dim=-1)

    def _run_layers_decode_batched(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_tensor = input_ids.to(device="cuda", dtype=torch.long)
        hidden_state = self.weights["embedding"][input_tensor]
        bsz = input_tensor.shape[0]

        for current_layer in range(self.layers):
            normalized_x = self._rms_norm(hidden_state)
            x = (
                normalized_x.to(torch.float16)
                * self.weights["layernormAttn_weight"][current_layer]
            )

            k_new = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v_new = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            if current_layer not in self.kv_cache:
                self.kv_cache[current_layer] = {"k": {}, "v": {}}

            k_total_list = []
            v_total_list = []
            for sample_idx in range(bsz):
                cache_k = self.kv_cache[current_layer]["k"].get(sample_idx)
                cache_v = self.kv_cache[current_layer]["v"].get(sample_idx)

                offset = 0 if cache_k is None else cache_k.shape[0]
                apply_rope(q[sample_idx], output=q[sample_idx], head_dim=self.head_dim, offset=offset)
                apply_rope(k_new[sample_idx], output=k_new[sample_idx], head_dim=self.head_dim, offset=offset)

                new_k = k_new[sample_idx].view(1, self.num_kv_heads, self.head_dim)
                new_v = v_new[sample_idx].view(1, self.num_kv_heads, self.head_dim)

                if cache_k is None:
                    total_k = new_k
                    total_v = new_v
                else:
                    total_k = torch.cat([cache_k, new_k], dim=0)
                    total_v = torch.cat([cache_v, new_v], dim=0)

                self.kv_cache[current_layer]["k"][sample_idx] = total_k
                self.kv_cache[current_layer]["v"][sample_idx] = total_v

                k_total_list.append(total_k)
                v_total_list.append(total_v)

            lengths = torch.tensor([k.shape[0] for k in k_total_list], device="cuda")
            max_len = int(lengths.max().item())

            k_padded = torch.zeros(
                bsz,
                max_len,
                self.num_kv_heads,
                self.head_dim,
                device="cuda",
                dtype=k_total_list[0].dtype,
            )
            v_padded = torch.zeros(
                bsz,
                max_len,
                self.num_kv_heads,
                self.head_dim,
                device="cuda",
                dtype=v_total_list[0].dtype,
            )
            key_mask = torch.zeros(bsz, max_len, device="cuda", dtype=torch.bool)

            for sample_idx in range(bsz):
                cur_len = k_total_list[sample_idx].shape[0]
                k_padded[sample_idx, :cur_len] = k_total_list[sample_idx]
                v_padded[sample_idx, :cur_len] = v_total_list[sample_idx]
                key_mask[sample_idx, :cur_len] = True

            sub_q = q.view(bsz, 1, self.num_qo_heads, self.head_dim)
            group_size = self.num_qo_heads // self.num_kv_heads
            k_expand = k_padded.repeat_interleave(group_size, dim=2)
            v_expand = v_padded.repeat_interleave(group_size, dim=2)

            sub_q_t = sub_q.permute(0, 2, 1, 3)
            k_t = k_expand.permute(0, 2, 1, 3)
            scores = torch.matmul(sub_q_t, k_t.transpose(-2, -1)) * (
                1.0 / (self.head_dim**0.5)
            )
            scores = scores.masked_fill(~key_mask.view(bsz, 1, 1, max_len), float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            v_t = v_expand.permute(0, 2, 1, 3)
            attn_output = torch.matmul(attn_weights, v_t)
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(
                bsz, 1, self.num_qo_heads * self.head_dim
            )

            decode_output = (
                attn_output.matmul(self.weights["o_proj_weight"][current_layer].t())
                + hidden_state
            )

            normalized_x = self._rms_norm(decode_output)
            layernormFFN_output = (
                normalized_x.to(torch.float16)
                * self.weights["layernormFFN_weight"][current_layer]
            )
            up_proj_output = layernormFFN_output.matmul(
                self.weights["up_proj_weight"][current_layer].t()
            )
            gate_proj_output = layernormFFN_output.matmul(
                self.weights["gate_proj_weight"][current_layer].t()
            )
            activation_output = up_proj_output * torch.nn.functional.silu(
                gate_proj_output
            )
            hidden_state = (
                activation_output.matmul(
                    self.weights["down_proj_weight"][current_layer].t()
                )
                + decode_output
            )

        normalized_x = self._rms_norm(hidden_state)
        model_output = (
            normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        )
        logits = model_output.matmul(self.weights["lm_head_weight"].t())
        return torch.argmax(logits[:, -1, :], dim=-1)

    def run(self, input_ids_list, prefill=True):
        tensor_inputs = []
        for ids in input_ids_list:
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)
            tensor_inputs.append(ids.to(dtype=torch.long))

        if prefill:
            padded = pad_sequence(tensor_inputs, batch_first=True, padding_value=0)
            attention_mask = torch.zeros_like(padded, dtype=torch.bool)
            for i, ids in enumerate(tensor_inputs):
                attention_mask[i, : ids.shape[0]] = True
            return self._run_layers_prefill(padded, attention_mask)

        step_tokens = torch.stack([ids[-1:] for ids in tensor_inputs], dim=0)
        return self._run_layers_decode_batched(step_tokens)

    @torch.inference_mode()
    def generate_batched(self, input_strings: List[str], rounds=20):
        self.reset_cache()

        input_ids_list = []
        for text in input_strings:
            ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
            input_ids_list.append(ids)

        output_ids_list = [ids.clone().to("cuda") for ids in input_ids_list]

        new_tokens = self.run(output_ids_list, prefill=True)
        for i in range(len(output_ids_list)):
            output_ids_list[i] = torch.cat(
                (output_ids_list[i], new_tokens[i : i + 1]), dim=0
            )

        for _ in range(rounds - 1):
            step_inputs = [output_ids[-1:] for output_ids in output_ids_list]
            new_tokens = self.run(step_inputs, prefill=False)
            for i in range(len(output_ids_list)):
                output_ids_list[i] = torch.cat(
                    (output_ids_list[i], new_tokens[i : i + 1]),
                    dim=0,
                )

        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(
                self.tokenizer.decode(output_ids, skip_special_tokens=True)
            )
        return output_text_list


if __name__ == "__main__":
    prompts = [
        "Hi, who are you?",
        "The University of Washington is ",
        "Seattle is a city in ",
    ]
    engine = Engine()
    output_text = engine.generate_batched(prompts, rounds=20)
    print("Generated Text:", output_text)
