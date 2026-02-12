import sys
from typing import List

import torch
from transformers import AutoTokenizer

sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


class Engine:
    """
    A class to manage the generation engine.
    """

    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
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

    def _run_single(
        self, input_ids: torch.Tensor, sample_idx: int, prefill: bool
    ) -> int:
        input_tensor = input_ids.to(device="cuda", dtype=torch.int32)
        hidden_state = self.weights["embedding"][input_tensor]

        for current_layer in range(self.layers):
            normalized_x = self._rms_norm(hidden_state)
            x = (
                normalized_x.to(torch.float16)
                * self.weights["layernormAttn_weight"][current_layer]
            )

            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            if current_layer not in self.kv_cache:
                self.kv_cache[current_layer] = {"k": {}, "v": {}}

            cache_k = self.kv_cache[current_layer]["k"].get(sample_idx)
            cache_v = self.kv_cache[current_layer]["v"].get(sample_idx)
            offset = 0 if (prefill or cache_k is None) else cache_k.shape[0]

            apply_rope(q, output=q, head_dim=self.head_dim, offset=offset)
            apply_rope(k, output=k, head_dim=self.head_dim, offset=offset)

            sub_q = q.view(-1, self.num_qo_heads, self.head_dim)
            sub_k_new = k.view(-1, self.num_kv_heads, self.head_dim)
            sub_v_new = v.view(-1, self.num_kv_heads, self.head_dim)

            if prefill or cache_k is None:
                sub_k_total = sub_k_new
                sub_v_total = sub_v_new
            else:
                sub_k_total = torch.cat([cache_k, sub_k_new], dim=0)
                sub_v_total = torch.cat([cache_v, sub_v_new], dim=0)

            self.kv_cache[current_layer]["k"][sample_idx] = sub_k_total
            self.kv_cache[current_layer]["v"][sample_idx] = sub_v_total

            group_size = self.num_qo_heads // self.num_kv_heads
            sub_k = sub_k_total.repeat_interleave(group_size, dim=1)
            sub_v = sub_v_total.repeat_interleave(group_size, dim=1)

            sub_q_t = sub_q.permute(1, 0, 2)
            sub_k_t = sub_k.permute(1, 0, 2)
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * (
                1.0 / (self.head_dim**0.5)
            )

            if prefill:
                n_q = sub_q.shape[0]
                n_k = sub_k.shape[0]
                causal_mask = torch.tril(
                    torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)
                )
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            v_t = sub_v.permute(1, 0, 2)
            attn_output = torch.matmul(attn_weights, v_t).permute(1, 0, 2)

            attn_output = attn_output.reshape(-1, self.num_qo_heads * self.head_dim)
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

        normalized_x = self._rms_norm(hidden_state)
        model_output = (
            normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        )
        logits = model_output.matmul(self.weights["lm_head_weight"].t())
        sample_output = torch.argmax(logits, dim=1)
        return sample_output[-1].item()

    def run(self, input_ids_list, prefill=True):
        """input_ids_list: list of [seq] tensors (different lengths allowed)."""
        tokens = []
        for idx, ids in enumerate(input_ids_list):
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.int32)
            tokens.append(self._run_single(ids, sample_idx=idx, prefill=prefill))
        return torch.tensor(tokens, dtype=torch.int32, device="cuda")

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
        "The University of Washington is located in",
        "How is the weather in Seattle right now?",
    ]
    engine = Engine()
    output_text = engine.generate_batched(prompts, rounds=20)
    print("Generated Text:", output_text)
