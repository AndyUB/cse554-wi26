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
        self.head_dim = 64  # Dimensionality of each attention head
        self.num_qo_heads = 32  # Total number of query/output heads
        self.num_kv_heads = 8  # Total number of key/value heads
        self.layers = 16  # Number of transformer layers

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

    def _apply_rope_batched(self, x: torch.Tensor, offset: int) -> None:
        for batch_idx in range(x.shape[0]):
            apply_rope(
                x[batch_idx], output=x[batch_idx], head_dim=self.head_dim, offset=offset
            )

    def run(self, input_ids, prefill=True):
        if not torch.is_tensor(input_ids):
            input_tensor = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        else:
            input_tensor = input_ids.to(device="cuda", dtype=torch.int32)

        hidden_state = self.weights["embedding"][input_tensor]  # [B, T, H]

        for current_layer in range(self.layers):
            normalized_x = self._rms_norm(hidden_state)
            x = (
                normalized_x.to(torch.float16)
                * self.weights["layernormAttn_weight"][current_layer]
            )

            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            if prefill or current_layer not in self.kv_cache:
                offset = 0
            else:
                offset = self.kv_cache[current_layer]["k"].shape[1]

            self._apply_rope_batched(q, offset=offset)
            self._apply_rope_batched(k, offset=offset)

            bsz = q.shape[0]
            sub_q = q.view(bsz, -1, self.num_qo_heads, self.head_dim)
            sub_k_new = k.view(bsz, -1, self.num_kv_heads, self.head_dim)
            sub_v_new = v.view(bsz, -1, self.num_kv_heads, self.head_dim)

            if prefill or current_layer not in self.kv_cache:
                sub_k_total = sub_k_new
                sub_v_total = sub_v_new
            else:
                sub_k_total = torch.cat(
                    [self.kv_cache[current_layer]["k"], sub_k_new], dim=1
                )
                sub_v_total = torch.cat(
                    [self.kv_cache[current_layer]["v"], sub_v_new], dim=1
                )

            self.kv_cache[current_layer] = {"k": sub_k_total, "v": sub_v_total}

            group_size = self.num_qo_heads // self.num_kv_heads
            sub_k = sub_k_total.repeat_interleave(group_size, dim=2)
            sub_v = sub_v_total.repeat_interleave(group_size, dim=2)

            # [B, H, Tq, D], [B, H, Tk, D]
            sub_q_t = sub_q.permute(0, 2, 1, 3)
            sub_k_t = sub_k.permute(0, 2, 1, 3)
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * (
                1.0 / (self.head_dim**0.5)
            )

            if prefill:
                t_q = sub_q.shape[1]
                t_k = sub_k.shape[1]
                causal_mask = torch.tril(
                    torch.ones(t_q, t_k, dtype=torch.bool, device=scores.device)
                )
                scores = scores.masked_fill(
                    ~causal_mask.view(1, 1, t_q, t_k),
                    float("-inf"),
                )

            attn_weights = torch.softmax(scores, dim=-1)
            v_t = sub_v.permute(0, 2, 1, 3)
            attn_output = torch.matmul(attn_weights, v_t)
            attn_output = attn_output.permute(0, 2, 1, 3)

            attn_output = attn_output.reshape(
                bsz, -1, self.num_qo_heads * self.head_dim
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

        normalized_x = self._rms_norm(hidden_state)
        model_output = (
            normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        )
        logits = model_output.matmul(self.weights["lm_head_weight"].t())

        sample_output = torch.argmax(logits[:, -1, :], dim=-1)
        return sample_output.view(-1, 1)

    @torch.inference_mode()
    def generate_batched(self, input_strings: List[str], rounds=20):
        self.reset_cache()
        input_ids = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=False,
        ).input_ids
        output_ids = input_ids.to("cuda")

        new_token = self.run(output_ids, prefill=True)
        output_ids = torch.cat((output_ids, new_token), dim=1)

        for _ in range(rounds - 1):
            new_token = self.run(output_ids[:, -1:], prefill=False)
            output_ids = torch.cat((output_ids, new_token), dim=1)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(8)]
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)
