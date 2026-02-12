import time
import torch
from transformers import AutoTokenizer
import sys

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

        # Load the tokenizer for text processing
        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weight_path, local_files_only=True
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)

        self.kv_cache = {}

    def run(self, input_ids: list[int], prefill: bool = True):
        # Unpack for convenient reference
        embedding = self.weights["embedding"]  # token_ids -> embedding
        # Attention weights, they are per layer. We'll index them by layer number
        layernormAttn_weight = self.weights["layernormAttn_weight"]
        self_attn_q_proj_weight = self.weights["self_attn_q_proj_weight"]
        self_attn_k_proj_weight = self.weights["self_attn_k_proj_weight"]
        self_attn_v_proj_weight = self.weights["self_attn_v_proj_weight"]
        o_proj_weight = self.weights["o_proj_weight"]
        # FFN weights
        layernormFFN_weight = self.weights["layernormFFN_weight"]
        up_proj_weight = self.weights["up_proj_weight"]
        gate_proj_weight = self.weights["gate_proj_weight"]
        down_proj_weight = self.weights["down_proj_weight"]
        # Final layer normalization
        model_layernorm_weight = self.weights["model_layernorm_weight"]
        # Final vocabulary projection
        lm_head_weight = self.weights["lm_head_weight"]

        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        hidden_state: torch.Tensor = embedding[input_tensor]
        for current_layer in range(self.layers):
            rms = torch.sqrt(torch.mean(hidden_state**2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x: torch.Tensor = normalized_x * layernormAttn_weight[current_layer]
            q = x.matmul(self_attn_q_proj_weight[current_layer].t())
            k = x.matmul(self_attn_k_proj_weight[current_layer].t())
            v = x.matmul(self_attn_v_proj_weight[current_layer].t())

            if prefill or current_layer not in self.kv_cache:
                offset = 0
            else:
                offset = self.kv_cache[current_layer]["k"].shape[1]
            apply_rope(q, output=q, head_dim=self.head_dim, offset=offset)
            apply_rope(k, output=k, head_dim=self.head_dim, offset=offset)

            sub_q = q.view(-1, self.num_qo_heads, self.head_dim)
            sub_k = k.view(-1, self.num_kv_heads, self.head_dim)
            sub_v = v.view(-1, self.num_kv_heads, self.head_dim)

            if not prefill and current_layer in self.kv_cache:
                sub_k = torch.cat([self.kv_cache[current_layer]["k"], sub_k], dim=0)
                sub_v = torch.cat([self.kv_cache[current_layer]["v"], sub_v], dim=0)
            self.kv_cache[current_layer] = {"k": sub_k, "v": sub_v}

            scale: float = 1.0 / (self.head_dim**0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            n_q = sub_q.shape[0]
            n_k = sub_k.shape[0]

            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)

            sub_q_t = sub_q.permute(1, 0, 2)
            sub_k_t = sub_k.permute(1, 0, 2)
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

            if prefill:
                causal_mask = torch.tril(
                    torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)
                )
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            v_t = sub_v.permute(1, 0, 2)
            attn_output = torch.matmul(attn_weights, v_t)
            attn_output = attn_output.permute(1, 0, 2)
            attn_output = attn_output.reshape(-1, self.num_qo_heads * self.head_dim)
            o_proj_residual = (
                attn_output.matmul(o_proj_weight[current_layer].t()) + hidden_state
            )

            # --- Feed-Forward Network (FFN) ---
            rms = torch.sqrt(
                torch.mean(o_proj_residual**2, dim=-1, keepdim=True) + 1e-5
            )
            normalized_x = o_proj_residual / rms
            layernormFFN_output = (
                normalized_x.to(torch.float16) * layernormFFN_weight[current_layer]
            )
            up_proj_output = layernormFFN_output.matmul(
                up_proj_weight[current_layer].t()
            )
            gate_proj_output = layernormFFN_output.matmul(
                gate_proj_weight[current_layer].t()
            )
            activation_output = up_proj_output * torch.nn.functional.silu(
                gate_proj_output
            )
            down_proj_output = activation_output.matmul(
                down_proj_weight[current_layer].t()
            )
            hidden_state = down_proj_output + o_proj_residual

        # --- Final Layer Normalization, Projection to Vocabulary, Sampling ---
        rms = torch.sqrt(torch.mean(hidden_state**2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = normalized_x.to(torch.float16) * model_layernorm_weight
        logits = model_output.matmul(lm_head_weight.t())

        sample_output = torch.argmax(logits, dim=1)
        return sample_output[-1].item()

    def generate(self, input_string: str, rounds: int = 20):
        input_ids: list[int] = self.tokenizer.encode(input_string)
        # print("Token IDs:", input_ids)
        output_ids = self.generate_from_ids(input_ids, rounds)
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    def generate_from_ids(self, input_ids: list[int], rounds: int = 20):
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(rounds - 1):
            # print(f"Round {round}")
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)

        return output_ids

    def timed_generate(
        self, input_ids: list[int], num_rounds: int, time_rounds: list[int]
    ):

        output_ids = input_ids
        start_event = torch.cuda.Event(enable_timing=True)
        end_events: dict[int, torch.cuda.Event] = {}
        for time_round in time_rounds:
            end_events[time_round] = torch.cuda.Event(enable_timing=True)

        end_times = {}
        torch.cuda.synchronize()
        start_event.record()
        start_time = time.perf_counter()

        # First round
        new_token = self.run(output_ids)
        output_ids.append(new_token)

        for round in range(2, num_rounds + 1):
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)
            if round in time_rounds:
                end_events[round].record()
                end_times[round] = time.perf_counter()

        torch.cuda.synchronize()
        timings_cuda = {
            round: start_event.elapsed_time(end_events[round]) for round in time_rounds
        }
        timings_time = {
            round: (end_times[round] - start_time) * 1000 for round in time_rounds
        }
        return timings_cuda, timings_time


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=20)
    print("Generated Text:", output_text)
