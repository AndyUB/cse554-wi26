from chunked_engine import Engine, Request
import torch

class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len
        
class Scheduler:
    def __init__(self, engine: Engine, token_batch_size: int):
        self.engine = engine
        self.token_batch_size = token_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
        self.pending_prefill:list[Request] = []
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req and not self.pending_prefill

    def get_token_batch_size(self) -> int:
        sum = 0
        for req in self.scheduled_prefill_req:
            sum += req.scheduling_length
        for req in self.decode_req:
            sum += 1
        return sum

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        
        while self.pending_input_req:
            pending_req = self.pending_input_req.pop(0)
            req_id = self.unique_req_id
            self.unique_req_id += 1
            prompt_ids = self.engine.tokenizer(
                pending_req.input_str, return_tensors="pt"
            ).input_ids[0]
            req = Request(
                req_id, prompt_ids, pending_req.output_len
            )
            self.pending_prefill.append(req)

        current_budget_used = self.get_token_batch_size()
        available_budget = self.token_batch_size - current_budget_used
        # Schedule prefill request and move it to the scheduled_prefill_req.
        # Limit the budget, and chunk the request as necessary.
        # If a request is chunked, keep it still in the pending prefill queue. Otherwise, remove from the queue.
        # Set tokens of the current chunk in the request scheduling_pf_tokens and set remaining_prefill_tokens properly
        remaining_pending: list[Request] = []
        for req in self.pending_prefill:
            if available_budget <= 0:
                remaining_pending.append(req)
                continue

            # Slice out the next chunk for this request
            already_processed = req.prompt_length - req.remaining_prefill_tokens
            tokens_this_chunk = min(req.remaining_prefill_tokens, available_budget)
            req.scheduling_pf_tokens = req.prompt_token_ids[
                already_processed : already_processed + tokens_this_chunk
            ]
            req.remaining_prefill_tokens -= tokens_this_chunk
            req.last_chunk = (req.remaining_prefill_tokens == 0)
            available_budget -= tokens_this_chunk

            self.scheduled_prefill_req.append(req)
            # If chunked, keep the request in pending_prefill for future iterations
            if not req.last_chunk:
                remaining_pending.append(req)

        self.pending_prefill = remaining_pending

        # Build the list of requests to send to the engine
        request_list_total = self.decode_req + self.scheduled_prefill_req
        decode_num = len(self.decode_req)

        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        # For prefill, only append if this cycle is the last chunk
        for i, req in enumerate(request_list_total):
            if i < decode_num or req.last_chunk:
                new_token = new_tokens[i].unsqueeze(0)
                req.output_token_ids = torch.cat([req.output_token_ids, new_token])

        # Check which decode requests have finished
        ongoing_decode: list[Request] = []
        for req in self.decode_req:
            tokens_generated = req.current_length - req.prompt_length
            if tokens_generated >= req.output_length:
                self.engine.kv_cache_map[req.request_id].release()
                del self.engine.kv_cache_map[req.request_id]
                self.completed.append(req)
            else:
                ongoing_decode.append(req)
        self.decode_req = ongoing_decode

        # Move scheduled prefill requests into decode queue
        for req in self.scheduled_prefill_req:
            if req.last_chunk:
                self.decode_req.append(req)
        self.scheduled_prefill_req = []
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
