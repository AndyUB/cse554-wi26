from continous_engine import Engine, Request
import torch

class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len
        
class Scheduler:
    def __init__(self, engine: Engine, req_batch_size: int):
        self.engine = engine
        self.req_batch_size = req_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req

    def get_req_batch_size(self) -> int:
        return len(self.decode_req) + len(self.scheduled_prefill_req)

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        while self.pending_input_req and self.get_req_batch_size() < self.req_batch_size:
            input_req = self.pending_input_req.pop(0)
            token_ids = self.engine.tokenizer.encode(
                input_req.input_str, return_tensors="pt"
            ).squeeze(0)
            req = Request(
                req_id=self.unique_req_id,
                prompt_ids=token_ids,
                target_len=input_req.output_len,
            )
            self.unique_req_id += 1
            self.scheduled_prefill_req.append(req)

        # Build the list of requests to send to the engine
        # Decode requests come first so the engine can correctly split them
        request_list_total = self.decode_req + self.scheduled_prefill_req
        decode_num = len(self.decode_req)

        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        for i, req in enumerate(request_list_total):
            new_token = new_tokens[i].unsqueeze(0)
            req.output_token_ids = torch.cat([req.output_token_ids, new_token])

        # Check which decode requests have finished and remove from the queue
        ongoing_decode: list[Request] = []
        for req in self.decode_req:
            tokens_generated = req.current_length - req.prompt_length
            if tokens_generated >= req.output_length:
                # Free KV cache pages and mark as completed
                self.engine.kv_cache_map[req.request_id].release()
                del self.engine.kv_cache_map[req.request_id]
                self.completed.append(req)
            else:
                ongoing_decode.append(req)
        self.decode_req = ongoing_decode

        # Move scheduled prefill requests into decode queue
        self.decode_req.extend(self.scheduled_prefill_req)
        self.scheduled_prefill_req = []
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
