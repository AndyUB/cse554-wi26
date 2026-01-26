import torch
from torch.profiler import profile, record_function, ProfilerActivity


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * (1.0 / (1.0 + torch.exp(-x)))


if __name__ == "__main__":
    num_warmups = 5
    num_iters = 20

    t = torch.randn(8192, 8192, device="cuda")
    for _ in range(num_warmups):
        silu(t)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        start.record()
        with record_function("silu"):
            for i in range(num_iters):
                silu(t)
        end.record()
        torch.cuda.synchronize()
        total_elapse = start.elapsed_time(end) / 1000  # in seconds

    prof.export_chrome_trace("torch_silu.json")
    elapse = total_elapse / num_iters
    bandwidth = t.element_size() * t.numel() * 2 / elapse / 1e9  # in GB/s
    print(f"Time per iteration: {elapse} seconds")
    print(f"Bandwidth: {bandwidth} GB/s")
