import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * (1.0 / (1.0 + torch.exp(-x)))


if __name__ == "__main__":
    t = torch.randn(8192, 8192, device="cpu")
    t = t.to("cuda")
    torch.cuda.synchronize()

    num_iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(num_iters):
        silu(t)
    end.record()
    torch.cuda.synchronize()

    total_elapse = start.elapsed_time(end) / 1000  # in seconds
    elapse = total_elapse / num_iters
    bandwidth = t.element_size() * t.numel() * 2 / elapse / 1e9  # in GB/s
    print(f"Time per iteration: {elapse} seconds")
    print(f"Bandwidth: {bandwidth} GB/s")
