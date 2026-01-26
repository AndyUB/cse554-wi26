import torch
from silu_triton_kernel import silu_triton


SHAPE = (8192, 8192)


def test_silu_triton_correctness():
    torch.manual_seed(0)
    x = torch.randn(*SHAPE, device="cuda")

    output_torch = torch.nn.functional.silu(x)
    output_triton = silu_triton(x)

    assert torch.allclose(output_torch, output_triton)
    print("Silu Triton correctness test passed!")


def time_silu_triton(block_size: int, num_warmups: int = 5, num_iters: int = 20):
    torch.manual_seed(0)
    x = torch.randn(*SHAPE, device="cuda")

    for _ in range(num_warmups):
        silu_triton(x, block_size=block_size)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(num_iters):
        silu_triton(x, block_size=block_size)
    end_ev.record()
    torch.cuda.synchronize()

    total_elapse = start_ev.elapsed_time(end_ev) / 1000  # in seconds
    elapse = total_elapse / num_iters
    bandwidth = x.numel() * x.element_size() * 2 / elapse / 1e9  # in GB/s
    print(
        f"Block size: {block_size}, Elapse: {elapse * 1000:.6f} ms, Bandwidth: {bandwidth:.2f} GB/s"
    )
    return elapse, bandwidth


def test_silu_triton_performance():
    block_sizes = [64, 128, 256, 512, 1024]
    perf_map = {}
    for block_size in block_sizes:
        perf_map[block_size] = time_silu_triton(block_size)

    best_block_size = max(perf_map, key=lambda k: perf_map[k][1])
    best_elapse, best_bandwidth = perf_map[best_block_size]
    print(
        f"Best block size: {best_block_size}, Elapse: {best_elapse * 1000:.6f} ms, "
        f"Bandwidth: {best_bandwidth:.2f} GB/s"
    )


if __name__ == "__main__":
    test_silu_triton_correctness()
    test_silu_triton_performance()
