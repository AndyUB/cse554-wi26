#include <cuda_runtime.h>

#define BLOCKSIZE 256

__global__ void silu_kernel(float *input, float *output, size_t num) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    if (index < num) {
        output[index] = input[index] / (1.0f + expf(-input[index]));
    }
}

void silu(float *input, float *output, int n) {
    dim3 num_block((n + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);
    silu_kernel<<<num_block, num_threads>>>(input, output, n);
}
