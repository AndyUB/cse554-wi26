#include <cuda_runtime.h>
#include <stdio.h>

#define COLS 8192
#define THREADS_PER_BLOCK 256
constexpr int read_iter = COLS / THREADS_PER_BLOCK;

__global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, float epsilon) {
    __shared__ float sdata[THREADS_PER_BLOCK];
    __shared__ float inv_rms;

    int tid = threadIdx.x;
    int row = blockIdx.x;

    float *input_row = input + row * COLS;
    float *output_row = output + row * COLS;

    float local_sum = 0.0f;
    int idx = tid;

    for (int j = 0; j < read_iter; j++) {
        float val = input_row[idx];
        local_sum += val * val;
        idx += THREADS_PER_BLOCK;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_square = sdata[0] / COLS;
        inv_rms = rsqrtf(mean_square + epsilon);
    }
    __syncthreads();

    float inv_rms_local = inv_rms;
    idx = tid;
    for (int j = 0; j < read_iter; j++) {
        float val = input_row[idx];
        output_row[idx] = val * inv_rms_local * weight[idx];
        idx += THREADS_PER_BLOCK;
    }
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(rows);
    rms_norm_matrix_kernel<<<grid, block>>>(input, weight, output, epsilon);
}
