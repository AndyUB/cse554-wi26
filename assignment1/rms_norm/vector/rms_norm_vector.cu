#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define COLS (1024 * 1024)
#define THREADS_PER_BLOCK 512

__global__ void rms_norm_vector_kernel(float *input, float *weight, float *output, float epsilon, float *global_sum, float *inv_rms) {
   cg::grid_group grid = cg::this_grid();

    __shared__ float sdata[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    for (int j = idx; j < COLS; j += stride) {
        float val = input[j];
        local_sum += val * val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }

    grid.sync();

    if (tid == 0 && blockIdx.x == 0) {
        float mean_square = (*global_sum) / COLS;
        *inv_rms = rsqrtf(mean_square + epsilon);
    }

    grid.sync();

    float inv_rms_local = *inv_rms;

    for (int j = idx; j < COLS; j += stride) {
        float val = input[j];
        output[j] = val * inv_rms_local * weight[j];
    }
}


void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    float *d_global_sum, *d_inv_rms;
    cudaMalloc((void**)&d_global_sum, sizeof(float));
    cudaMalloc((void**)&d_inv_rms, sizeof(float));
    cudaMemset(d_global_sum, 0, sizeof(float));

    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, rms_norm_vector_kernel, THREADS_PER_BLOCK, 0
    );
    int blocks = sm_count * blocks_per_sm;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(blocks);
    void* args[] = {
        (void*)&input,
        (void*)&weight,
        (void*)&output,
        (void*)&epsilon,
        (void*)&d_global_sum,
        (void*)&d_inv_rms
    };

    cudaLaunchCooperativeKernel((void*)rms_norm_vector_kernel, grid, block, args);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_global_sum);
    cudaFree(d_inv_rms);
}
