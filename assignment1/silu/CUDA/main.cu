#include <cuda_runtime.h>
#include "silu.h"
#include <iostream>
#include <cmath>


int main() {
    int num_warmups = 5;
    int num_iters = 20;

    size_t num = 8192 * 8192;

    float * host_inp = new float[num];
    float * host_out = new float[num];
    // Initialize host arrays
    for (int i = 0; i < num; i++) {
        host_inp[i] = static_cast<float>(i);
        host_out[i] = 0.0f;
    }

    // Allocate memory on the device
    float *d_inp, *d_out;
    cudaMalloc((void**)&d_inp, num * sizeof(float));
    cudaMalloc((void**)&d_out, num * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_inp, host_inp, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, host_out, num * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < num_warmups; i++) {
        silu(d_inp, d_out, num);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        silu(d_inp, d_out, num);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float total_elapse = 0.0f;
    cudaEventElapsedTime(&total_elapse, start, stop);
    float elapse = total_elapse / num_iters;
    float bandwidth = (num * sizeof(float) * 2) / (elapse / 1000.0f) / (1e9); // GB/s
    std::cout << "Average time per iteration: " << elapse << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Copy result back to host
    cudaMemcpy(host_out, d_out, num * sizeof(float), cudaMemcpyDeviceToHost);

    float eps = 1e-6f;
    for (int i = 0; i < num; i++) {
        float expected = host_inp[i] / (1.0f + expf(-host_inp[i]));
        float diff = fabs(host_out[i] - expected);
        if (diff > eps) {
            std::cerr << "Error at index " << i << ": " << host_out[i] << std::endl;
            break;
        }
    }
    std::cout << "Correctness check passed." << std::endl;

    // Free device memory
    cudaFree(d_inp);
    cudaFree(d_out);
    delete[] host_inp;
    delete[] host_out;

    return 0; 
}