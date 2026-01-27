#include <cuda_runtime.h>
#include "silu.h"
#include <iostream>
#include <cmath>


int main() {
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

    silu(d_inp, d_out, num);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

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