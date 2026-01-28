#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
    constexpr int ROWS = 8192;
    constexpr int COLS = 65536;
    
    float * h_A = (float *)malloc(ROWS * COLS * sizeof(float));
    float * h_col_pin;
    cudaMallocHost(&h_col_pin, ROWS * sizeof(float)); // pinned memory
    // float * h_A;
    // cudaMallocHost(&h_A, ROWS * COLS * sizeof(float)); // pinned memory

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            h_A[i * COLS + j] = static_cast<float>(i * COLS + j);
        }
    }

    float * d_col;
    cudaMalloc(&d_col, ROWS * sizeof(float));

    // check correctness
    copy_first_column(h_A, h_col_pin, d_col, ROWS, COLS);
    cudaDeviceSynchronize();
    float * h_col = (float *)malloc(ROWS * sizeof(float));
    cudaMemcpy(h_col, d_col, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // verify results
    for (int i = 0; i < ROWS; ++i) {
        if (h_col[i] != static_cast<float>(i * COLS)) {
            printf("Error: h_col[%d] = %f, expected %f\n", i, h_col[i], static_cast<float>(i * COLS));
        }
    }

    // warmup
    copy_first_column(h_A, h_col_pin, d_col, ROWS, COLS);
    cudaDeviceSynchronize();

    // time the copy
    const int iters = 1000;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; ++i) {
        copy_first_column(h_A, h_col_pin, d_col, ROWS, COLS);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);
    printf("Average time per copy: %f ms\n", ms / iters);

    cudaFree(d_col);
    cudaFreeHost(h_A);
    free(h_col);

    return 0;
}