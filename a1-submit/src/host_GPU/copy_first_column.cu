#include "copy_first_column.h"
#include <cuda_runtime.h>

void copy_first_column(float *h_A, float * h_col_pin, float *d_A, int rows, int cols) {
    cudaMemcpy2D(d_A, sizeof(float), h_A, cols * sizeof(float), sizeof(float), rows, cudaMemcpyHostToDevice);
    for (int i = 0; i < rows; ++i) h_col_pin[i] = h_A[i * cols];
    cudaMemcpyAsync(d_A, h_col_pin, rows * sizeof(float), cudaMemcpyHostToDevice);
}