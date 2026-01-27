#include <cuda_runtime.h>
#include "rms_norm_matrix.h"
#include <stdio.h>

int main() {
    int rows = 8192;
    int cols = 8192;
    int N = rows * cols;
    float epsilon = 1e-6f;
    float *d_input, *d_weight, *d_output;
    float *h_input, *h_weight, *h_output;

    h_input = (float*)malloc(N * sizeof(float));
    h_weight = (float*)malloc(cols * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_weight, cols * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_input[i * cols + j] = static_cast<float>(i + j);
        }
    }
    for (int j = 0; j < cols; j++) {
        h_weight[j] = static_cast<float>(j + 1) / cols;
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, cols * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm_matrix(d_input, d_weight, d_output, rows, cols, epsilon);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = h_input[i * cols + j];
            sum += val * val;
        }
        float rms = sqrtf(sum / cols + epsilon);
        float inv_rms = 1.0f / rms;
        for (int j = 0; j < cols; j++) {
            float expected = h_input[i * cols + j] * inv_rms * h_weight[j];
            float diff = fabs(h_output[i * cols + j] - expected);
            if (diff > 1e-6) {
                printf("Mismatch at row %d, col %d: expected %f, got %f\n", i, j, expected, h_output[i * cols + j]);
                correct = false;
            }
        }
    }
    printf("Correctness check: %s\n", correct ? "PASSED" : "FAILED");

    free(h_input);
    free(h_weight);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    return 0;

}