#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>


int main() {
    int cols = 1024 * 1024;
    int N = cols;
    float epsilon = 1e-6f;
    float *d_input, *d_weight, *d_output;
    float *h_input, *h_weight, *h_output;

    h_input = (float*)malloc(N * sizeof(float));
    h_weight = (float*)malloc(cols * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_weight, cols * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    for (int j = 0; j < cols; j++) {
        h_input[j] = static_cast<float>(j);
        h_weight[j] = static_cast<float>(j + 1) / cols;
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, cols * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm_vector(d_input, d_weight, d_output, cols, epsilon);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float val = h_input[j];
        sum += val * val;
    }
    float rms = sqrtf(sum / cols + epsilon);
    float inv_rms = 1.0f / rms;
    for (int j = 0; j < cols; j++) {
        float expected = h_input[j] * inv_rms * h_weight[j];
        float diff = fabs(h_output[j] - expected);
        if (diff > 1e-3 && correct) {
            printf("Mismatch at col %d: expected %f, got %f\n", j, expected, h_output[j]);
            correct = false;
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