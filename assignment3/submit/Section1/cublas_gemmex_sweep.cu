#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <string>

#define CHECK_CUDA(call) do {                                \
  cudaError_t err = (call);                                  \
  if (err != cudaSuccess) {                                  \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)   \
              << " at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                  \
    std::exit(1);                                             \
  }                                                          \
} while(0)

#define CHECK_CUBLAS(call) do {                              \
  cublasStatus_t st = (call);                                \
  if (st != CUBLAS_STATUS_SUCCESS) {                         \
    std::cerr << "cuBLAS error: " << (int)st                 \
              << " at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                  \
    std::exit(1);                                             \
  }                                                          \
} while(0)

static void fill_device_buffer(void* ptr, size_t bytes, unsigned char pattern) {
  CHECK_CUDA(cudaMemset(ptr, pattern, bytes));
}

int main() {
  const int M_start = 128, M_end = 2048, M_step = 128;

  struct NK { int N; int K; const char* name; };
  std::vector<NK> nk_cases = {
    {  512,   512,   "(512,512)"   },
    { 4096,  4096,   "(4096,4096)" },
    {14336,  4096,   "(14336,4096)"},
    { 4096,  1024,   "(4096,1024)" },
    { 1024,  4096,   "(1024,4096)" },
  };

  const int warm_up_count = 50;
  const int profile_count = 100;

  const size_t L2_flush_bytes = 50ull * 1024ull * 1024ull;

  const cudaDataType_t Atype = CUDA_R_16F;
  const cudaDataType_t Btype = CUDA_R_16F;
  const cudaDataType_t Ctype = CUDA_R_16F;
  const cudaDataType_t computeType = CUDA_R_32F;

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  void* clear_l2_buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&clear_l2_buffer, L2_flush_bytes));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::cout << "library,N,K,batch_size,avg_ms,tflops\n";
  std::cout << std::fixed << std::setprecision(6);

  for (const auto& c : nk_cases) {
    const int N = c.N;
    const int K = c.K;

    const int Mmax = M_end;

    const int lda = Mmax;
    const int ldb = K;
    const int ldc = Mmax;

    size_t bytesA = (size_t)lda * (size_t)K * sizeof(__half);
    size_t bytesB = (size_t)ldb * (size_t)N * sizeof(__half);
    size_t bytesC = (size_t)ldc * (size_t)N * sizeof(__half);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    fill_device_buffer(dA, bytesA, 0x3c);
    fill_device_buffer(dB, bytesB, 0x2a);
    fill_device_buffer(dC, bytesC, 0x00);

    for (int M = M_start; M <= M_end; M += M_step) {
      for (int i = 0; i < warm_up_count; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
          handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          dA, Atype, lda,
          dB, Btype, ldb,
          &beta,
          dC, Ctype, ldc,
          computeType,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      float total_ms = 0.0f;
      for (int i = 0; i < profile_count; ++i) {
        CHECK_CUDA(cudaMemset(clear_l2_buffer, 0, L2_flush_bytes));

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasGemmEx(
          handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          dA, Atype, lda,
          dB, Btype, ldb,
          &beta,
          dC, Ctype, ldc,
          computeType,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
      }

      float avg_ms = total_ms / profile_count;

      double flops = 2.0 * (double)M * (double)N * (double)K;
      double tflops = (flops / 1e12) / (avg_ms / 1e3);

      std::cout
        << "cublas" << ","
        << N << "," << K << "," << M << ","
        << avg_ms << ","
        << tflops
        << "\n";
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(clear_l2_buffer));
  CHECK_CUBLAS(cublasDestroy(handle));

  return 0;
}