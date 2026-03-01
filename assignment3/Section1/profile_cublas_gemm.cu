#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupIterations = 100;
constexpr int kProfileIterations = 100;
constexpr std::size_t kL2FlushSizeBytes = 50ULL * 1024ULL * 1024ULL;

struct GemmShape {
  int m;
  int n;
  int k;
};

/**
 * Computes the total floating-point operations for one GEMM.
 */
double ComputeFlops(const GemmShape &shape) {
  return 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
         static_cast<double>(shape.k);
}

/**
 * Allocates and initializes FP16 matrices used by GEMM.
 */
void AllocateGemmBuffers(const GemmShape &shape, half **d_a, half **d_b, half **d_c) {
  const std::size_t size_a = static_cast<std::size_t>(shape.m) * shape.k * sizeof(half);
  const std::size_t size_b = static_cast<std::size_t>(shape.k) * shape.n * sizeof(half);
  const std::size_t size_c = static_cast<std::size_t>(shape.m) * shape.n * sizeof(half);

  cudaMalloc(reinterpret_cast<void **>(d_a), size_a);
  cudaMalloc(reinterpret_cast<void **>(d_b), size_b);
  cudaMalloc(reinterpret_cast<void **>(d_c), size_c);

  cudaMemset(*d_a, 0, size_a);
  cudaMemset(*d_b, 0, size_b);
  cudaMemset(*d_c, 0, size_c);
}

/**
 * Profiles one GEMM shape using cublasGemmEx with warm-up and L2 flush.
 */
std::pair<float, double> ProfileShape(cublasHandle_t handle, const GemmShape &shape,
                                      int warmup_iterations, int profile_iterations,
                                      int *l2_flush_buffer) {
  half *d_a = nullptr;
  half *d_b = nullptr;
  half *d_c = nullptr;
  AllocateGemmBuffers(shape, &d_a, &d_b, &d_c);

  const int lda = shape.k;
  const int ldb = shape.n;
  const int ldc = shape.n;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < warmup_iterations; ++i) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 shape.n, shape.m, shape.k,
                 &alpha,
                 d_b, CUDA_R_16F, ldb,
                 d_a, CUDA_R_16F, lda,
                 &beta,
                 d_c, CUDA_R_16F, ldc,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }

  cudaDeviceSynchronize();

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_ms = 0.0f;
  for (int i = 0; i < profile_iterations; ++i) {
    cudaMemset(l2_flush_buffer, 0, kL2FlushSizeBytes);
    cudaEventRecord(start);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 shape.n, shape.m, shape.k,
                 &alpha,
                 d_b, CUDA_R_16F, ldb,
                 d_a, CUDA_R_16F, lda,
                 &beta,
                 d_c, CUDA_R_16F, ldc,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const float avg_ms = total_ms / static_cast<float>(profile_iterations);
  const double tflops = ComputeFlops(shape) / (static_cast<double>(avg_ms) * 1.0e-3) / 1.0e12;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return {avg_ms, tflops};
}

}  // namespace

int main(int argc, char **argv) {
  std::string output_csv = "cublas_gemm_perf.csv";
  if (argc > 1) {
    output_csv = argv[1];
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  int *l2_flush_buffer = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&l2_flush_buffer), kL2FlushSizeBytes);

  const std::array<std::pair<int, int>, 5> nk_shapes = {
      std::pair<int, int>{512, 512},
      std::pair<int, int>{4096, 4096},
      std::pair<int, int>{14336, 4096},
      std::pair<int, int>{4096, 1024},
      std::pair<int, int>{1024, 4096},
  };

  std::ofstream out(output_csv);
  out << "M,N,K,library,time_ms,tflops\n";

  for (const auto &nk : nk_shapes) {
    for (int m = 128; m <= 2048; m += 128) {
      GemmShape shape{m, nk.first, nk.second};
      const auto [avg_ms, tflops] = ProfileShape(
          handle, shape, kWarmupIterations, kProfileIterations, l2_flush_buffer);
      std::cout << "M=" << shape.m << " N=" << shape.n << " K=" << shape.k
                << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms
                << " tflops=" << std::setprecision(3) << tflops << '\n';
      out << shape.m << ',' << shape.n << ',' << shape.k << ",cublas,"
          << std::fixed << std::setprecision(6) << avg_ms << ','
          << std::setprecision(6) << tflops << '\n';
    }
  }

  out.close();
  cudaFree(l2_flush_buffer);
  cublasDestroy(handle);
  return 0;
}
