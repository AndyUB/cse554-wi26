// memcpy_sweep.cu
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(e));                                  \
      std::exit(1);                                                        \
    }                                                                       \
  } while (0)

static double time_memcpy_h2d(void* d, const void* h, size_t bytes, int iters) {
  // Warmup
  CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Time iters copies
  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // total ms for iters

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double seconds_total = static_cast<double>(ms) * 1e-3;
  return seconds_total / static_cast<double>(iters); // seconds per copy
}

static double time_memcpy_d2h(void* h, const void* d, size_t bytes, int iters) {
  // Warmup
  CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double seconds_total = static_cast<double>(ms) * 1e-3;
  return seconds_total / static_cast<double>(iters); // seconds per copy
}

static int pick_iters(size_t bytes) {
  // Keep runtime reasonable while still averaging enough for tiny sizes.
  if (bytes <= 1024) return 200000;
  if (bytes <= (1u << 12)) return 100000;
  if (bytes <= (1u << 14)) return 50000;
  if (bytes <= (1u << 16)) return 20000;
  if (bytes <= (1u << 18)) return 5000;
  return 2000; // up to 1 MiB
}

int main() {
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Device: " << prop.name << "\n\n";

  const char* csv_name = "memcpy_results.csv";
  std::ofstream csv(csv_name);
  if (!csv) {
    std::cerr << "Failed to open " << csv_name << " for writing\n";
    return 1;
  }

  // All data in one file: pageable + pinned
  csv << "bytes,iters,"
         "h2d_pageable_gbs,d2h_pageable_gbs,h2d_pageable_us,d2h_pageable_us,"
         "h2d_pinned_gbs,d2h_pinned_gbs,h2d_pinned_us,d2h_pinned_us\n";

  std::cout << std::left
            << std::setw(10) << "bytes"
            << std::setw(10) << "iters"
            << std::setw(16) << "H2Dpg(GB/s)"
            << std::setw(16) << "D2Hpg(GB/s)"
            << std::setw(16) << "H2Dpin(GB/s)"
            << std::setw(16) << "D2Hpin(GB/s)"
            << "\n";
  std::cout << std::string(74, '-') << "\n";

  for (int p = 0; p <= 20; ++p) {
    size_t bytes = size_t{1} << p;
    int iters = pick_iters(bytes);

    // -------- Pageable host buffers --------
    void* h_src_pg = std::malloc(bytes);
    void* h_dst_pg = std::malloc(bytes);
    if (!h_src_pg || !h_dst_pg) {
      std::fprintf(stderr, "Host malloc failed at %zu bytes\n", bytes);
      return 1;
    }
    std::memset(h_src_pg, 0xAB, bytes);
    std::memset(h_dst_pg, 0x00, bytes);

    // -------- Pinned host buffers --------
    void* h_src_pin = nullptr;
    void* h_dst_pin = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_src_pin, bytes)); // pinned
    CUDA_CHECK(cudaMallocHost(&h_dst_pin, bytes)); // pinned
    std::memset(h_src_pin, 0xAB, bytes);
    std::memset(h_dst_pin, 0x00, bytes);

    // Device buffer
    void* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));

    // -------- Measure pageable --------
    double h2d_pg_s = time_memcpy_h2d(d_buf, h_src_pg, bytes, iters);
    double d2h_pg_s = time_memcpy_d2h(h_dst_pg, d_buf, bytes, iters);

    double h2d_pg_gbs = (static_cast<double>(bytes) / 1e9) / h2d_pg_s;
    double d2h_pg_gbs = (static_cast<double>(bytes) / 1e9) / d2h_pg_s;
    double h2d_pg_us  = h2d_pg_s * 1e6;
    double d2h_pg_us  = d2h_pg_s * 1e6;

    // -------- Measure pinned --------
    double h2d_pin_s = time_memcpy_h2d(d_buf, h_src_pin, bytes, iters);
    double d2h_pin_s = time_memcpy_d2h(h_dst_pin, d_buf, bytes, iters);

    double h2d_pin_gbs = (static_cast<double>(bytes) / 1e9) / h2d_pin_s;
    double d2h_pin_gbs = (static_cast<double>(bytes) / 1e9) / d2h_pin_s;
    double h2d_pin_us  = h2d_pin_s * 1e6;
    double d2h_pin_us  = d2h_pin_s * 1e6;

    // Print a compact view
    std::cout << std::left
              << std::setw(10) << bytes
              << std::setw(10) << iters
              << std::setw(16) << std::fixed << std::setprecision(3) << h2d_pg_gbs
              << std::setw(16) << std::fixed << std::setprecision(3) << d2h_pg_gbs
              << std::setw(16) << std::fixed << std::setprecision(3) << h2d_pin_gbs
              << std::setw(16) << std::fixed << std::setprecision(3) << d2h_pin_gbs
              << "\n";

    // Write CSV row (all data)
    csv << bytes << ","
        << iters << ","
        << std::setprecision(10)
        << h2d_pg_gbs << "," << d2h_pg_gbs << "," << h2d_pg_us << "," << d2h_pg_us << ","
        << h2d_pin_gbs << "," << d2h_pin_gbs << "," << h2d_pin_us << "," << d2h_pin_us
        << "\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_src_pin));
    CUDA_CHECK(cudaFreeHost(h_dst_pin));
    std::free(h_src_pg);
    std::free(h_dst_pg);
  }

  csv.close();
  std::cout << "\nWrote CSV: " << csv_name << "\n";
  return 0;
}