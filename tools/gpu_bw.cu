// GPU memory-bandwidth probe (counterpart of stream_bw.cpp): cudaMemcpy device-to-device, a STREAM triad kernel,
// and a random-gather kernel y[i] = x[idx[i]] that mimics the irregular x[src] reads of the PageRank SpMV.
// Reports the best of --reps runs in GB/s (bytes moved as seen by the program, not DRAM sectors).
// Usage: gpu_bw [--mb 2048] [--reps 20]
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

__global__ void triad(double* a, const double* b, const double* c, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    a[i] = b[i] + 3.0 * c[i];
}

__global__ void gather(double* y, const double* x, const int* idx, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    y[i] = x[idx[i]];
}

int main(int argc, char** argv) {
  long long mb = 2048;
  int reps = 20;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--mb")) mb = std::atoll(argv[i + 1]);
    if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
  }
  const long long n = mb * 1024 * 1024 / 8;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  double *a, *b, *c;
  int* idx;
  CK(cudaMalloc(&a, n * 8));
  CK(cudaMalloc(&b, n * 8));
  CK(cudaMalloc(&c, n * 8));
  CK(cudaMalloc(&idx, n * 4));
  CK(cudaMemset(a, 0, n * 8));
  CK(cudaMemset(b, 0, n * 8));
  CK(cudaMemset(c, 0, n * 8));
  {
    std::vector<int> h(n);
    std::mt19937_64 rng(1);
    std::uniform_int_distribution<long long> U(0, n - 1);
    for (long long i = 0; i < n; ++i) h[i] = int(U(rng));
    CK(cudaMemcpy(idx, h.data(), n * 4, cudaMemcpyHostToDevice));
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const int blocks = prop.multiProcessorCount * 8, threads = 256;
  float best_copy = 1e30f, best_triad = 1e30f, best_gather = 1e30f, ms;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0));
    CK(cudaMemcpy(a, b, n * 8, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    best_copy = std::min(best_copy, ms);
    CK(cudaEventRecord(e0));
    triad<<<blocks, threads>>>(a, b, c, n);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    best_triad = std::min(best_triad, ms);
    CK(cudaEventRecord(e0));
    gather<<<blocks, threads>>>(a, b, idx, n);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    best_gather = std::min(best_gather, ms);
  }
  CK(cudaGetLastError());
  // copy: read+write 16 B/elem; triad: 24 B/elem; gather: idx 4 B + x 8 B + y 8 B = 20 B/elem.
  std::printf("device,mib_per_array,copy_gbs,triad_gbs,random_gather_gbs\n%s,%lld,%.1f,%.1f,%.1f\n", prop.name, mb,
              16.0 * n / (best_copy * 1e-3) / 1e9, 24.0 * n / (best_triad * 1e-3) / 1e9,
              20.0 * n / (best_gather * 1e-3) / 1e9);
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(c));
  CK(cudaFree(idx));
  return 0;
}
