// Single-GPU pull-style PageRank. The whole graph and all vectors stay resident in device memory;
// each iteration copies back only two scalars (dangling mass is consumed on device, residual for convergence).
// Kernels for the SpMV step:
//   thread   : one thread per row
//   warp     : one warp per row
//   hybrid   : rows bucketed by in-degree -> thread (deg<=16), warp (deg<=1024), block (deg>1024)
//   cusparse : cusparseSpMV on a CSR matrix of ones, followed by an update kernel
// Usage: pagerank_cuda --graph g.bin [--kernel all|thread|warp|hybrid|cusparse] [--iters 50] [--tol 0] [--reps 5]
//                      [--exp tag] [--out prefix] [--log prefix] [--header 1]
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cmath>
#include "../common/graph.hpp"

using namespace prg;

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) die(std::string(#x) + ": " + cudaGetErrorString(e_));                \
  } while (0)
#define CKS(x)                                                                                  \
  do {                                                                                          \
    cusparseStatus_t s_ = (x);                                                                  \
    if (s_ != CUSPARSE_STATUS_SUCCESS) die(std::string(#x) + ": cusparse status " + std::to_string(int(s_))); \
  } while (0)

constexpr int TPB = 256;
constexpr int SMALL_DEG = 16, LARGE_DEG = 1024;

struct Dev {
  int N;
  const int* off;
  const int* src;
  const int* outdeg;
  const double* r;
  double* rn;
  double* x;
  double* dang;   // device scalar: total dangling mass
  double* resid;  // device scalar: L1 residual
  double d;
};

template <typename T>
__device__ void block_sum_atomic(T v, T* out) {
  typedef cub::BlockReduce<T, TPB> BR;
  __shared__ typename BR::TempStorage tmp;
  T s = BR(tmp).Sum(v);
  if (threadIdx.x == 0 && s != T(0)) atomicAdd(out, s);
}

__global__ void k_contrib(Dev g) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  double dl = 0;
  if (u < g.N) {
    int od = g.outdeg[u];
    double ru = g.r[u];
    if (od) {
      g.x[u] = ru / od;
    } else {
      g.x[u] = 0;
      dl = ru;
    }
  }
  block_sum_atomic(dl, g.dang);
}

__device__ __forceinline__ double finish_row(const Dev& g, int v, double s) {
  double base = (1 - g.d) / g.N + g.d * (*g.dang) / g.N;
  double nv = base + g.d * s;
  g.rn[v] = nv;
  return fabs(nv - g.r[v]);
}

// One thread per row (rows optionally taken from a list).
__global__ void k_thread(Dev g, const int* rows, int nrows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double res = 0;
  if (i < nrows) {
    int v = rows ? rows[i] : i;
    double s = 0;
    for (int e = g.off[v]; e < g.off[v + 1]; ++e) s += g.x[g.src[e]];
    res = finish_row(g, v, s);
  }
  block_sum_atomic(res, g.resid);
}

// One warp per row.
__global__ void k_warp(Dev g, const int* rows, int nrows) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  double res = 0;
  if (w < nrows) {
    int v = rows ? rows[w] : w;
    double s = 0;
    for (int e = g.off[v] + lane; e < g.off[v + 1]; e += 32) s += g.x[g.src[e]];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) res = finish_row(g, v, s);
  }
  block_sum_atomic(res, g.resid);
}

// One block per row (for very high in-degree rows).
__global__ void k_block(Dev g, const int* rows) {
  int v = rows[blockIdx.x];
  double s = 0;
  for (int e = g.off[v] + threadIdx.x; e < g.off[v + 1]; e += blockDim.x) s += g.x[g.src[e]];
  typedef cub::BlockReduce<double, TPB> BR;
  __shared__ typename BR::TempStorage tmp;
  double tot = BR(tmp).Sum(s);
  if (threadIdx.x == 0) atomicAdd(g.resid, finish_row(g, v, tot));
}

// After cuSPARSE computed y = A x into rn: rn = base + d*y, residual.
__global__ void k_update(Dev g) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  double res = 0;
  if (v < g.N) res = finish_row(g, v, g.rn[v]);
  block_sum_atomic(res, g.resid);
}

static int blocks(long long n, int per = TPB) { return int((n + per - 1) / per); }

int main(int argc, char** argv) {
  Args a(argc, argv);
  std::string path = a.get("graph", "");
  if (path.empty()) die("--graph is required");
  std::string which = a.get("kernel", "all");
  const int max_iters = a.geti("iters", 50), reps = a.geti("reps", 1);
  const double tol = a.getd("tol", 0.0), damping = a.getd("damping", 0.85);
  const std::string exp = a.get("exp", "run"), outp = a.get("out", ""), logp = a.get("log", "");

  double t_load = now();
  Graph h = load_full(path);
  t_load = now() - t_load;
  const int64_t N = h.h.N, M = h.h.M;
  if (M >= (int64_t(1) << 31)) die("graph too large for 32-bit device offsets");
  std::vector<int> off32(N + 1);
  for (int64_t v = 0; v <= N; ++v) off32[v] = int(h.off[v]);

  // Degree buckets for the hybrid kernel.
  std::vector<int> small, mid, large;
  for (int64_t v = 0; v < N; ++v) {
    int64_t deg = h.off[v + 1] - h.off[v];
    (deg <= SMALL_DEG ? small : deg <= LARGE_DEG ? mid : large).push_back(int(v));
  }

  int *d_off, *d_src, *d_deg, *d_small, *d_mid, *d_large;
  double *d_r, *d_rn, *d_x, *d_scal, *d_ones = nullptr;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaFree(0));  // create the context outside the H2D timing
  double t_h2d = now();
  CK(cudaMalloc(&d_off, (N + 1) * sizeof(int)));
  CK(cudaMalloc(&d_src, M * sizeof(int)));
  CK(cudaMalloc(&d_deg, N * sizeof(int)));
  CK(cudaMalloc(&d_r, N * sizeof(double)));
  CK(cudaMalloc(&d_rn, N * sizeof(double)));
  CK(cudaMalloc(&d_x, N * sizeof(double)));
  CK(cudaMalloc(&d_scal, 2 * sizeof(double)));
  CK(cudaMalloc(&d_small, (small.size() + 1) * sizeof(int)));
  CK(cudaMalloc(&d_mid, (mid.size() + 1) * sizeof(int)));
  CK(cudaMalloc(&d_large, (large.size() + 1) * sizeof(int)));
  CK(cudaMemcpy(d_off, off32.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_src, h.src.data(), M * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_deg, h.outdeg.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_small, small.data(), small.size() * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_mid, mid.data(), mid.size() * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_large, large.data(), large.size() * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  t_h2d = now() - t_h2d;

  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  if (a.get("header", "0") == "1") std::printf("%s\n", csv_header());
  std::fprintf(stderr, "device=%s N=%lld M=%lld buckets small=%zu mid=%zu large=%zu\n", prop.name, (long long)N,
               (long long)M, small.size(), mid.size(), large.size());

  std::vector<std::string> kernels;
  if (which == "all")
    kernels = {"thread", "warp", "hybrid", "cusparse"};
  else
    kernels = {which};

  for (const std::string& kname : kernels) {
    // cuSPARSE setup (matrix of ones; x already holds r/outdeg).
    cusparseHandle_t sp = nullptr;
    cusparseSpMatDescr_t A = nullptr;
    cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
    void* sp_buf = nullptr;
    if (kname == "cusparse") {
      CK(cudaMalloc(&d_ones, M * sizeof(double)));
      std::vector<double> ones(M, 1.0);
      CK(cudaMemcpy(d_ones, ones.data(), M * sizeof(double), cudaMemcpyHostToDevice));
      CKS(cusparseCreate(&sp));
      CKS(cusparseCreateCsr(&A, N, N, M, d_off, d_src, d_ones, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
      CKS(cusparseCreateDnVec(&vx, N, d_x, CUDA_R_64F));
      CKS(cusparseCreateDnVec(&vy, N, d_rn, CUDA_R_64F));
      size_t bsz = 0;
      double one = 1, zero = 0;
      CKS(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vy, CUDA_R_64F,
                                  CUSPARSE_SPMV_CSR_ALG1, &bsz));
      CK(cudaMalloc(&sp_buf, bsz + 1));
    }

    for (int rep = -1; rep < reps; ++rep) {  // rep -1 = warm-up
      std::vector<double> init(N, 1.0 / N);
      CK(cudaMemcpy(d_r, init.data(), N * sizeof(double), cudaMemcpyHostToDevice));
      std::vector<double> resid_log;
      double resid = 0;
      int it = 0;
      CK(cudaEventRecord(e0));
      while (it < max_iters) {
        Dev g{int(N), d_off, d_src, d_deg, d_r, d_rn, d_x, d_scal, d_scal + 1, damping};
        CK(cudaMemsetAsync(d_scal, 0, 2 * sizeof(double)));
        k_contrib<<<blocks(N), TPB>>>(g);
        if (kname == "thread") {
          k_thread<<<blocks(N), TPB>>>(g, nullptr, int(N));
        } else if (kname == "warp") {
          k_warp<<<blocks(N * 32), TPB>>>(g, nullptr, int(N));
        } else if (kname == "hybrid") {
          if (!small.empty()) k_thread<<<blocks(small.size()), TPB>>>(g, d_small, int(small.size()));
          if (!mid.empty()) k_warp<<<blocks(mid.size() * 32), TPB>>>(g, d_mid, int(mid.size()));
          if (!large.empty()) k_block<<<int(large.size()), TPB>>>(g, d_large);
        } else if (kname == "cusparse") {
          double one = 1, zero = 0;
          CKS(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, vx, &zero, vy, CUDA_R_64F,
                           CUSPARSE_SPMV_CSR_ALG1, sp_buf));
          k_update<<<blocks(N), TPB>>>(g);
        } else {
          die("unknown kernel " + kname);
        }
        CK(cudaMemcpy(&resid, d_scal + 1, sizeof(double), cudaMemcpyDeviceToHost));
        std::swap(d_r, d_rn);
        if (kname == "cusparse") CKS(cusparseDnVecSetValues(vy, d_rn));
        ++it;
        if (!logp.empty()) resid_log.push_back(resid);
        if (resid < tol) break;
      }
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      CK(cudaGetLastError());
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      double tt = ms / 1000.0;
      double t_d2h = now();
      std::vector<double> r(N);
      CK(cudaMemcpy(r.data(), d_r, N * sizeof(double), cudaMemcpyDeviceToHost));
      t_d2h = now() - t_d2h;
      double sumr = 0;
      for (double v : r) sumr += v;
      if (rep >= 0) {
        std::printf("%s,cuda_%s,%s,%lld,%lld,1,1,1,1,none,none,%d,%d,%.6f,%.6e,%.6f,0,1,0,%.3e,%.15f,%.3f,%.4f,%.4f,%.2f\n",
                    exp.c_str(), kname.c_str(), basename_noext(path).c_str(), (long long)N, (long long)M, rep, it, tt,
                    tt / it, tt, resid, sumr, t_load, t_h2d, t_d2h, bytes_per_iter(N, M, 4) / (tt / it) / 1e9);
        std::fflush(stdout);
      }
      if (rep == 0 && !outp.empty()) write_vector(outp + "_" + kname + ".bin", r);
      if (rep == 0 && !logp.empty()) {
        FILE* f = open_or_die(logp + "_" + kname + ".csv", "w");
        std::fprintf(f, "iter,resid\n");
        for (size_t i = 0; i < resid_log.size(); ++i) std::fprintf(f, "%zu,%.6e\n", i + 1, resid_log[i]);
        std::fclose(f);
      }
    }
    if (kname == "cusparse") {
      cusparseDestroyDnVec(vx);
      cusparseDestroyDnVec(vy);
      cusparseDestroySpMat(A);
      cusparseDestroy(sp);
      cudaFree(sp_buf);
      cudaFree(d_ones);
      d_ones = nullptr;
    }
  }
  return 0;
}
