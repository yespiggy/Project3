// Multi-GPU pull-style PageRank with MPI + CUDA: one MPI rank per GPU.
// Same 1D edge-balanced row partition and ghost exchange as the CPU version (dist.hpp). Each rank keeps its
// slice of the CSR and its [owned | ghost] x vector resident on its GPU. Per iteration:
//   GPU: x_own = r / outdeg, local dangling mass, pack the x entries other ranks need   (t_comp)
//   D2H of the send buffer + dangling scalar                                            (t_stage)
//   MPI_Neighbor_alltoallv of x entries, MPI_Allreduce of the dangling mass              (t_comm)
//   H2D of the received ghosts                                                          (t_stage)
//   GPU: SpMV over local rows + residual                                                (t_comp)
//   D2H of the residual scalar (t_stage), MPI_Allreduce of the residual                 (t_comm)
// Host staging is used instead of CUDA-aware MPI so the code runs with any MPI build.
// Device offsets are 64-bit so one GPU can hold the large graph (> 2^31 edges); cuSPARSE is therefore not used
// here (its CSR SpMV needs matching offset/index types); the single-GPU program covers the cuSPARSE baseline.
// Kernels: thread (one thread per row) and hybrid (rows bucketed by in-degree: thread / warp / block).
// Usage: mpirun -np <gpus> pagerank_mpi_cuda --graph g.bin [--kernel thread,hybrid] [--iters 50] [--tol 0]
//                                            [--reps 5] [--exp tag] [--out prefix] [--header 1]
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <sstream>
#include "../common/dist.hpp"

using namespace prg;

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) die(std::string(#x) + ": " + cudaGetErrorString(e_));                \
  } while (0)

constexpr int TPB = 256;
constexpr int SMALL_DEG = 16, LARGE_DEG = 1024;

struct Dev {
  int nloc;
  const int64_t* off;
  const int* cols;    // local indices into x = [owned | ghosts]
  const int* outdeg;
  const double* r;
  double* rn;
  double* x;
  double* dang;   // device scalar: local dangling mass
  double* resid;  // device scalar: local L1 residual
  double base;    // (1-d)/N + d*dangling/N, computed on the host after the global reduction
  double d;
};

__device__ void block_sum_atomic(double v, double* out) {
  typedef cub::BlockReduce<double, TPB> BR;
  __shared__ typename BR::TempStorage tmp;
  double s = BR(tmp).Sum(v);
  if (threadIdx.x == 0 && s != 0.0) atomicAdd(out, s);
}

__global__ void k_contrib(Dev g) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  double dl = 0;
  if (u < g.nloc) {
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

__global__ void k_pack(const double* x, const int* idx, int64_t n, double* out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[idx[i]];
}

__device__ __forceinline__ double finish_row(const Dev& g, int v, double s) {
  double nv = g.base + g.d * s;
  g.rn[v] = nv;
  return fabs(nv - g.r[v]);
}

__global__ void k_thread(Dev g, const int* rows, int nrows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double res = 0;
  if (i < nrows) {
    int v = rows ? rows[i] : i;
    double s = 0;
    for (int64_t e = g.off[v]; e < g.off[v + 1]; ++e) s += g.x[g.cols[e]];
    res = finish_row(g, v, s);
  }
  block_sum_atomic(res, g.resid);
}

__global__ void k_warp(Dev g, const int* rows, int nrows) {
  int w = int((blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
  double res = 0;
  if (w < nrows) {
    int v = rows[w];
    double s = 0;
    for (int64_t e = g.off[v] + lane; e < g.off[v + 1]; e += 32) s += g.x[g.cols[e]];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) res = finish_row(g, v, s);
  }
  block_sum_atomic(res, g.resid);
}

__global__ void k_block(Dev g, const int* rows) {
  int v = rows[blockIdx.x];
  double s = 0;
  for (int64_t e = g.off[v] + threadIdx.x; e < g.off[v + 1]; e += blockDim.x) s += g.x[g.cols[e]];
  typedef cub::BlockReduce<double, TPB> BR;
  __shared__ typename BR::TempStorage tmp;
  double tot = BR(tmp).Sum(s);
  if (threadIdx.x == 0) atomicAdd(g.resid, finish_row(g, v, tot));
}

static int blocks(long long n, int per = TPB) { return int((n + per - 1) / per); }

template <typename T>
static T* to_device(const std::vector<T>& v) {
  T* p = nullptr;
  CK(cudaMalloc(&p, (v.size() + 1) * sizeof(T)));
  if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  Args a(argc, argv);
  std::string path = a.get("graph", "");
  if (path.empty()) die("--graph is required");
  const int max_iters = a.geti("iters", 50), reps = a.geti("reps", 1);
  const double tol = a.getd("tol", 0.0), damping = a.getd("damping", 0.85);
  const std::string exp = a.get("exp", "run"), outp = a.get("out", "");
  std::vector<std::string> kernels;
  {
    std::stringstream ss(a.get("kernel", "thread,hybrid"));
    for (std::string k; std::getline(ss, k, ',');) kernels.push_back(k);
  }

  // One GPU per rank: the i-th rank on a node takes the i-th visible device.
  int nodes, local_rank, ndev = 0;
  node_info(rank, &nodes, &local_rank);
  CK(cudaGetDeviceCount(&ndev));
  if (ndev == 0) die("no visible GPU");
  const int dev = local_rank % ndev;
  CK(cudaSetDevice(dev));
  CK(cudaFree(0));
  {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    char bus[32];
    CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), dev));
    char host[256];
    int hl;
    MPI_Get_processor_name(host, &hl);
    std::fprintf(stderr, "rank %d/%d host=%s local_rank=%d visible_gpus=%d device=%d %s bus=%s\n", rank, p, host,
                 local_rank, ndev, dev, prop.name, bus);
    if (local_rank >= ndev) std::fprintf(stderr, "warning: rank %d shares a GPU (more ranks than GPUs on node)\n", rank);
  }

  // ---- Load and partition (same as the CPU version), then upload.
  double t_load = MPI_Wtime();
  Slice s = load_slice(path, rank, p, true);
  Ghosts gx = build_ghosts(s, p);
  const int64_t N = s.h.N, M = s.h.M, nloc = s.nloc, mloc = s.mloc, ns = gx.send_idx.size(), ng = gx.ng;
  if (nloc + ng >= (int64_t(1) << 31)) die("local vertex count does not fit in 32 bits");
  std::vector<int> small, mid, large;  // degree buckets for the hybrid kernel
  for (int64_t v = 0; v < nloc; ++v) {
    int64_t deg = s.off[v + 1] - s.off[v];
    (deg <= SMALL_DEG ? small : deg <= LARGE_DEG ? mid : large).push_back(int(v));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t_load = MPI_Wtime() - t_load;

  double t_h2d = MPI_Wtime();
  int64_t* d_off = to_device(s.off);
  int* d_cols = to_device(s.cols);
  int* d_deg = to_device(s.outdeg);
  int* d_send_idx = to_device(gx.send_idx);
  int* d_small = to_device(small);
  int* d_mid = to_device(mid);
  int* d_large = to_device(large);
  double *d_r, *d_rn, *d_x, *d_send, *d_scal;
  CK(cudaMalloc(&d_r, (nloc + 1) * sizeof(double)));
  CK(cudaMalloc(&d_rn, (nloc + 1) * sizeof(double)));
  CK(cudaMalloc(&d_x, (nloc + ng + 1) * sizeof(double)));
  CK(cudaMalloc(&d_send, (ns + 1) * sizeof(double)));
  CK(cudaMalloc(&d_scal, 2 * sizeof(double)));
  CK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  t_h2d = MPI_Wtime() - t_h2d;
  std::vector<int32_t>().swap(s.cols);  // host copy no longer needed

  double *h_send, *h_recv, *h_scal;  // pinned staging buffers
  CK(cudaMallocHost(&h_send, (ns + 1) * sizeof(double)));
  CK(cudaMallocHost(&h_recv, (ng + 1) * sizeof(double)));
  CK(cudaMallocHost(&h_scal, 2 * sizeof(double)));

  double mloc_d = double(mloc), mmax, msum, comm_bytes, comm_bytes_local = double(ng) * 8;
  MPI_Allreduce(&mloc_d, &mmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&mloc_d, &msum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&comm_bytes_local, &comm_bytes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double imbalance = mmax / (msum / p);
  if (rank == 0 && a.get("header", "0") == "1") std::printf("%s\n", csv_header());

  std::vector<int> cnt(p), dsp(p);
  for (int r = 0; r < p; ++r) {
    cnt[r] = int(s.bounds[r + 1] - s.bounds[r]);
    dsp[r] = int(s.bounds[r]);
  }
  const std::vector<double> init(nloc, 1.0 / N);

  for (const std::string& kname : kernels) {
    if (kname != "thread" && kname != "hybrid") die("unknown kernel " + kname + " (thread|hybrid)");
    for (int rep = -1; rep < reps; ++rep) {  // rep -1 = warm-up
      CK(cudaMemcpy(d_r, init.data(), nloc * sizeof(double), cudaMemcpyHostToDevice));
      double t_comp = 0, t_comm = 0, t_stage = 0, resid = 0;
      int it = 0;
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      while (it < max_iters) {
        Dev g{int(nloc), d_off, d_cols, d_deg, d_r, d_rn, d_x, d_scal, d_scal + 1, 0.0, damping};
        // (1) contributions, local dangling mass, pack send buffer
        double t = MPI_Wtime();
        CK(cudaMemsetAsync(d_scal, 0, 2 * sizeof(double)));
        if (nloc) k_contrib<<<blocks(nloc), TPB>>>(g);
        if (ns) k_pack<<<blocks(ns), TPB>>>(d_x, d_send_idx, ns, d_send);
        CK(cudaDeviceSynchronize());
        double t2 = MPI_Wtime();
        t_comp += t2 - t;
        // (2) stage to host
        if (ns) CK(cudaMemcpy(h_send, d_send, ns * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_scal, d_scal, sizeof(double), cudaMemcpyDeviceToHost));
        t = MPI_Wtime();
        t_stage += t - t2;
        // (3) exchange ghosts, reduce dangling mass
        MPI_Neighbor_alltoallv(h_send, gx.scnt.data(), gx.sdsp.data(), MPI_DOUBLE, h_recv, gx.rcnt.data(),
                               gx.rdsp.data(), MPI_DOUBLE, gx.comm);
        double dang = 0;
        MPI_Allreduce(h_scal, &dang, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        t_comm += t2 - t;
        // (4) ghosts back to the device
        if (ng) CK(cudaMemcpy(d_x + nloc, h_recv, ng * sizeof(double), cudaMemcpyHostToDevice));
        t = MPI_Wtime();
        t_stage += t - t2;
        // (5) local SpMV + residual
        g.base = (1 - damping) / N + damping * dang / N;
        if (kname == "thread") {
          if (nloc) k_thread<<<blocks(nloc), TPB>>>(g, nullptr, int(nloc));
        } else {
          if (!small.empty()) k_thread<<<blocks(small.size()), TPB>>>(g, d_small, int(small.size()));
          if (!mid.empty()) k_warp<<<blocks(mid.size() * 32), TPB>>>(g, d_mid, int(mid.size()));
          if (!large.empty()) k_block<<<int(large.size()), TPB>>>(g, d_large);
        }
        CK(cudaDeviceSynchronize());
        t2 = MPI_Wtime();
        t_comp += t2 - t;
        CK(cudaMemcpy(h_scal + 1, d_scal + 1, sizeof(double), cudaMemcpyDeviceToHost));
        t = MPI_Wtime();
        t_stage += t - t2;
        // (6) global residual
        MPI_Allreduce(h_scal + 1, &resid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - t;
        std::swap(d_r, d_rn);
        ++it;
        if (resid < tol) break;
      }
      double tt = MPI_Wtime() - t0;
      CK(cudaGetLastError());
      std::vector<double> r(nloc);
      double t_d2h = MPI_Wtime();
      CK(cudaMemcpy(r.data(), d_r, nloc * sizeof(double), cudaMemcpyDeviceToHost));
      t_d2h = MPI_Wtime() - t_d2h;
      double sum_loc = 0, sumr = 0, comp_max, comm_max, stage_max, d2h_max;
      for (double v : r) sum_loc += v;
      MPI_Reduce(&t_comp, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&t_comm, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&t_stage, &stage_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&t_d2h, &d2h_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&sum_loc, &sumr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rep >= 0 && rank == 0) {
        std::printf("%s,mpi_cuda_%s,%s,%lld,%lld,%d,%d,1,%d,edge,ghost,%d,%d,%.6f,%.6e,%.6f,%.6f,%.4f,%.0f,%.3e,%.15f,"
                    "%.3f,%.4f,%.4f,%.2f,%.6f\n",
                    exp.c_str(), kname.c_str(), basename_noext(path).c_str(), (long long)N, (long long)M, nodes, p, p,
                    rep, it, tt, tt / it, comp_max, comm_max, imbalance, comm_bytes, resid, sumr, t_load, t_h2d,
                    d2h_max, bytes_per_iter(N, M, 8) / (tt / it) / 1e9, stage_max);
        std::fflush(stdout);
      }
      if (rep == 0 && !outp.empty()) {
        std::vector<double> full(rank == 0 ? N : 0);
        MPI_Gatherv(r.data(), int(nloc), MPI_DOUBLE, full.data(), cnt.data(), dsp.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) write_vector(outp + "_" + kname + ".bin", full);
      }
    }
  }

  cudaFreeHost(h_send);
  cudaFreeHost(h_recv);
  cudaFreeHost(h_scal);
  for (void* q : {(void*)d_off, (void*)d_cols, (void*)d_deg, (void*)d_send_idx, (void*)d_small, (void*)d_mid,
                  (void*)d_large, (void*)d_r, (void*)d_rn, (void*)d_x, (void*)d_send, (void*)d_scal})
    cudaFree(q);
  if (gx.comm != MPI_COMM_NULL) MPI_Comm_free(&gx.comm);
  MPI_Finalize();
  return 0;
}
