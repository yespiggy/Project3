// Distributed pull-style PageRank with MPI + OpenMP.
// Each rank owns a contiguous block of vertices (rows of the in-edge CSR).
//   --part vertex : equal number of vertices per rank
//   --part edge   : equal number of in-edges per rank (balances SpMV work on power-law graphs)
//   --comm allgather : every rank receives the full x vector each iteration (MPI_Allgatherv)
//   --comm ghost     : every rank receives only the x entries its edges reference (MPI_Neighbor_alltoallv)
// Usage: pagerank_mpi_omp --graph g.bin --part edge --comm ghost [--iters 50] [--tol 0] [--reps 5]
//                         [--exp tag] [--out r.bin] [--log resid.csv] [--header 1]
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include "../common/dist.hpp"

using namespace prg;

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  Args a(argc, argv);
  std::string path = a.get("graph", "");
  if (path.empty()) die("--graph is required");
  const std::string part = a.get("part", "edge"), comm = a.get("comm", "ghost");
  const bool ghost = comm == "ghost";
  const int max_iters = a.geti("iters", 50), reps = a.geti("reps", 1);
  const double tol = a.getd("tol", 0.0), d = a.getd("damping", 0.85);
  const std::string exp = a.get("exp", "run"), out = a.get("out", ""), log = a.get("log", "");

  int nodes, local_rank;
  node_info(rank, &nodes, &local_rank);
  const int threads = omp_get_max_threads();

  // ---- Load this rank's slice (see dist.hpp).
  double t_load = MPI_Wtime();
  Slice s = load_slice(path, rank, p, part == "edge");
  const int64_t N = s.h.N, M = s.h.M, lo = s.lo, nloc = s.nloc, mloc = s.mloc;
  const std::vector<int64_t>& off = s.off;
  const std::vector<int32_t>& cols = s.cols;
  const std::vector<int32_t>& outdeg = s.outdeg;

  // ---- Communication setup.
  std::vector<int> cnt(p), dsp(p);  // allgather: per-rank block sizes
  for (int r = 0; r < p; ++r) {
    cnt[r] = int(s.bounds[r + 1] - s.bounds[r]);
    dsp[r] = int(s.bounds[r]);
  }
  std::vector<double> xbuf;   // allgather: global x (size N); ghost: [owned | ghosts]
  int64_t own_off = 0;        // where this rank's own x entries live inside xbuf
  Ghosts gx;
  std::vector<double> sendbuf;
  double comm_bytes_local = 0;

  if (!ghost) {
    xbuf.assign(N, 0.0);
    own_off = lo;
    comm_bytes_local = double(N - nloc) * 8;
  } else {
    gx = build_ghosts(s, p);  // remaps cols to the local [owned | ghost] index space
    sendbuf.resize(gx.send_idx.size());
    xbuf.assign(nloc + gx.ng, 0.0);
    own_off = 0;
    comm_bytes_local = double(gx.ng) * 8;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t_load = MPI_Wtime() - t_load;

  double mloc_d = double(mloc), mmax, msum, comm_bytes;
  MPI_Allreduce(&mloc_d, &mmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&mloc_d, &msum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&comm_bytes_local, &comm_bytes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double imbalance = mmax / (msum / p);

  if (rank == 0 && a.get("header", "0") == "1") std::printf("%s\n", csv_header());

  std::vector<double> r(nloc), rn(nloc);
  double* xown = xbuf.data() + own_off;
  // rep -1 is an unreported warm-up.
  for (int rep = -1; rep < reps; ++rep) {
    std::fill(r.begin(), r.end(), 1.0 / N);
    std::vector<double> resid_log;
    double t_comp = 0, t_comm = 0, resid = 0;
    int it = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    while (it < max_iters) {
      // (1) local contributions and dangling mass
      double tc = MPI_Wtime(), dang_loc = 0;
#pragma omp parallel for schedule(static) reduction(+ : dang_loc)
      for (int64_t u = 0; u < nloc; ++u) {
        if (outdeg[u]) {
          xown[u] = r[u] / outdeg[u];
        } else {
          xown[u] = 0;
          dang_loc += r[u];
        }
      }
      double tm = MPI_Wtime();
      t_comp += tm - tc;
      // (2) exchange x, (3) global dangling mass
      if (ghost) {
        const int64_t ns = gx.send_idx.size();
        const int32_t* sidx = gx.send_idx.data();
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < ns; ++i) sendbuf[i] = xown[sidx[i]];
        MPI_Neighbor_alltoallv(sendbuf.data(), gx.scnt.data(), gx.sdsp.data(), MPI_DOUBLE, xbuf.data() + nloc,
                               gx.rcnt.data(), gx.rdsp.data(), MPI_DOUBLE, gx.comm);
      } else {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, xbuf.data(), cnt.data(), dsp.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
      }
      double dang = 0;
      MPI_Allreduce(&dang_loc, &dang, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      tc = MPI_Wtime();
      t_comm += tc - tm;
      // (4) local SpMV
      const double base = (1 - d) / N + d * dang / N;
      double res_loc = 0;
      const double* xb = xbuf.data();
#pragma omp parallel for schedule(dynamic, 512) reduction(+ : res_loc)
      for (int64_t v = 0; v < nloc; ++v) {
        double s = 0;
        for (int64_t e = off[v]; e < off[v + 1]; ++e) s += xb[cols[e]];
        double nv = base + d * s;
        res_loc += std::fabs(nv - r[v]);
        rn[v] = nv;
      }
      tm = MPI_Wtime();
      t_comp += tm - tc;
      // (5) global residual
      MPI_Allreduce(&res_loc, &resid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      t_comm += MPI_Wtime() - tm;
      r.swap(rn);
      ++it;
      if (!log.empty()) resid_log.push_back(resid);
      if (resid < tol) break;
    }
    double tt = MPI_Wtime() - t0;
    double comp_max, comm_max, sum_loc = 0, sumr = 0;
    for (double v : r) sum_loc += v;
    MPI_Reduce(&t_comp, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_loc, &sumr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rep >= 0 && rank == 0) {
      std::printf("%s,mpi_omp,%s,%lld,%lld,%d,%d,%d,%d,%s,%s,%d,%d,%.6f,%.6e,%.6f,%.6f,%.4f,%.0f,%.3e,%.15f,%.3f,,,%.2f,\n",
                  exp.c_str(), basename_noext(path).c_str(), (long long)N, (long long)M, nodes, p, threads, p * threads,
                  part.c_str(), comm.c_str(), rep, it, tt, tt / it, comp_max, comm_max, imbalance, comm_bytes, resid, sumr,
                  t_load, bytes_per_iter(N, M, 8) / (tt / it) / 1e9);
      std::fflush(stdout);
    }
    if (rep == 0 && !out.empty()) {
      std::vector<double> full(rank == 0 ? N : 0);
      MPI_Gatherv(r.data(), int(nloc), MPI_DOUBLE, full.data(), cnt.data(), dsp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (rank == 0) write_vector(out, full);
    }
    if (rep == 0 && rank == 0 && !log.empty()) {
      FILE* f = open_or_die(log, "w");
      std::fprintf(f, "iter,resid\n");
      for (size_t i = 0; i < resid_log.size(); ++i) std::fprintf(f, "%zu,%.6e\n", i + 1, resid_log[i]);
      std::fclose(f);
    }
  }
  if (gx.comm != MPI_COMM_NULL) MPI_Comm_free(&gx.comm);
  MPI_Finalize();
  return 0;
}
