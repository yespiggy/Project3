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
#include "../common/graph.hpp"

using namespace prg;

static std::vector<int64_t> make_bounds(const std::vector<int64_t>& off, int64_t N, int p, bool by_edges) {
  std::vector<int64_t> b(p + 1);
  b[0] = 0;
  b[p] = N;
  const int64_t M = off[N];
  for (int r = 1; r < p; ++r) {
    if (by_edges) {
      int64_t target = M * r / p;
      b[r] = std::lower_bound(off.begin(), off.end(), target) - off.begin();
      b[r] = std::min<int64_t>(std::max<int64_t>(b[r], b[r - 1]), N);
    } else {
      b[r] = N * r / p;
    }
  }
  return b;
}

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

  // Count distinct nodes for reporting.
  MPI_Comm shm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm);
  int shm_rank;
  MPI_Comm_rank(shm, &shm_rank);
  int is_leader = shm_rank == 0, nodes = 0;
  MPI_Allreduce(&is_leader, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const int threads = omp_get_max_threads();

  // ---- Load: rank 0 reads offsets and broadcasts; each rank reads only its own slice of edges.
  double t_load = MPI_Wtime();
  Header h;
  std::vector<int64_t> goff;
  if (rank == 0) {
    FILE* f = open_or_die(path, "rb");
    h = read_header(f);
    goff.resize(h.N + 1);
    read_at(f, off_pos(), goff.data(), goff.size() * sizeof(int64_t));
    std::fclose(f);
  }
  MPI_Bcast(&h, 2, MPI_INT64_T, 0, MPI_COMM_WORLD);
  const int64_t N = h.N, M = h.M;
  goff.resize(N + 1);
  MPI_Bcast(goff.data(), N + 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

  std::vector<int64_t> bounds = make_bounds(goff, N, p, part == "edge");
  const int64_t lo = bounds[rank], hi = bounds[rank + 1], nloc = hi - lo;
  const int64_t ebase = goff[lo], mloc = goff[hi] - goff[lo];
  std::vector<int64_t> off(nloc + 1);
  for (int64_t v = 0; v <= nloc; ++v) off[v] = goff[lo + v] - ebase;
  std::vector<int32_t> cols(mloc), outdeg(nloc);
  {
    FILE* f = open_or_die(path, "rb");
    if (mloc) read_at(f, src_pos(h) + ebase * (long long)sizeof(int32_t), cols.data(), mloc * sizeof(int32_t));
    if (nloc) read_at(f, deg_pos(h) + lo * (long long)sizeof(int32_t), outdeg.data(), nloc * sizeof(int32_t));
    std::fclose(f);
  }
  std::vector<int64_t>().swap(goff);

  // ---- Communication setup.
  std::vector<int> cnt(p), dsp(p);  // allgather: per-rank block sizes
  for (int r = 0; r < p; ++r) {
    cnt[r] = int(bounds[r + 1] - bounds[r]);
    dsp[r] = int(bounds[r]);
  }
  std::vector<double> xbuf;   // allgather: global x (size N); ghost: [owned | ghosts]
  int64_t own_off = 0;        // where this rank's own x entries live inside xbuf
  MPI_Comm gcomm = MPI_COMM_NULL;
  std::vector<int> scnt, sdsp, rcnt, rdsp;  // neighbour exchange layout
  std::vector<int32_t> send_idx;            // local indices to pack, grouped by destination
  std::vector<double> sendbuf;
  double comm_bytes_local = 0;

  if (!ghost) {
    xbuf.assign(N, 0.0);
    own_off = lo;
    comm_bytes_local = double(N - nloc) * 8;
  } else {
    // Ghosts: distinct sources outside [lo, hi), sorted by global id => grouped by owner rank.
    std::vector<int32_t> gh;
    for (int32_t c : cols)
      if (c < lo || c >= hi) gh.push_back(c);
    std::sort(gh.begin(), gh.end());
    gh.erase(std::unique(gh.begin(), gh.end()), gh.end());
    const int64_t ng = gh.size();
    // Remap columns to local index space.
#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < mloc; ++e) {
      int32_t c = cols[e];
      if (c >= lo && c < hi)
        cols[e] = int32_t(c - lo);
      else
        cols[e] = int32_t(nloc + (std::lower_bound(gh.begin(), gh.end(), c) - gh.begin()));
    }
    // How many ghosts I need from each owner.
    std::vector<int> need(p, 0), give(p, 0);
    {
      int r = 0;
      for (int32_t g : gh) {
        while (g >= bounds[r + 1]) ++r;
        need[r]++;
      }
    }
    MPI_Alltoall(need.data(), 1, MPI_INT, give.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> nd(p, 0), gd(p, 0);
    for (int r = 1; r < p; ++r) {
      nd[r] = nd[r - 1] + need[r - 1];
      gd[r] = gd[r - 1] + give[r - 1];
    }
    int64_t ngive = gd[p - 1] + give[p - 1];
    std::vector<int32_t> req(ngive);
    MPI_Alltoallv(gh.data(), need.data(), nd.data(), MPI_INT32_T, req.data(), give.data(), gd.data(), MPI_INT32_T,
                  MPI_COMM_WORLD);
    send_idx.resize(ngive);
    for (int64_t i = 0; i < ngive; ++i) send_idx[i] = int32_t(req[i] - lo);
    sendbuf.resize(ngive);
    // Distributed graph topology: receive from ranks that own my ghosts, send to ranks that need my vertices.
    std::vector<int> srcs, dsts;
    for (int r = 0; r < p; ++r) {
      if (need[r] > 0) {
        srcs.push_back(r);
        rcnt.push_back(need[r]);
        rdsp.push_back(nd[r]);
      }
      if (give[r] > 0) {
        dsts.push_back(r);
        scnt.push_back(give[r]);
        sdsp.push_back(gd[r]);
      }
    }
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, int(srcs.size()), srcs.data(), MPI_UNWEIGHTED, int(dsts.size()),
                                   dsts.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &gcomm);
    xbuf.assign(nloc + ng, 0.0);
    own_off = 0;
    comm_bytes_local = double(ng) * 8;
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
        const int64_t ns = send_idx.size();
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < ns; ++i) sendbuf[i] = xown[send_idx[i]];
        MPI_Neighbor_alltoallv(sendbuf.data(), scnt.data(), sdsp.data(), MPI_DOUBLE, xbuf.data() + nloc, rcnt.data(),
                               rdsp.data(), MPI_DOUBLE, gcomm);
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
      std::printf("%s,mpi_omp,%s,%lld,%lld,%d,%d,%d,%d,%s,%s,%d,%d,%.6f,%.6e,%.6f,%.6f,%.4f,%.0f,%.3e,%.15f,%.3f,,,%.2f\n",
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
  if (gcomm != MPI_COMM_NULL) MPI_Comm_free(&gcomm);
  MPI_Comm_free(&shm);
  MPI_Finalize();
  return 0;
}
