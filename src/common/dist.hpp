// Distributed graph setup shared by the MPI+OpenMP and MPI+CUDA implementations:
// 1D row partition of the in-edge CSR, per-rank loading of its slice, and the ghost-exchange plan.
#pragma once
#include <mpi.h>
#include <algorithm>
#include "graph.hpp"

namespace prg {

// Row bounds of each rank: equal vertices (by_edges=false) or equal in-edges (by_edges=true).
inline std::vector<int64_t> make_bounds(const std::vector<int64_t>& off, int64_t N, int p, bool by_edges) {
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

// This rank's rows [lo, hi) with offsets rebased to 0 and global source ids in cols.
struct Slice {
  Header h;
  std::vector<int64_t> bounds;  // p + 1 row bounds
  int64_t lo = 0, hi = 0, nloc = 0, mloc = 0;
  std::vector<int64_t> off;
  std::vector<int32_t> cols, outdeg;
};

// Rank 0 reads all offsets to compute the partition and broadcasts only the bounds; each rank then reads
// its own slice of offsets, edges and out-degrees, so no rank keeps the full offset array (which matters
// for pure MPI on the large graph: 72 ranks per node x 0.5 GB).
inline Slice load_slice(const std::string& path, int rank, int p, bool by_edges) {
  Slice s;
  s.bounds.resize(p + 1);
  if (rank == 0) {
    FILE* f = open_or_die(path, "rb");
    s.h = read_header(f);
    std::vector<int64_t> goff(s.h.N + 1);
    read_at(f, off_pos(), goff.data(), goff.size() * sizeof(int64_t));
    std::fclose(f);
    s.bounds = make_bounds(goff, s.h.N, p, by_edges);
  }
  MPI_Bcast(&s.h, 2, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(s.bounds.data(), p + 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  s.lo = s.bounds[rank];
  s.hi = s.bounds[rank + 1];
  s.nloc = s.hi - s.lo;
  s.off.resize(s.nloc + 1);
  s.outdeg.resize(s.nloc);
  FILE* f = open_or_die(path, "rb");
  read_at(f, off_pos() + s.lo * (long long)sizeof(int64_t), s.off.data(), s.off.size() * sizeof(int64_t));
  const int64_t ebase = s.off[0];
  for (int64_t v = 0; v <= s.nloc; ++v) s.off[v] -= ebase;
  s.mloc = s.off[s.nloc];
  s.cols.resize(s.mloc);
  if (s.mloc) read_at(f, src_pos(s.h) + ebase * (long long)sizeof(int32_t), s.cols.data(), s.mloc * sizeof(int32_t));
  if (s.nloc) read_at(f, deg_pos(s.h) + s.lo * (long long)sizeof(int32_t), s.outdeg.data(), s.nloc * sizeof(int32_t));
  std::fclose(f);
  return s;
}

// Ghost exchange plan: every rank receives only the x entries its edges reference (MPI_Neighbor_alltoallv).
// x is laid out as [owned (nloc) | ghosts (ng)], ghosts sorted by global id and therefore grouped by owner.
struct Ghosts {
  int64_t ng = 0;
  MPI_Comm comm = MPI_COMM_NULL;           // distributed graph topology
  std::vector<int> scnt, sdsp, rcnt, rdsp;  // neighbour exchange layout
  std::vector<int32_t> send_idx;            // local indices to pack, grouped by destination
};

// Builds the plan and remaps s.cols from global ids to the local [owned | ghost] index space.
inline Ghosts build_ghosts(Slice& s, int p) {
  Ghosts g;
  const int64_t lo = s.lo, hi = s.hi, nloc = s.nloc, mloc = s.mloc;
  std::vector<int32_t> gh;
  for (int32_t c : s.cols)
    if (c < lo || c >= hi) gh.push_back(c);
  std::sort(gh.begin(), gh.end());
  gh.erase(std::unique(gh.begin(), gh.end()), gh.end());
  g.ng = gh.size();
  int32_t* cols = s.cols.data();
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
    for (int32_t v : gh) {
      while (v >= s.bounds[r + 1]) ++r;
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
  g.send_idx.resize(ngive);
  for (int64_t i = 0; i < ngive; ++i) g.send_idx[i] = int32_t(req[i] - lo);
  // Distributed graph topology: receive from ranks that own my ghosts, send to ranks that need my vertices.
  std::vector<int> srcs, dsts;
  for (int r = 0; r < p; ++r) {
    if (need[r] > 0) {
      srcs.push_back(r);
      g.rcnt.push_back(need[r]);
      g.rdsp.push_back(nd[r]);
    }
    if (give[r] > 0) {
      dsts.push_back(r);
      g.scnt.push_back(give[r]);
      g.sdsp.push_back(gd[r]);
    }
  }
  MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, int(srcs.size()), srcs.data(), MPI_UNWEIGHTED, int(dsts.size()),
                                 dsts.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &g.comm);
  return g;
}

// Number of distinct nodes the job spans, and this rank's index among the ranks on its node.
inline void node_info(int rank, int* nodes, int* local_rank) {
  MPI_Comm shm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm);
  MPI_Comm_rank(shm, local_rank);
  int is_leader = *local_rank == 0;
  MPI_Allreduce(&is_leader, nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&shm);
}

}  // namespace prg
