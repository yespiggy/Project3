// R-MAT graph generator (Graph500 parameters a=.57 b=.19 c=.19 d=.05) writing binary in-edge CSR.
// Usage: rmat_gen <N> <edge_factor> <seed> <out.bin>
// Vertex labels are randomly permuted, self loops and duplicate edges are removed.
#include <algorithm>
#include <numeric>
#include <random>
#include <parallel/algorithm>
#include <omp.h>
#include "../src/common/graph.hpp"

using namespace prg;

int main(int argc, char** argv) {
  if (argc < 5) die("usage: rmat_gen <N> <edge_factor> <seed> <out.bin>");
  double t0 = now();
  int64_t N = std::atoll(argv[1]);
  int64_t ef = std::atoll(argv[2]);
  uint64_t seed = std::strtoull(argv[3], nullptr, 10);
  int scale = 0;
  while ((int64_t(1) << scale) < N) ++scale;
  int64_t target = N * ef;
  const double a = 0.57, b = 0.19, c = 0.19;

  std::vector<int32_t> perm(N);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937_64(seed ^ 0x9e3779b97f4a7c15ULL));

  std::vector<uint64_t> edges(target);
#pragma omp parallel
  {
    int t = omp_get_thread_num(), nt = omp_get_num_threads();
    std::mt19937_64 rng(seed * 1000003ULL + t);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int64_t lo = target * t / nt, hi = target * (t + 1) / nt;
    for (int64_t i = lo; i < hi;) {
      uint64_t u = 0, v = 0;
      for (int s = 0; s < scale; ++s) {
        double r = U(rng);
        int bu = 0, bv = 0;
        if (r < a) {
        } else if (r < a + b) {
          bv = 1;
        } else if (r < a + b + c) {
          bu = 1;
        } else {
          bu = bv = 1;
        }
        u = (u << 1) | bu;
        v = (v << 1) | bv;
      }
      if ((int64_t)u >= N || (int64_t)v >= N) continue;
      uint64_t s = perm[u], d = perm[v];
      edges[i++] = (d << 32) | s;
    }
  }
  edges.erase(std::remove_if(edges.begin(), edges.end(), [](uint64_t e) { return (e >> 32) == (e & 0xffffffffULL); }),
              edges.end());
  __gnu_parallel::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  int64_t M = edges.size();
  std::vector<int64_t> off(N + 1, 0);
  std::vector<int32_t> src(M), outdeg(N, 0);
  for (int64_t i = 0; i < M; ++i) {
    int32_t d = int32_t(edges[i] >> 32), s = int32_t(uint32_t(edges[i]));
    off[d + 1]++;
    src[i] = s;
    outdeg[s]++;
  }
  for (int64_t v = 0; v < N; ++v) off[v + 1] += off[v];
  write_graph(argv[4], N, off, src, outdeg);
  long long dangling = 0;
  for (int64_t v = 0; v < N; ++v) dangling += outdeg[v] == 0;
  std::printf("graph=%s N=%lld M=%lld scale=%d dangling=%lld time=%.1fs\n", basename_noext(argv[4]).c_str(), (long long)N,
              (long long)M, scale, dangling, now() - t0);
  return 0;
}
