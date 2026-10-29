// Serial pull-style PageRank: baseline for speedup and reference for validation.
// Usage: pagerank_serial --graph g.bin [--iters 200] [--tol 1e-10] [--reps 1] [--exp tag] [--out r.bin] [--log resid.csv]
#include <cmath>
#include "../common/graph.hpp"

using namespace prg;

int main(int argc, char** argv) {
  Args a(argc, argv);
  std::string path = a.get("graph", "");
  if (path.empty()) die("--graph is required");
  int max_iters = a.geti("iters", 200);
  double tol = a.getd("tol", 1e-10), d = a.getd("damping", 0.85);
  int reps = a.geti("reps", 1);
  std::string exp = a.get("exp", "run"), out = a.get("out", ""), log = a.get("log", "");

  double tl = now();
  Graph g = load_full(path);
  tl = now() - tl;
  const int64_t N = g.h.N, M = g.h.M;
  std::vector<double> r(N), rn(N), x(N);
  if (a.get("header", "0") == "1") std::printf("%s\n", csv_header());

  for (int rep = 0; rep < reps; ++rep) {
    std::fill(r.begin(), r.end(), 1.0 / N);
    std::vector<double> resid_log;
    double t0 = now(), resid = 0;
    int it = 0;
    while (it < max_iters) {
      double dang = 0;
      for (int64_t u = 0; u < N; ++u) {
        if (g.outdeg[u]) {
          x[u] = r[u] / g.outdeg[u];
        } else {
          x[u] = 0;
          dang += r[u];
        }
      }
      const double base = (1 - d) / N + d * dang / N;
      resid = 0;
      for (int64_t v = 0; v < N; ++v) {
        double s = 0;
        for (int64_t e = g.off[v]; e < g.off[v + 1]; ++e) s += x[g.src[e]];
        double nv = base + d * s;
        resid += std::fabs(nv - r[v]);
        rn[v] = nv;
      }
      r.swap(rn);
      ++it;
      if (!log.empty()) resid_log.push_back(resid);
      if (resid < tol) break;
    }
    double tt = now() - t0;
    double sumr = 0;
    for (double v : r) sumr += v;
    std::printf("%s,serial,%s,%lld,%lld,1,1,1,1,none,none,%d,%d,%.6f,%.6e,%.6f,0,1,0,%.3e,%.15f,%.3f,,,%.2f\n", exp.c_str(),
                basename_noext(path).c_str(), (long long)N, (long long)M, rep, it, tt, tt / it, tt, resid, sumr, tl,
                bytes_per_iter(N, M, 8) / (tt / it) / 1e9);
    std::fflush(stdout);
    if (rep == 0 && !out.empty()) write_vector(out, r);
    if (rep == 0 && !log.empty()) {
      FILE* f = open_or_die(log, "w");
      std::fprintf(f, "iter,resid\n");
      for (size_t i = 0; i < resid_log.size(); ++i) std::fprintf(f, "%zu,%.6e\n", i + 1, resid_log[i]);
      std::fclose(f);
    }
  }
  return 0;
}
