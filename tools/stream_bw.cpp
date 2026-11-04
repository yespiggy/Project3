// STREAM-style memory-bandwidth probe for one CPU node (OpenMP), used to put the PageRank effective-bandwidth
// estimate (gbps column) in context. Reports the best of --reps runs for copy (a=b) and triad (a=b+s*c), counting
// 16 and 24 bytes per element respectively (no write-allocate traffic counted, as in STREAM).
// Usage: stream_bw [--mb 4096] [--reps 10]      (array size per vector in MiB; must exceed the last-level cache)
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  long long mb = 4096;
  int reps = 10;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--mb")) mb = std::atoll(argv[i + 1]);
    if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
  }
  const long long n = mb * 1024 * 1024 / 8;
  double* a = static_cast<double*>(std::aligned_alloc(64, n * 8));
  double* b = static_cast<double*>(std::aligned_alloc(64, n * 8));
  double* c = static_cast<double*>(std::aligned_alloc(64, n * 8));
  if (!a || !b || !c) {
    std::fprintf(stderr, "allocation failed\n");
    return 1;
  }
  // First touch in parallel so pages are spread over the NUMA domains the same way the kernels access them.
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < n; ++i) {
    a[i] = 0;
    b[i] = 1;
    c[i] = 2;
  }
  double best_copy = 1e30, best_triad = 1e30;
  for (int r = 0; r < reps; ++r) {
    double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) a[i] = b[i];
    best_copy = std::min(best_copy, omp_get_wtime() - t);
    t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    best_triad = std::min(best_triad, omp_get_wtime() - t);
  }
  double chk = 0;
  for (long long i = 0; i < n; i += 4096) chk += a[i];
  std::printf("threads,mib_per_array,copy_gbs,triad_gbs,check\n%d,%lld,%.1f,%.1f,%.1f\n", omp_get_max_threads(), mb,
              16.0 * n / best_copy / 1e9, 24.0 * n / best_triad / 1e9, chk);
  std::free(a);
  std::free(b);
  std::free(c);
  return 0;
}
