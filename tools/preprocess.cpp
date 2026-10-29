// SNAP edge list (text) -> binary in-edge CSR.
// Usage: preprocess <edges.txt> <out.bin> [idmap.bin]
// Lines starting with '#' are skipped. Node IDs are remapped to 0..N-1 (sorted by original ID),
// self loops and duplicate edges are removed. idmap.bin stores the original ID of each new ID (int64).
#include <algorithm>
#include <parallel/algorithm>
#include "../src/common/graph.hpp"

using namespace prg;

int main(int argc, char** argv) {
  if (argc < 3) die("usage: preprocess <edges.txt> <out.bin> [idmap.bin]");
  double t0 = now();
  FILE* f = open_or_die(argv[1], "rb");
  std::fseek(f, 0, SEEK_END);
  long long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> buf(sz + 1);
  if ((long long)std::fread(buf.data(), 1, sz, f) != sz) die("read failed");
  buf[sz] = '\n';
  std::fclose(f);

  std::vector<uint64_t> edges;  // (src << 32 | dst) with original IDs
  edges.reserve(sz / 12);
  const char* p = buf.data();
  const char* end = p + sz;
  while (p < end) {
    if (*p == '#') {
      while (p < end && *p != '\n') ++p;
      ++p;
      continue;
    }
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    if (p >= end) break;
    uint64_t a = 0, b = 0;
    while (*p >= '0' && *p <= '9') a = a * 10 + (*p++ - '0');
    while (*p == ' ' || *p == '\t') ++p;
    while (*p >= '0' && *p <= '9') b = b * 10 + (*p++ - '0');
    while (p < end && *p != '\n') ++p;
    edges.push_back((a << 32) | b);
  }
  long long raw = edges.size();

  // Remap IDs over all endpoints.
  std::vector<uint32_t> ids;
  ids.reserve(edges.size() * 2);
  for (uint64_t e : edges) {
    ids.push_back(uint32_t(e >> 32));
    ids.push_back(uint32_t(e));
  }
  __gnu_parallel::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  int64_t N = ids.size();
  std::vector<int32_t> newid(ids.back() + 1, -1);
  for (int64_t i = 0; i < N; ++i) newid[ids[i]] = int32_t(i);

  // Re-encode as (dst << 32 | src) so sorting groups edges by destination row.
  long long selfloops = 0;
  size_t w = 0;
  for (uint64_t e : edges) {
    uint64_t s = newid[e >> 32], d = newid[uint32_t(e)];
    if (s == d) {
      ++selfloops;
      continue;
    }
    edges[w++] = (d << 32) | s;
  }
  edges.resize(w);
  __gnu_parallel::sort(edges.begin(), edges.end());
  size_t before = edges.size();
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  long long dups = before - edges.size();

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
  write_graph(argv[2], N, off, src, outdeg);

  if (argc > 3) {
    FILE* m = open_or_die(argv[3], "wb");
    std::vector<int64_t> orig(ids.begin(), ids.end());
    std::fwrite(orig.data(), sizeof(int64_t), orig.size(), m);
    std::fclose(m);
  }

  long long dangling = 0, maxin = 0, maxout = 0;
  for (int64_t v = 0; v < N; ++v) {
    dangling += outdeg[v] == 0;
    maxin = std::max<long long>(maxin, off[v + 1] - off[v]);
    maxout = std::max<long long>(maxout, outdeg[v]);
  }
  std::printf("graph=%s N=%lld M=%lld raw_edges=%lld selfloops=%lld duplicates=%lld dangling=%lld max_in=%lld max_out=%lld time=%.1fs\n",
              basename_noext(argv[2]).c_str(), (long long)N, (long long)M, raw, selfloops, dups, dangling, maxin, maxout,
              now() - t0);
  return 0;
}
