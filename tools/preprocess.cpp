// SNAP edge list (text) -> binary in-edge CSR.
// Usage: preprocess [--undirected] <edges.txt | -> <out.bin> [idmap.bin]
// "-" reads the edge list from stdin (e.g. zcat big.txt.gz | preprocess - big.bin), streamed in chunks so the
// text never has to fit in memory. --undirected adds both directions of every edge (SNAP *.ungraph files).
// Lines starting with '#' are skipped. Node IDs are remapped to 0..N-1 (sorted by original ID),
// self loops and duplicate edges are removed. idmap.bin stores the original ID of each new ID (int64).
#include <algorithm>
#include <parallel/algorithm>
#include "../src/common/graph.hpp"

using namespace prg;

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  bool undirected = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--undirected")
      undirected = true;
    else
      pos.push_back(argv[i]);
  }
  if (pos.size() < 2) die("usage: preprocess [--undirected] <edges.txt|-> <out.bin> [idmap.bin]");
  double t0 = now();
  FILE* f = pos[0] == "-" ? stdin : open_or_die(pos[0], "rb");

  // Stream the text in chunks; a partial last line is carried over to the next chunk.
  std::vector<uint64_t> edges;  // (src << 32 | dst) with original IDs
  uint64_t maxid = 0;
  const size_t CHUNK = size_t(1) << 26;
  std::vector<char> buf(CHUNK + 1);
  size_t carry = 0;
  bool eof = false;
  while (!eof) {
    size_t got = std::fread(buf.data() + carry, 1, CHUNK - carry, f);
    eof = got < CHUNK - carry;
    size_t len = carry + got;
    size_t stop = len;  // parse up to the last newline unless this is the end of input
    if (!eof) {
      while (stop > 0 && buf[stop - 1] != '\n') --stop;
      if (stop == 0) die("line longer than chunk");
    } else {
      buf[len] = '\n';
      stop = len + 1;
    }
    const char* p = buf.data();
    const char* end = p + stop;
    while (p < end) {
      while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
      if (p >= end) break;
      if (*p == '#') {
        while (p < end && *p != '\n') ++p;
        continue;
      }
      uint64_t a = 0, b = 0;
      while (*p >= '0' && *p <= '9') a = a * 10 + (*p++ - '0');
      while (*p == ' ' || *p == '\t') ++p;
      while (*p >= '0' && *p <= '9') b = b * 10 + (*p++ - '0');
      while (p < end && *p != '\n') ++p;
      if (a > 0xffffffffULL || b > 0xffffffffULL) die("node id does not fit in 32 bits");
      maxid = std::max(maxid, std::max(a, b));
      edges.push_back((a << 32) | b);
      if (undirected) edges.push_back((b << 32) | a);
    }
    carry = len - std::min(stop, len);
    if (carry) std::memmove(buf.data(), buf.data() + stop, carry);
  }
  if (f != stdin) std::fclose(f);
  std::vector<char>().swap(buf);
  long long raw = edges.size();
  std::fprintf(stderr, "parsed %lld directed edges in %.1fs\n", raw, now() - t0);

  // Remap IDs over all endpoints (dense marker array indexed by original ID, so order follows original ID).
  std::vector<int32_t> newid(maxid + 1, -1);
  for (uint64_t e : edges) {
    newid[e >> 32] = 0;
    newid[uint32_t(e)] = 0;
  }
  int64_t N = 0;
  for (uint64_t i = 0; i <= maxid; ++i)
    if (newid[i] == 0) newid[i] = int32_t(N++);
  if (N >= (int64_t(1) << 31)) die("too many nodes for 32-bit IDs");

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
  std::vector<uint64_t>().swap(edges);
  for (int64_t v = 0; v < N; ++v) off[v + 1] += off[v];
  write_graph(pos[1], N, off, src, outdeg);

  if (pos.size() > 2) {
    std::vector<int64_t> orig(N);
    for (uint64_t i = 0; i <= maxid; ++i)
      if (newid[i] >= 0) orig[newid[i]] = int64_t(i);
    FILE* m = open_or_die(pos[2], "wb");
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
              basename_noext(pos[1]).c_str(), (long long)N, (long long)M, raw, selfloops, dups, dangling, maxin, maxout,
              now() - t0);
  return 0;
}
