// Binary in-edge CSR graph shared by all PageRank implementations.
// File layout: char[8] "PRGRAPH1", int64 N, int64 M, int64 off[N+1], int32 src[M], int32 outdeg[N]
// Row v of the CSR lists the sources u of the edges u -> v.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

namespace prg {

static const char MAGIC[8] = {'P', 'R', 'G', 'R', 'A', 'P', 'H', '1'};

struct Header {
  int64_t N = 0, M = 0;
};

inline long long header_bytes() { return 8 + 2 * sizeof(int64_t); }
inline long long off_pos() { return header_bytes(); }
inline long long src_pos(const Header& h) { return off_pos() + (h.N + 1) * (long long)sizeof(int64_t); }
inline long long deg_pos(const Header& h) { return src_pos(h) + h.M * (long long)sizeof(int32_t); }

inline void die(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(1);
}

inline FILE* open_or_die(const std::string& path, const char* mode) {
  FILE* f = std::fopen(path.c_str(), mode);
  if (!f) die("cannot open " + path);
  return f;
}

inline void read_at(FILE* f, long long pos, void* buf, size_t bytes) {
  if (fseeko(f, pos, SEEK_SET) != 0) die("seek failed");
  char* p = static_cast<char*>(buf);
  while (bytes > 0) {
    size_t chunk = bytes < (size_t(1) << 30) ? bytes : (size_t(1) << 30);
    size_t got = std::fread(p, 1, chunk, f);
    if (got != chunk) die("short read");
    p += got;
    bytes -= got;
  }
}

inline Header read_header(FILE* f) {
  char magic[8];
  read_at(f, 0, magic, 8);
  if (std::memcmp(magic, MAGIC, 8) != 0) die("bad graph file magic");
  Header h;
  int64_t nm[2];
  read_at(f, 8, nm, sizeof(nm));
  h.N = nm[0];
  h.M = nm[1];
  return h;
}

struct Graph {
  Header h;
  std::vector<int64_t> off;
  std::vector<int32_t> src;
  std::vector<int32_t> outdeg;
};

inline Graph load_full(const std::string& path) {
  FILE* f = open_or_die(path, "rb");
  Graph g;
  g.h = read_header(f);
  g.off.resize(g.h.N + 1);
  g.src.resize(g.h.M);
  g.outdeg.resize(g.h.N);
  read_at(f, off_pos(), g.off.data(), g.off.size() * sizeof(int64_t));
  read_at(f, src_pos(g.h), g.src.data(), g.src.size() * sizeof(int32_t));
  read_at(f, deg_pos(g.h), g.outdeg.data(), g.outdeg.size() * sizeof(int32_t));
  std::fclose(f);
  return g;
}

inline void write_graph(const std::string& path, int64_t N, const std::vector<int64_t>& off,
                        const std::vector<int32_t>& src, const std::vector<int32_t>& outdeg) {
  FILE* f = open_or_die(path, "wb");
  int64_t nm[2] = {N, (int64_t)src.size()};
  std::fwrite(MAGIC, 1, 8, f);
  std::fwrite(nm, sizeof(int64_t), 2, f);
  std::fwrite(off.data(), sizeof(int64_t), off.size(), f);
  std::fwrite(src.data(), sizeof(int32_t), src.size(), f);
  std::fwrite(outdeg.data(), sizeof(int32_t), outdeg.size(), f);
  std::fclose(f);
}

inline void write_vector(const std::string& path, const std::vector<double>& r) {
  FILE* f = open_or_die(path, "wb");
  std::fwrite(r.data(), sizeof(double), r.size(), f);
  std::fclose(f);
}

inline double now() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Rough DRAM traffic of one pull-style iteration, used to report effective bandwidth.
inline double bytes_per_iter(int64_t N, int64_t M, int off_bytes) {
  return double(M) * (4 + 8) + double(N) * (off_bytes + 36);
}

// Shared CSV header for every implementation.
// t_comp / t_comm / t_stage are totals over the run (max over ranks); t_stage is the host<->device copy time
// of the multi-GPU halo exchange (empty for CPU runs). t_h2d / t_d2h are the one-off graph upload / result download.
inline const char* csv_header() {
  return "exp,impl,graph,N,M,nodes,ranks,threads,cores,part,comm,rep,iters,t_total,t_iter,t_comp,t_comm,"
         "imbalance,comm_bytes_iter,resid,sumr,t_load,t_h2d,t_d2h,gbps,t_stage";
}

struct Args {
  std::vector<std::string> kv;
  Args(int argc, char** argv) : kv(argv + 1, argv + argc) {}
  std::string get(const std::string& key, const std::string& def) const {
    for (size_t i = 0; i + 1 < kv.size(); ++i)
      if (kv[i] == "--" + key) return kv[i + 1];
    return def;
  }
  long long geti(const std::string& key, long long def) const { return std::atoll(get(key, std::to_string(def)).c_str()); }
  double getd(const std::string& key, double def) const {
    std::string s = get(key, "");
    return s.empty() ? def : std::atof(s.c_str());
  }
};

inline std::string basename_noext(const std::string& p) {
  size_t s = p.find_last_of("/\\");
  std::string b = s == std::string::npos ? p : p.substr(s + 1);
  size_t d = b.find_last_of('.');
  return d == std::string::npos ? b : b.substr(0, d);
}

}  // namespace prg
