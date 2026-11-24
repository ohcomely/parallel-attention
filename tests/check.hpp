#pragma once
#include <mpi.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "attn/attention.hpp"
#include "attn/costmodel.hpp"
#include "bench/inputs.hpp"

namespace attn::test {

constexpr int64_t sanity_n = 2048, sanity_d = 512;
constexpr uint64_t sanity_seed = 7;

struct Suite {
  int rank = 0, size = 1, failures = 0, passes = 0;

  Suite() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  void expect(bool ok, const std::string& what) {
    ok ? ++passes : ++failures;
    if (!ok) std::fprintf(stderr, "[rank %d] FAIL %s\n", rank, what.c_str());
  }

  void log(const std::string& msg) const {
    if (rank == 0) std::printf("%s\n", msg.c_str());
  }

  int finish() {
    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    log(total ? "FAILED (" + std::to_string(total) + " failures summed over ranks)"
              : "PASSED (" + std::to_string(passes) + " checks on rank 0)");
    return total ? 1 : 0;
  }
};

struct Case {
  std::string alg;
  Shape shape;
  Options opts;

  std::string describe() const {
    std::string s = alg + " n=" + std::to_string(shape.n) + " d1=" + std::to_string(shape.d1) +
                    " d2=" + std::to_string(shape.d2);
    if (opts.grid) s += " grid=" + opts.grid->str();
    if (opts.T != 1) s += " T=" + std::to_string(opts.T);
    if (!opts.blocks.empty()) {
      s += " blocks=";
      for (size_t b = 0; b < opts.blocks.size(); ++b) s += (b ? "," : "") + std::to_string(opts.blocks[b]);
    }
    return s;
  }

  // Builds the algorithm with generated inputs; nullptr (and a log line) if the case does not fit P ranks.
  std::unique_ptr<Attention> build(const Suite& s) const {
    try {
      auto a = make(alg, MPI_COMM_WORLD, shape, opts);
      bench::fill_generated(*a, sanity_seed);
      return a;
    } catch (const std::invalid_argument& e) {
      s.log("skip  " + describe() + ": " + e.what());
      return nullptr;
    }
  }
};

// Every configuration of the sanity check for P ranks: all 2D/3D grids (also with d2 != d1), Ring, and Split
// with T in {1, 2, 4} on every group grid.
inline std::vector<Case> sanity_cases(int P, bool include_d2 = true) {
  std::vector<Case> out;
  const Shape eq{sanity_n, sanity_d, sanity_d};
  const Shape uneq{sanity_n, sanity_d, 128};
  for (const Shape& s : include_d2 ? std::vector{eq, uneq} : std::vector{eq})
    for (Dims d : all_grids(P)) out.push_back({d.p3 == 1 ? "2d" : "3d", s, {.grid = d}});
  out.push_back({"ring", eq, {}});
  if (P % 2 == 0)
    for (Dims d : all_grids(P / 2)) {
      for (int T : {1, 2, 4}) out.push_back({"split", eq, {.grid = d, .T = T}});
      out.push_back({"split", eq, {.grid = d, .blocks = {1, 3, 3, 1}}});  // unequal pipeline blocks
    }
  return out;
}

}  // namespace attn::test
