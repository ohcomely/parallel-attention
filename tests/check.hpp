#pragma once
#include <mpi.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "attn/algorithm.hpp"
#include "attn/costmodel.hpp"

namespace attn::test {

constexpr int64_t sanity_n = 2048, sanity_d = 512;

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

inline std::string describe(const Config& c) {
  std::string s = c.alg + " n=" + std::to_string(c.prob.n) + " d1=" + std::to_string(c.prob.d1) +
                  " d2=" + std::to_string(c.prob.d2);
  if (c.alg != "ring") s += " grid=" + c.grid.str();
  if (c.alg == "split") s += " T=" + std::to_string(c.T);
  return s;
}

// Every configuration of the sanity check that is valid for P ranks.
inline std::vector<Config> sanity_configs(int P, bool include_d2 = true) {
  std::vector<Config> out;
  const Problem eq{sanity_n, sanity_d, sanity_d, 7};
  const Problem uneq{sanity_n, sanity_d, 128, 7};
  for (const Problem& p : include_d2 ? std::vector{eq, uneq} : std::vector{eq})
    for (Dims d : all_grids(P)) out.push_back({d.p3 == 1 ? "2d" : "3d", p, d, 1, "kji"});
  out.push_back({"ring", eq, {}, 1, "kji"});
  if (P % 2 == 0)
    for (Dims d : all_grids(P / 2))
      for (int T : {1, 2, 4}) out.push_back({"split", eq, d, T, "kji"});
  return out;
}

}  // namespace attn::test
