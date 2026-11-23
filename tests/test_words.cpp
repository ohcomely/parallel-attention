#include <cmath>
#include <map>

#include "bench/harness.hpp"
#include "check.hpp"

using namespace attn;

namespace {

using Words = std::array<double, num_phases>;

double c(int r, double N) { return (1.0 - 1.0 / r) * N; }  // all-gather / reduce-scatter among r

// Closed forms from the cost model (research notes §6), per rank.
Words expected_3d(const Shape& p, Dims g) {
  const double n = p.n, d1 = p.d1, d2 = p.d2;
  Words w{};
  w[int(Phase::Q)] = c(g.p2, n * d1 / (g.p1 * g.p3));
  w[int(Phase::K)] = c(g.p1, n * d1 / (g.p2 * g.p3));
  w[int(Phase::V)] = c(g.p1, n * d2 / (g.p2 * g.p3));
  w[int(Phase::S_rs)] = c(g.p3, n * n / (g.p1 * g.p2));
  w[int(Phase::S_ag)] = c(g.p3, n * n / (g.p1 * g.p2));
  w[int(Phase::O)] = c(g.p2, n * d2 / (g.p1 * g.p3));
  w[int(Phase::rowstat)] = 2 * 2 * c(g.p2, n / g.p1);
  return w;
}

Words expected_split(const Shape& p, Dims g, Role role) {
  const double n = p.n, d1 = p.d1, d2 = p.d2;
  Words w{};
  if (role == Role::producer) {
    w[int(Phase::Q)] = c(g.p2, n * d1 / (g.p1 * g.p3));
    w[int(Phase::K)] = c(g.p1, n * d1 / (g.p2 * g.p3));
    w[int(Phase::S_rs)] = c(g.p3, n * n / (g.p1 * g.p2));
  } else {
    w[int(Phase::S_handoff)] = n * n / (g.p1 * g.p2 * g.p3);
    w[int(Phase::S_ag)] = c(g.p3, n * n / (g.p1 * g.p2));
    w[int(Phase::V)] = c(g.p1, n * d2 / (g.p2 * g.p3));
    w[int(Phase::O)] = c(g.p2, n * d2 / (g.p1 * g.p3));
    w[int(Phase::rowstat)] = 2 * 2 * c(g.p2, n / g.p1);
  }
  return w;
}

Words expected_ring(const Shape& p, int P) {
  Words w{};
  w[int(Phase::ring_kv)] = c(P, double(p.n) * (p.d1 + p.d2));
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rc;
  {
    test::Suite s;
    std::map<std::string, Words> split_by_grid;  // words must not depend on T
    for (const test::Case& c : test::sanity_cases(s.size)) {
      auto alg = c.build(s);
      if (!alg) continue;
      bench::run_timed(*alg, MPI_COMM_WORLD);
      const Timers& t = alg->timers();
      const Plan& p = alg->plan();
      const Words want = p.alg == "ring"    ? expected_ring(p.shape, s.size)
                         : p.alg == "split" ? expected_split(p.shape, p.grid, alg->role())
                                            : expected_3d(p.shape, p.grid);
      bool ok = true;
      for (int ph = 0; ph < num_phases; ++ph) {
        const bool match = std::abs(t.words[ph] - want[ph]) <= 1e-9 * std::max(1.0, want[ph]);
        ok = ok && match;
        s.expect(match, c.describe() + " role " + name(alg->role()) + " phase " +
                            std::string(phase_names[ph]) + ": got " + std::to_string(t.words[ph]) + ", want " +
                            std::to_string(want[ph]));
      }
      double words = 0, busiest = 0;
      for (int ph = 0; ph < num_phases; ++ph)
        if (ph != int(Phase::rowstat)) words += t.words[ph];
      MPI_Allreduce(&words, &busiest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const double predicted = p.words;
      s.expect(std::abs(busiest - predicted) <= 1e-9 * predicted,
               c.describe() + ": busiest rank received " + std::to_string(busiest) + " words, cost model says " +
                   std::to_string(predicted));
      if (p.alg == "split") {
        const std::string key = p.grid.str() + name(alg->role());
        if (!split_by_grid.count(key)) split_by_grid[key] = t.words;
        s.expect(split_by_grid[key] == t.words, c.describe() + ": words depend on T");
      }
      s.log((ok ? "ok    " : "FAIL  ") + c.describe());
    }
    rc = s.finish();
  }
  MPI_Finalize();
  return rc;
}
