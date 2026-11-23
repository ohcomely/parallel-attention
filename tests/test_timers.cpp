#include <cmath>
#include <cstdio>

#include "bench/harness.hpp"
#include "check.hpp"

using namespace attn;

namespace {

double timed_scope_cost() {
  Timers t;
  constexpr int reps = 200000;
  const double t0 = now();
  for (int r = 0; r < reps; ++r) Timed _(t, "probe");
  return (now() - t0) / reps;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rc;
  {
    test::Suite s;
    const double scope_cost = timed_scope_cost();
    s.log("cost of one timed scope: " + std::to_string(scope_cost * 1e9) + " ns");
    for (const test::Case& c : test::sanity_cases(s.size, false)) {
      if (c.alg == "split" && c.opts.T != 2) continue;
      auto alg = c.build(s);
      if (!alg) continue;
      const Timers& t = alg->timers();
      alg->timers().record = true;
      for (int rep = 0; rep < 3; ++rep) bench::run_timed(*alg, MPI_COMM_WORLD);

      const double parts = t.compute + t.comm + t.idle_handoff;
      const double gap = std::abs(t.total - parts) / t.total;
      s.expect(gap <= 0.02, c.describe() + ": components miss total by " + std::to_string(100 * gap) + "%");

      double worst_overhead = t.compute_calls * scope_cost / t.compute;
      for (int ph = 0; ph < num_phases; ++ph)
        if (t.words[ph] > 0)
          worst_overhead = std::max(worst_overhead, t.phase_calls[ph] * scope_cost / t.phase_time[ph]);
      s.expect(worst_overhead < 0.05, c.describe() + ": timer overhead " +
                                          std::to_string(100 * worst_overhead) + "% of a phase");

      bool timeline_ok = !t.events.empty();
      for (size_t e = 0; e < t.events.size(); ++e) {
        timeline_ok = timeline_ok && t.events[e].start <= t.events[e].end && t.events[e].end <= t.total + 1e-9;
        if (e) timeline_ok = timeline_ok && t.events[e - 1].end <= t.events[e].start + 1e-9;
      }
      s.expect(timeline_ok, c.describe() + ": timeline events not ordered");

      double worst_gap = 0, worst_overhead_all = 0;
      MPI_Reduce(&gap, &worst_gap, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&worst_overhead, &worst_overhead_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      char line[200];
      std::snprintf(line, sizeof line, "%-45s other/total %.3f%%  timer overhead %.3f%%", c.describe().c_str(),
                    100 * worst_gap, 100 * worst_overhead_all);
      s.log(line);
    }
    rc = s.finish();
  }
  MPI_Finalize();
  return rc;
}
