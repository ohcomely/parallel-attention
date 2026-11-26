#include "bench/harness.hpp"

#include <algorithm>

#include "attn/device.hpp"

namespace attn::bench {

void run_timed(Attention& alg, MPI_Comm comm) {
  Timers& t = alg.timers();
  t.reset();
  MPI_Barrier(comm);
  const double t0 = now();
  t.origin = t0;
  alg.run();
  device_sync();
  const double t1 = now();
  t.total = t1 - t0;
  MPI_Barrier(comm);
  t.idle_tail = now() - t1;
}

std::vector<Sample> gather_samples(const Attention& alg, MPI_Comm comm) {
  constexpr int fields = 6;
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  const Timers& t = alg.timers();
  const double mine[fields] = {t.total, t.compute, t.comm, t.idle_handoff, t.idle_tail, double(int(alg.role()))};
  std::vector<double> all(rank == 0 ? size * fields : 0);
  MPI_Gather(mine, fields, MPI_DOUBLE, all.data(), fields, MPI_DOUBLE, 0, comm);
  std::vector<Sample> out;
  for (int r = 0; r < int(all.size()) / fields; ++r) {
    const double* x = &all[r * fields];
    out.push_back({x[0], x[1], x[2], x[3], x[4], Role(int(x[5]))});
  }
  return out;
}

int slowest(const std::vector<Sample>& s, Role r) {
  int best = -1;
  for (int i = 0; i < int(s.size()); ++i)
    if ((r == Role::all || s[i].role == r) && (best < 0 || s[i].total > s[best].total)) best = i;
  return best;
}

Measurement measure(Attention& alg, MPI_Comm comm, const Repetitions& r, bool timeline,
                    const std::function<void(int, const std::vector<Sample>&)>& on_rep) {
  require(r.reps > 0 && r.warmup >= 0, "reps must be positive and warmup non-negative");
  int rank;
  MPI_Comm_rank(comm, &rank);
  for (int w = 0; w < r.warmup; ++w) run_timed(alg, comm);

  Measurement m;
  std::vector<double> totals;
  const int max_reps = std::max(r.max_reps, r.reps);
  for (int rep = 0, more = 1; more; ++rep) {
    alg.timers().record = timeline && rep == 0;
    run_timed(alg, comm);
    const auto samples = gather_samples(alg, comm);
    if (rank == 0) totals.push_back(samples[slowest(samples)].total);
    on_rep(rep, samples);
    if (rank == 0)
      more = rep + 1 < r.reps || (rep + 1 < max_reps && relative_halfwidth(median_ci(totals)) > r.ci_target);
    MPI_Bcast(&more, 1, MPI_INT, 0, comm);
    m.reps = rep + 1;
  }
  alg.timers().record = false;
  if (rank == 0) m.t_total_max = median_ci(totals);
  return m;
}

}  // namespace attn::bench
