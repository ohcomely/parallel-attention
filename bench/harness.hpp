#pragma once
#include <mpi.h>

#include <functional>
#include <vector>

#include "attn/attention.hpp"
#include "bench/stats.hpp"

// Measurement protocol (Hoefler & Belli): opening barrier, per-rank timers, closing barrier, max over ranks;
// warm-up runs, then repetitions until the median's 95% CI is tight enough.
namespace attn::bench {

// One repetition: opening barrier, run, own finish (t_total), closing barrier (t_idle_tail).
void run_timed(Attention& alg, MPI_Comm comm);

struct Sample {
  double total, compute, comm, idle_handoff, idle_tail;
  Role role;
  double other() const { return total - compute - comm - idle_handoff; }
};

// Every rank's timers of the last run, on rank 0 of `comm` (empty elsewhere). Collective.
std::vector<Sample> gather_samples(const Attention& alg, MPI_Comm comm);

// Index of the slowest rank with the given role (any role for Role::all); -1 if there is none.
int slowest(const std::vector<Sample>& s, Role r = Role::all);

struct Repetitions {
  int warmup = 5, reps = 20, max_reps = 100;
  double ci_target = 0.05;  // stop once the CI half-width is within this fraction of the median
};

struct Measurement {
  MedianCI t_total_max;  // over repetitions of the slowest rank's t_total (rank 0 only)
  int reps = 0;
};

// Warm-up, then at least `reps` repetitions, continuing until the CI target or `max_reps`. After every repetition
// `on_rep(rep, samples)` is called on every rank (samples are empty except on rank 0). Timeline events are recorded
// for repetition 0 when `timeline` is set. Collective.
Measurement measure(Attention& alg, MPI_Comm comm, const Repetitions& r, bool timeline,
                    const std::function<void(int rep, const std::vector<Sample>&)>& on_rep);

}  // namespace attn::bench
