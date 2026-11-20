#include "attn/bench.hpp"

#include <algorithm>
#include <cmath>

#include "attn/reference.hpp"

namespace attn {

void run_timed(Algorithm& alg, Timers& t, MPI_Comm comm) {
  t.reset();
  MPI_Barrier(comm);
  const double t0 = now();
  t.origin = t0;
  alg.run();
  const double t1 = now();
  t.total = t1 - t0;
  MPI_Barrier(comm);
  t.idle_tail = now() - t1;
}

double max_rel_error(const Algorithm& alg, const Problem& p, MPI_Comm comm, const Mat<double>* full) {
  double local[2] = {0, 0};  // max abs diff, max abs reference
  for (const auto& [b, data] : alg.output()) {
    const Mat<double> ref = full ? Mat<double>{} : reference_output(p, b);
    for (int64_t r = 0; r < b.rows; ++r)
      for (int64_t c = 0; c < b.cols; ++c) {
        const double expected = full ? (*full)(b.row0 + r, b.col0 + c) : ref(r, c);
        const double diff = std::abs(double(data[r * b.cols + c]) - expected);
        local[0] = std::isfinite(diff) ? std::max(local[0], diff) : INFINITY;
        local[1] = std::max(local[1], std::abs(expected));
      }
  }
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, comm);
  return global[0] / global[1];
}

}  // namespace attn
