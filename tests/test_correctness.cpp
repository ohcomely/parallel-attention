#include <cstdio>
#include <map>

#include "bench/harness.hpp"
#include "bench/reference.hpp"
#include "check.hpp"

using namespace attn;

namespace {

Mat<double> broadcast_reference(const Shape& s, int rank) {
  Mat<double> ref(s.n, s.d2);
  if (rank == 0) ref = bench::reference_output(s, test::sanity_seed, {0, s.n, 0, s.d2});
  MPI_Bcast(ref.ptr(), int(ref.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return ref;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rc;
  {
    test::Suite s;
    std::map<int64_t, Mat<double>> refs;  // keyed by d2
    for (const test::Case& c : test::sanity_cases(s.size)) {
      if (!refs.count(c.shape.d2)) refs[c.shape.d2] = broadcast_reference(c.shape, s.rank);
      auto alg = c.build(s);
      if (!alg) continue;
      double err = 0;
      for (int rep = 0; rep < 2; ++rep) {  // second run checks that state is reset between repetitions
        bench::run_timed(*alg, MPI_COMM_WORLD);
        err = std::max(err, bench::max_rel_error(*alg, test::sanity_seed, MPI_COMM_WORLD, &refs[c.shape.d2]));
      }
      const bool ok = err < 1e-4;
      s.expect(ok, c.describe() + " rel err " + std::to_string(err));
      char line[160];
      std::snprintf(line, sizeof line, "%s %-40s rel err %.2e", ok ? "ok   " : "FAIL ", c.describe().c_str(), err);
      s.log(line);
    }

    // The default (cost-model) grid gives the same answer through the public entry point.
    for (const char* name : {"2d", "ring"}) {
      const test::Case c{name, {test::sanity_n, test::sanity_d, 0}, {}};
      auto alg = c.build(s);
      if (!alg) continue;
      bench::run_timed(*alg, MPI_COMM_WORLD);
      const double err = bench::max_rel_error(*alg, test::sanity_seed, MPI_COMM_WORLD, &refs[test::sanity_d]);
      s.expect(err < 1e-4, c.describe() + " (default grid) rel err " + std::to_string(err));
    }
    rc = s.finish();
  }
  MPI_Finalize();
  return rc;
}
