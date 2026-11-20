#include <cstdio>
#include <map>

#include "attn/bench.hpp"
#include "attn/reference.hpp"
#include "check.hpp"

using namespace attn;

namespace {

Mat<double> broadcast_reference(const Problem& p, int rank) {
  Mat<double> ref(p.n, p.d2);
  if (rank == 0) ref = reference_output(p, {0, p.n, 0, p.d2});
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
    for (const Config& c : test::sanity_configs(s.size)) {
      if (!refs.count(c.prob.d2)) refs[c.prob.d2] = broadcast_reference(c.prob, s.rank);
      Timers t;
      std::unique_ptr<Algorithm> alg;
      try {
        alg = make_algorithm(c, MPI_COMM_WORLD, t);
      } catch (const std::invalid_argument& e) {
        s.log("skip  " + test::describe(c) + ": " + e.what());
        continue;
      }
      double err = 0;
      for (int rep = 0; rep < 2; ++rep) {  // second run checks that state is reset between repetitions
        run_timed(*alg, t, MPI_COMM_WORLD);
        err = std::max(err, max_rel_error(*alg, c.prob, MPI_COMM_WORLD, &refs[c.prob.d2]));
      }
      const bool ok = err < 1e-4;
      s.expect(ok, test::describe(c) + " rel err " + std::to_string(err));
      char line[160];
      std::snprintf(line, sizeof line, "%s %-40s rel err %.2e", ok ? "ok   " : "FAIL ", test::describe(c).c_str(),
                    err);
      s.log(line);
    }
    rc = s.finish();
  }
  MPI_Finalize();
  return rc;
}
