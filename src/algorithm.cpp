#include "attn/algorithm.hpp"

namespace attn {

const char* name(Role r) {
  switch (r) {
    case Role::producer: return "producer";
    case Role::consumer: return "consumer";
    default: return "all";
  }
}

std::unique_ptr<Algorithm> make_algorithm(const Config& c, MPI_Comm comm, Timers& t) {
  require(c.prob.n > 0 && c.prob.d1 > 0 && c.prob.d2 > 0, "n, d1, d2 must be positive");
  if (c.alg == "2d") {
    require(c.grid.p3 == 1, "2d needs p3 = 1");
    return make_attn3d(c, comm, t);
  }
  if (c.alg == "3d") return make_attn3d(c, comm, t);
  if (c.alg == "split") return make_attn_split(c, comm, t);
  if (c.alg == "ring") return make_attn_ring(c, comm, t);
  throw std::invalid_argument("unknown algorithm: " + c.alg);
}

}  // namespace attn
