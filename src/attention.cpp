#include "attn/attention.hpp"

#include "attn/registry.hpp"

namespace attn {

const char* name(Role r) {
  switch (r) {
    case Role::producer: return "producer";
    case Role::consumer: return "consumer";
    default: return "all";
  }
}

const char* name(Array a) {
  switch (a) {
    case Array::Q: return "Q";
    case Array::K: return "K";
    default: return "V";
  }
}

Plan plan(std::string_view alg, Shape s, int P, const Options& o) {
  const AlgorithmSpec& spec = algorithm(alg);
  const std::string what(spec.name);
  if (s.d2 == 0) s.d2 = s.d1;
  require(s.n > 0 && s.d1 > 0 && s.d2 > 0, "n, d1, d2 must be positive");
  require(o.T >= 1, "T must be positive");
  require(spec.pipelined || o.T == 1, what + " is not pipelined: T must be 1");
  require(P >= spec.groups && P % spec.groups == 0,
          what + " needs a multiple of " + std::to_string(spec.groups) + " ranks, have " + std::to_string(P));
  coords_of(0, {}, o.map);  // validates the map

  Dims g;
  if (o.grid) {
    g = *o.grid;
    require(g.size() * spec.groups == P, what + " grid " + g.str() + " needs " +
                                             std::to_string(g.size() * spec.groups) + " ranks, have " +
                                             std::to_string(P));
    require(spec.allows(g), what + " does not accept grid " + g.str());
    spec.validate(s, g, o.T);
  } else {
    const auto best = best_grid(spec, s, P, o.T);
    require(best.has_value(), "no valid " + what + " grid for n=" + std::to_string(s.n) + " d1=" +
                                  std::to_string(s.d1) + " d2=" + std::to_string(s.d2) + " on " +
                                  std::to_string(P) + " ranks with T=" + std::to_string(o.T));
    g = *best;
  }
  return {what, s, P, g, o.T, o.map, spec.words(s, g, P)};
}

void device_init(MPI_Comm comm) {
  static bool done = false;
  if (done) return;
  MPI_Comm node;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int local;
  MPI_Comm_rank(node, &local);
  MPI_Comm_free(&node);
  select_device(local);
  done = true;
}

std::unique_ptr<Attention> make(std::string_view alg, MPI_Comm comm, Shape shape, const Options& opts) {
  int P;
  MPI_Comm_size(comm, &P);
  const Plan p = plan(alg, shape, P, opts);
  device_init(comm);
  return algorithm(p.alg).make(p, comm);
}

}  // namespace attn
