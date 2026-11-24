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
  const int T = o.blocks.empty() ? o.T : int(o.blocks.size());
  require(o.blocks.empty() || o.T == 1 || o.T == T, "T and blocks disagree (T = number of blocks)");
  require(spec.pipelined || (T == 1 && o.blocks.empty()), what + " is not pipelined: T must be 1, no blocks");
  for (int w : o.blocks) require(w >= 1, "block sizes must be positive");
  const std::vector<int64_t> rows = pipeline_rows(s.n, T, o.blocks);  // empty if they do not divide n
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
    require(!rows.empty(), "pipeline blocks do not divide n = " + std::to_string(s.n));
    spec.validate(s, g, rows);
  } else {
    const auto best = best_grid(spec, s, P, rows);
    require(best.has_value(), "no valid " + what + " grid for n=" + std::to_string(s.n) + " d1=" +
                                  std::to_string(s.d1) + " d2=" + std::to_string(s.d2) + " on " +
                                  std::to_string(P) + " ranks with T=" + std::to_string(T));
    g = *best;
  }
  return {what, s, P, g, T, rows, o.blocks, o.map, spec.words(s, g, P)};
}

std::unique_ptr<Attention> make(std::string_view alg, MPI_Comm comm, Shape shape, const Options& opts) {
  int P;
  MPI_Comm_size(comm, &P);
  const Plan p = plan(alg, shape, P, opts);
  return algorithm(p.alg).make(p, comm);
}

}  // namespace attn
