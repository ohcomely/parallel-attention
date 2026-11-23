#include "attn/registry.hpp"

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "algs/algs.hpp"
#include "attn/costmodel.hpp"

namespace attn {

namespace {
constexpr std::array<const AlgorithmSpec*, 4> table{&algs::attn2d, &algs::attn3d, &algs::split, &algs::ring};
}

std::span<const AlgorithmSpec* const> algorithms() { return table; }

const AlgorithmSpec& algorithm(std::string_view name) {
  std::string known;
  for (const AlgorithmSpec* s : table) {
    if (s->name == name) return *s;
    known += (known.empty() ? "" : ", ") + std::string(s->name);
  }
  throw std::invalid_argument("unknown algorithm '" + std::string(name) + "' (known: " + known + ")");
}

std::optional<Dims> best_grid(const AlgorithmSpec& spec, const Shape& s, int P, int T) {
  if (P % spec.groups) return std::nullopt;
  std::optional<Dims> best;
  double best_words = 0;
  for (Dims g : all_grids(P / spec.groups)) {
    if (!spec.allows(g)) continue;
    try {
      spec.validate(s, g, T);
    } catch (const std::invalid_argument&) {
      continue;
    }
    const double w = spec.words(s, g, P);
    const bool tie = best && std::abs(w - best_words) <= 1e-9 * best_words;
    if (!best || (!tie && w < best_words) || (tie && std::tie(g.p3, g.p1) < std::tie(best->p3, best->p1))) {
      best = g;
      best_words = w;
    }
  }
  return best;
}

}  // namespace attn
