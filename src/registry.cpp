#include "attn/registry.hpp"

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

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

std::vector<int64_t> pipeline_rows(int64_t n, int T, const std::vector<int>& blocks) {
  if (blocks.empty()) {
    if (T < 1 || n % T) return {};
    return std::vector<int64_t>(T, n / T);
  }
  int64_t sum = 0;
  for (int w : blocks) {
    if (w < 1) return {};
    sum += w;
  }
  if (n % sum) return {};
  std::vector<int64_t> rows;
  for (int w : blocks) rows.push_back(w * (n / sum));
  return rows;
}

std::optional<Dims> best_grid(const AlgorithmSpec& spec, const Shape& s, int P, int T) {
  return best_grid(spec, s, P, pipeline_rows(s.n, T, {}));
}

std::optional<Dims> best_grid(const AlgorithmSpec& spec, const Shape& s, int P, std::span<const int64_t> rows) {
  if (P % spec.groups || rows.empty()) return std::nullopt;
  std::optional<Dims> best;
  double best_words = 0;
  for (Dims g : all_grids(P / spec.groups)) {
    if (!spec.allows(g)) continue;
    try {
      spec.validate(s, g, rows);
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
