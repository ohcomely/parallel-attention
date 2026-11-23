#include "attn/costmodel.hpp"

#include <cmath>
#include <limits>

namespace attn {

namespace {

double c(int r, double N) { return (1.0 - 1.0 / r) * N; }

double G(double t, double d) { return t <= d * d * d ? 3 * std::cbrt(t * t) : t / d + 2 * std::sqrt(t * d); }

}  // namespace

double thm_a_words(const Shape& p, int P) {
  const double n = p.n, d1 = p.d1, d2 = p.d2;
  return G(n * n * d1 / P, d1) + G(n * n * d2 / P, d2) - (2 * n * d1 + 2 * n * d2 + 2 * n * n) / P;
}

double thm_b_words(const Shape& p, int P) {
  if (p.d1 != p.d2) return std::numeric_limits<double>::quiet_NaN();
  const double n = p.n, d = p.d1, F = n * n * d / P;
  if (P <= n * n / (4 * d * d)) return 4 * n * d / std::sqrt(double(P)) - 4 * n * d / P;
  return 3 * std::cbrt(4 * F * F) - (4 * n * d + n * n) / P;
}

double words_3d(const Shape& p, Dims g) {
  const double n = p.n, d1 = p.d1, d2 = p.d2;
  return c(g.p2, n * d1 / (g.p1 * g.p3)) + c(g.p2, n * d2 / (g.p1 * g.p3)) + c(g.p1, n * d1 / (g.p2 * g.p3)) +
         c(g.p1, n * d2 / (g.p2 * g.p3)) + 2 * c(g.p3, n * n / (g.p1 * g.p2));
}

SplitWords words_split(const Shape& p, Dims g) {
  const double n = p.n, d1 = p.d1, d2 = p.d2;
  const double s_over_k = c(g.p3, n * n / (g.p1 * g.p2));
  return {c(g.p2, n * d1 / (g.p1 * g.p3)) + c(g.p1, n * d1 / (g.p2 * g.p3)) + s_over_k,
          n * n / g.size() + s_over_k + c(g.p1, n * d2 / (g.p2 * g.p3)) + c(g.p2, n * d2 / (g.p1 * g.p3))};
}

double words_ring(const Shape& p, int P) { return c(P, double(p.n) * (p.d1 + p.d2)); }

std::vector<Dims> all_grids(int ranks) {
  std::vector<Dims> out;
  for (int p1 = 1; p1 <= ranks; ++p1)
    for (int p2 = 1; p1 * p2 <= ranks; ++p2)
      if (ranks % (p1 * p2) == 0) out.push_back({p1, p2, ranks / (p1 * p2)});
  return out;
}

}  // namespace attn
