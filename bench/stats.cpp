#include "bench/stats.hpp"

#include <algorithm>
#include <cmath>

namespace attn::bench {

namespace {

// P(B <= k) for B ~ Binomial(n, 1/2)
double binom_half_cdf(int n, int k) {
  double term = std::pow(0.5, n), sum = 0;
  for (int i = 0; i <= k; ++i) {
    sum += term;
    term *= double(n - i) / (i + 1);
  }
  return sum;
}

}  // namespace

MedianCI median_ci(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  const int n = int(x.size());
  const double median = n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
  // [x_j, x_{n-1-j}] covers the median with probability 1 - 2 P(B <= j); take the largest j with coverage >= 95%.
  int j = -1;
  while (j + 1 < n / 2 && 2 * binom_half_cdf(n, j + 1) <= 0.05) ++j;
  if (j < 0) return {median, x.front(), x.back(), false};
  return {median, x[j], x[n - 1 - j], true};
}

double relative_halfwidth(const MedianCI& m) { return std::max(m.median - m.lo, m.hi - m.median) / m.median; }

}  // namespace attn::bench
