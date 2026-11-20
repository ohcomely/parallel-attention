#pragma once
#include <vector>

// Median with a nonparametric (order-statistics) 95% confidence interval; mirrored by tools/stats.py.
namespace attn {

struct MedianCI {
  double median, lo, hi;
  bool exact;  // false when there are too few samples (< 6) for 95% coverage; then [lo, hi] = [min, max]
};

MedianCI median_ci(std::vector<double> x);

// CI half-width relative to the median: max(median - lo, hi - median) / median.
double relative_halfwidth(const MedianCI& m);

}  // namespace attn
