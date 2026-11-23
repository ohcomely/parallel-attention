#pragma once
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace attn {

// Q, K: n x d1.  V, O: n x d2.  A zero d2 means d2 = d1.
struct Shape {
  int64_t n = 0, d1 = 0, d2 = 0;
};

// A rectangle of a global matrix: rows [row0, row0 + rows), columns [col0, col0 + cols).
struct Block {
  int64_t row0 = 0, rows = 0, col0 = 0, cols = 0;
};

inline void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

// MPI counts are int.
inline int to_int(int64_t x) {
  require(x >= 0 && x <= INT_MAX, "count exceeds int range: " + std::to_string(x));
  return int(x);
}

// a / b, or std::invalid_argument naming `what` when b does not divide a. Sizes are never truncated.
inline int64_t exact_div(int64_t a, int64_t b, std::string_view what) {
  require(b > 0 && a % b == 0,
          std::string(what) + ": " + std::to_string(a) + " not divisible by " + std::to_string(b));
  return a / b;
}

}  // namespace attn
