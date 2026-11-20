#pragma once
#include <cstdint>

#include "attn/layout.hpp"
#include "attn/matrix.hpp"

namespace attn {

enum class ArrayId : uint64_t { Q = 1, K = 2, V = 3 };

// Uniform in [-0.1, 0.1), a pure function of its arguments so any rank can generate any block.
float generated_value(ArrayId id, int64_t row, int64_t col, uint64_t seed);

template <class T = float>
Mat<T> generate(ArrayId id, Block b, uint64_t seed) {
  Mat<T> m(b.rows, b.cols);
  for (int64_t r = 0; r < b.rows; ++r)
    for (int64_t c = 0; c < b.cols; ++c) m(r, c) = T(generated_value(id, b.row0 + r, b.col0 + c, seed));
  return m;
}

}  // namespace attn
