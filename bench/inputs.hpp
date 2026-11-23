#pragma once
#include <cstdint>

#include "attn/attention.hpp"
#include "attn/matrix.hpp"

// Benchmark inputs: a counter-based hash of (array, row, column, seed), uniform in [-0.1, 0.1).
// Any rank can produce any block without communicating; values do not affect dense-attention time.
namespace attn::bench {

float generated_value(Array a, int64_t row, int64_t col, uint64_t seed);

template <class T = float>
Mat<T> generate(Array a, Block b, uint64_t seed) {
  Mat<T> m(b.rows, b.cols);
  for (int64_t r = 0; r < b.rows; ++r)
    for (int64_t c = 0; c < b.cols; ++c) m(r, c) = T(generated_value(a, b.row0 + r, b.col0 + c, seed));
  return m;
}

inline void fill_generated(Attention& alg, uint64_t seed) {
  fill_inputs(alg, [seed](Array a, int64_t r, int64_t c) { return generated_value(a, r, c, seed); });
}

}  // namespace attn::bench
