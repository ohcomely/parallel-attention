#include "attn/reference.hpp"

#include <cmath>
#include <vector>

#include "attn/kernels.hpp"
#include "attn/rng.hpp"

namespace attn {

Mat<double> reference_output(const Problem& p, Block b) {
  const auto q = generate<double>(ArrayId::Q, {b.row0, b.rows, 0, p.d1}, p.seed);
  const auto k = generate<double>(ArrayId::K, {0, p.n, 0, p.d1}, p.seed);
  const auto v = generate<double>(ArrayId::V, {0, p.n, b.col0, b.cols}, p.seed);
  Mat<double> s(b.rows, p.n), o(b.rows, b.cols);
  std::vector<double> max(b.rows), sum(b.rows);

  gemm_nt(b.rows, p.n, p.d1, 1.0 / std::sqrt(double(p.d1)), q.ptr(), p.d1, k.ptr(), p.d1, 0.0, s.ptr(), p.n);
  row_max(b.rows, p.n, s.ptr(), p.n, max.data());
  exp_shift_rowsum(b.rows, p.n, s.ptr(), p.n, max.data(), sum.data());
  gemm_nn(b.rows, b.cols, p.n, 1.0, s.ptr(), p.n, v.ptr(), b.cols, 0.0, o.ptr(), b.cols);
  divide_rows(b.rows, b.cols, o.ptr(), b.cols, sum.data());
  return o;
}

}  // namespace attn
