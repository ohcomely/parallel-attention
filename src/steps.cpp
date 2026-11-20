#include "steps.hpp"

#include "attn/kernels.hpp"

namespace attn {

void softmax_v_reduce(Comm& cj, Mat<float>& s, const Mat<float>& v, Mat<float>& o_partial, std::vector<float>& max,
                      std::vector<float>& sum, float* o_owned, Timers& t) {
  const int64_t rows = s.rows, keys = s.cols, cols = v.cols, owned = rows / cj.size();

  { Timed _(t, "rowmax"); row_max(rows, keys, s.ptr(), keys, max.data()); }
  cj.allreduce(max.data(), rows, Op::max, Phase::rowstat);
  { Timed _(t, "exp"); exp_shift_rowsum(rows, keys, s.ptr(), keys, max.data(), sum.data()); }
  cj.allreduce(sum.data(), rows, Op::sum, Phase::rowstat);
  { Timed _(t, "gemm_o"); gemm_nn(rows, cols, keys, 1.f, s.ptr(), keys, v.ptr(), cols, 0.f, o_partial.ptr(), cols); }
  cj.reduce_scatter(o_partial.ptr(), o_owned, owned * cols, Phase::O);
  { Timed _(t, "normalize"); divide_rows(owned, cols, o_owned, cols, sum.data() + cj.rank() * owned); }
}

}  // namespace attn
