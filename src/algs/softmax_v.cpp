#include "softmax_v.hpp"

#include "attn/kernels.hpp"

namespace attn {

void softmax_exp(Comm& cj, float* shard, int64_t rows, int64_t keys, std::vector<float>& max, Timers& t) {
  { Timed _(t, "rowmax"); row_max(rows, keys, shard, keys, max.data()); }
  cj.allreduce(max.data(), rows, Op::max, Phase::rowstat);
  { Timed _(t, "exp"); exp_shift(rows, keys, shard, keys, max.data()); }
}

void softmax_v_reduce(Comm& cj, const float* s, int64_t rows, int64_t keys, const Mat<float>& v, float* o_partial,
                      std::vector<float>& sum, float* o_owned, Timers& t) {
  const int64_t cols = v.cols, owned = rows / cj.size();

  { Timed _(t, "rowsum"); row_sum(rows, keys, s, keys, sum.data()); }
  cj.allreduce(sum.data(), rows, Op::sum, Phase::rowstat);
  { Timed _(t, "gemm_o"); gemm_nn(rows, cols, keys, 1.f, s, keys, v.ptr(), cols, 0.f, o_partial, cols); }
  cj.reduce_scatter(o_partial, o_owned, owned * cols, Phase::O);
  { Timed _(t, "normalize"); divide_rows(owned, cols, o_owned, cols, sum.data() + cj.rank() * owned); }
}

}  // namespace attn
