// 3D attention on a p1 x p2 x p3 grid (splits the head dimension over p3); 2D is the same with p3 = 1.
#include <cmath>

#include "algs.hpp"
#include "attn/costmodel.hpp"
#include "attn/kernels.hpp"
#include "attn/layout.hpp"
#include "softmax_v.hpp"

namespace attn {

namespace {

class Attn3D final : public Attention {
 public:
  Attn3D(const Plan& p, MPI_Comm comm)
      : Attention(p), L_(make_layout3d(p.shape, p.grid, p.shape.n)), g_(make_grid(comm, p.grid, p.map, timers_)) {
    const auto [i, j, k] = g_.at;
    q_blk_ = {i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c1, L_.c1};
    k_blk_ = {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c1, L_.c1};
    v_blk_ = {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c2, L_.c2};
    out_ = {i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c2, L_.c2};
    q_own_ = Mat<float>(q_blk_.rows, q_blk_.cols);
    k_own_ = Mat<float>(k_blk_.rows, k_blk_.cols);
    v_own_ = Mat<float>(v_blk_.rows, v_blk_.cols);
    q_ = Mat<float>(L_.a, L_.c1);
    k_ = Mat<float>(L_.b, L_.c1);
    v_ = Mat<float>(L_.b, L_.c2);
    s_ = Mat<float>(L_.a, L_.b);
    if (g_.dims.p3 > 1) {
      s_partial_ = Mat<float>(L_.a, L_.b);
      s_shard_ = Mat<float>(L_.s_rows, L_.b);
    }
    o_partial_ = Mat<float>(L_.a, L_.c2);
    o_ = Mat<float>(L_.a_chunk, L_.c2);
    max_.resize(L_.a);
    sum_.resize(L_.a);
  }

  std::vector<InputSlot> inputs() override {
    return {{Array::Q, q_blk_, q_own_.ptr()}, {Array::K, k_blk_, k_own_.ptr()}, {Array::V, v_blk_, v_own_.ptr()}};
  }

  void run() override {
    auto& t = timers_;
    g_.cj.allgather(q_own_.ptr(), q_.ptr(), q_own_.size(), Phase::Q);
    g_.ci.allgather(k_own_.ptr(), k_.ptr(), k_own_.size(), Phase::K);
    const bool split_depth = g_.dims.p3 > 1;
    {
      Timed _(t, "gemm_s");
      gemm_nt(L_.a, L_.b, L_.c1, 1.f / std::sqrt(float(plan().shape.d1)), q_.ptr(), L_.c1, k_.ptr(), L_.c1, 0.f,
              split_depth ? s_partial_.ptr() : s_.ptr(), L_.b);
    }
    if (split_depth) {
      g_.ck.reduce_scatter(s_partial_.ptr(), s_shard_.ptr(), s_shard_.size(), Phase::S_rs);
      g_.ck.allgather(s_shard_.ptr(), s_.ptr(), s_shard_.size(), Phase::S_ag);
    }
    g_.ci.allgather(v_own_.ptr(), v_.ptr(), v_own_.size(), Phase::V);
    softmax_v_reduce(g_.cj, s_, v_, o_partial_, max_, sum_, o_.ptr(), t);
  }

  std::vector<OutputPiece> output() const override { return {{out_, o_.ptr()}}; }

 private:
  Layout3D L_;
  Grid g_;
  Block q_blk_, k_blk_, v_blk_, out_;
  Mat<float> q_own_, k_own_, v_own_, q_, k_, v_, s_, s_partial_, s_shard_, o_partial_, o_;
  std::vector<float> max_, sum_;
};

void validate_3d(const Shape& s, Dims g, int) { make_layout3d(s, g, s.n); }
double predicted_3d(const Shape& s, Dims g, int) { return words_3d(s, g); }
std::unique_ptr<Attention> make_3d(const Plan& p, MPI_Comm comm) { return std::make_unique<Attn3D>(p, comm); }

}  // namespace

namespace algs {

const AlgorithmSpec attn2d{
    .name = "2d",
    .description = "2D (Attention2D / Mesh-Attention family), grid p1 x p2 x 1",
    .groups = 1,
    .pipelined = false,
    .allows = [](Dims g) { return g.p3 == 1; },
    .validate = validate_3d,
    .words = predicted_3d,
    .make = make_3d,
};

const AlgorithmSpec attn3d{
    .name = "3d",
    .description = "3D, splits the head dimension over p3 > 1",
    .groups = 1,
    .pipelined = false,
    .allows = [](Dims g) { return g.p3 > 1; },
    .validate = validate_3d,
    .words = predicted_3d,
    .make = make_3d,
};

}  // namespace algs

}  // namespace attn
