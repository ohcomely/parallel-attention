#include <cmath>

#include "attn/algorithm.hpp"
#include "attn/kernels.hpp"
#include "attn/rng.hpp"
#include "steps.hpp"

namespace attn {

namespace {

class Attn3D final : public Algorithm {
 public:
  Attn3D(const Config& c, MPI_Comm comm, Timers& t)
      : p_(c.prob), t_(t), L_(make_layout3d(c.prob, c.grid, c.prob.n)), g_(make_grid(comm, c.grid, c.map, t)) {
    const auto [i, j, k] = g_.at;
    q_own_ = generate(ArrayId::Q, {i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c1, L_.c1}, p_.seed);
    k_own_ = generate(ArrayId::K, {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c1, L_.c1}, p_.seed);
    v_own_ = generate(ArrayId::V, {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c2, L_.c2}, p_.seed);
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
    out_ = {i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c2, L_.c2};
  }

  void run() override {
    g_.cj.allgather(q_own_.ptr(), q_.ptr(), q_own_.size(), Phase::Q);
    g_.ci.allgather(k_own_.ptr(), k_.ptr(), k_own_.size(), Phase::K);
    const bool split_depth = g_.dims.p3 > 1;
    {
      Timed _(t_, "gemm_s");
      gemm_nt(L_.a, L_.b, L_.c1, 1.f / std::sqrt(float(p_.d1)), q_.ptr(), L_.c1, k_.ptr(), L_.c1, 0.f,
              split_depth ? s_partial_.ptr() : s_.ptr(), L_.b);
    }
    if (split_depth) {
      g_.ck.reduce_scatter(s_partial_.ptr(), s_shard_.ptr(), s_shard_.size(), Phase::S_rs);
      g_.ck.allgather(s_shard_.ptr(), s_.ptr(), s_shard_.size(), Phase::S_ag);
    }
    g_.ci.allgather(v_own_.ptr(), v_.ptr(), v_own_.size(), Phase::V);
    softmax_v_reduce(g_.cj, s_, v_, o_partial_, max_, sum_, o_.ptr(), t_);
  }

  std::vector<OutputPiece> output() const override { return {{out_, o_.ptr()}}; }

 private:
  Problem p_;
  Timers& t_;
  Layout3D L_;
  Grid g_;
  Mat<float> q_own_, k_own_, v_own_, q_, k_, v_, s_, s_partial_, s_shard_, o_partial_, o_;
  std::vector<float> max_, sum_;
  Block out_;
};

}  // namespace

std::unique_ptr<Algorithm> make_attn3d(const Config& c, MPI_Comm comm, Timers& t) {
  return std::make_unique<Attn3D>(c, comm, t);
}

}  // namespace attn
