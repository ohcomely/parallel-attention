// Split 3D (T = 1) and Pipelined Split 3D (T > 1): producers compute S = Q K^T, consumers compute softmax(S) V.
#include <algorithm>
#include <array>
#include <cmath>

#include "algs.hpp"
#include "attn/costmodel.hpp"
#include "attn/kernels.hpp"
#include "attn/layout.hpp"
#include "softmax_v.hpp"

namespace attn {

namespace {

// Producer/consumer pairs are adjacent ranks (2r, 2r+1) so a pair shares a node; within each group the
// group rank r is mapped to (i,j,k) with the same rule as every other algorithm.
Role split_role(int rank) { return rank % 2 == 0 ? Role::producer : Role::consumer; }
int split_partner(int rank) { return rank ^ 1; }

Grid make_group_grid(MPI_Comm comm, Role role, Dims d, std::string_view map, Timers& t) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm group;
  MPI_Comm_split(comm, int(role), rank, &group);
  Grid g = make_grid(group, d, map, t);
  MPI_Comm_free(&group);
  return g;
}

class AttnSplit final : public Attention {
 public:
  AttnSplit(const Plan& p, MPI_Comm comm)
      : Attention(p),
        T_(p.T),
        block_rows_(p.shape.n / p.T),
        L_(make_layout3d(p.shape, p.grid, block_rows_)),
        pair_(Comm::dup(comm, timers_)),
        role_(split_role(pair_.rank())),
        partner_(split_partner(pair_.rank())),
        g_(make_group_grid(comm, role_, p.grid, p.map, timers_)) {
    const auto [i, j, k] = g_.at;
    for (auto& s : shard_) s = Mat<float>(L_.s_rows, L_.b);
    if (role_ == Role::producer) {
      k_blk_ = {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c1, L_.c1};
      k_own_ = Mat<float>(k_blk_.rows, k_blk_.cols);
      for (int b = 0; b < T_; ++b) {
        q_blk_.push_back({b * block_rows_ + i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c1, L_.c1});
        q_own_.emplace_back(L_.a_chunk, L_.c1);
      }
      k_ = Mat<float>(L_.b, L_.c1);
      q_ = Mat<float>(L_.a, L_.c1);
      s_ = Mat<float>(L_.a, L_.b);
    } else {
      v_blk_ = {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c2, L_.c2};
      v_own_ = Mat<float>(v_blk_.rows, v_blk_.cols);
      v_ = Mat<float>(L_.b, L_.c2);
      s_ = Mat<float>(L_.a, L_.b);
      o_partial_ = Mat<float>(L_.a, L_.c2);
      for (int b = 0; b < T_; ++b) {
        o_own_.emplace_back(L_.a_chunk, L_.c2);
        out_.push_back({b * block_rows_ + i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c2, L_.c2});
      }
      max_.resize(L_.a);
      sum_.resize(L_.a);
    }
  }

  Role role() const override { return role_; }

  std::vector<InputSlot> inputs() override {
    if (role_ == Role::consumer) return {{Array::V, v_blk_, v_own_.ptr()}};
    std::vector<InputSlot> slots{{Array::K, k_blk_, k_own_.ptr()}};
    for (int b = 0; b < T_; ++b) slots.push_back({Array::Q, q_blk_[b], q_own_[b].ptr()});
    return slots;
  }

  void run() override {
    role_ == Role::producer ? run_producer() : run_consumer();
    timers_.block = -1;
  }

  std::vector<OutputPiece> output() const override {
    std::vector<OutputPiece> pieces;
    for (size_t b = 0; b < out_.size(); ++b) pieces.push_back({out_[b], o_own_[b].ptr()});
    return pieces;
  }

 private:
  void run_producer() {
    g_.ci.allgather(k_own_.ptr(), k_.ptr(), k_own_.size(), Phase::K);
    for (int b = 0; b < T_; ++b) {
      timers_.block = b;
      auto& shard = shard_[b % 2];
      g_.cj.allgather(q_own_[b].ptr(), q_.ptr(), q_own_[b].size(), Phase::Q);
      {
        Timed _(timers_, "gemm_s");
        gemm_nt(L_.a, L_.b, L_.c1, 1.f / std::sqrt(float(plan().shape.d1)), q_.ptr(), L_.c1, k_.ptr(), L_.c1, 0.f,
                s_.ptr(), L_.b);
      }
      pair_.wait(req_[b % 2]);  // shard buffer from block b-2 must be sent before reuse
      g_.ck.reduce_scatter(s_.ptr(), shard.ptr(), shard.size(), Phase::S_rs);
      req_[b % 2] = pair_.isend(shard.ptr(), shard.size(), partner_, b, Phase::S_handoff);
    }
    for (auto& r : req_) pair_.wait(r);
  }

  void run_consumer() {
    req_[0] = pair_.irecv(shard_[0].ptr(), shard_[0].size(), partner_, 0, Phase::S_handoff, true);
    g_.ci.allgather(v_own_.ptr(), v_.ptr(), v_own_.size(), Phase::V);
    for (int b = 0; b < T_; ++b) {
      timers_.block = b;
      auto& shard = shard_[b % 2];
      pair_.wait(req_[b % 2]);
      if (b + 1 < T_)
        req_[(b + 1) % 2] =
            pair_.irecv(shard_[(b + 1) % 2].ptr(), shard.size(), partner_, b + 1, Phase::S_handoff, true);
      g_.ck.allgather(shard.ptr(), s_.ptr(), shard.size(), Phase::S_ag);
      softmax_v_reduce(g_.cj, s_, v_, o_partial_, max_, sum_, o_own_[b].ptr(), timers_);
    }
  }

  int T_;
  int64_t block_rows_;
  Layout3D L_;
  Comm pair_;
  Role role_;
  int partner_;
  Grid g_;
  std::array<Mat<float>, 2> shard_;
  std::array<Request, 2> req_;
  Block k_blk_, v_blk_;
  std::vector<Block> q_blk_, out_;
  Mat<float> k_own_, k_, q_, s_, v_own_, v_, o_partial_;
  std::vector<Mat<float>> q_own_, o_own_;
  std::vector<float> max_, sum_;
};

}  // namespace

namespace algs {

const AlgorithmSpec split{
    .name = "split",
    .description = "Split 3D (T = 1) / Pipelined Split 3D (T > 1); grid is per group of P/2 ranks",
    .groups = 2,
    .pipelined = true,
    .allows = [](Dims) { return true; },
    .validate = [](const Shape& s, Dims g, int T) { make_layout3d(s, g, exact_div(s.n, T, "n / T")); },
    .words =
        [](const Shape& s, Dims g, int) {
          const SplitWords w = words_split(s, g);
          return std::max(w.producer, w.consumer);
        },
    .make = [](const Plan& p, MPI_Comm comm) -> std::unique_ptr<Attention> {
      return std::make_unique<AttnSplit>(p, comm);
    },
};

}  // namespace algs

}  // namespace attn
