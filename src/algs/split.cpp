// Split 3D (T = 1) and Pipelined Split 3D (T > 1): producers compute exp(S - rowmax) for S = Q K^T on their shard,
// consumers gather it and compute the row sums and exp(S) V.
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
  // Block b covers query rows [row0_[b], row0_[b] + L_[b].a * p1); its layout L_[b] differs between blocks only in
  // the query-row sizes (a, a_chunk, s_rows). Buffers are sized for the largest block.
  AttnSplit(const Plan& p, MPI_Comm comm)
      : Attention(p),
        T_(p.T),
        pair_(Comm::dup(comm, timers_)),
        role_(split_role(pair_.rank())),
        partner_(split_partner(pair_.rank())),
        g_(make_group_grid(comm, role_, p.grid, p.map, timers_)) {
    const auto [i, j, k] = g_.at;
    int64_t row0 = 0, max_a = 0;
    for (int64_t rows : p.block_rows) {
      L_.push_back(make_layout3d(p.shape, p.grid, rows));
      row0_.push_back(row0);
      row0 += rows;
      max_a = std::max(max_a, L_.back().a);
    }
    const Layout3D& L = L_[0];  // b, b_chunk, c1, c2 are the same for every block
    for (auto& s : shard_) s = Mat<float>(max_a / p.grid.p3, L.b);
    if (role_ == Role::producer) {
      k_blk_ = {j * L.b + i * L.b_chunk, L.b_chunk, k * L.c1, L.c1};
      k_own_ = Mat<float>(k_blk_.rows, k_blk_.cols);
      for (int b = 0; b < T_; ++b) {
        q_blk_.push_back({row0_[b] + i * L_[b].a + j * L_[b].a_chunk, L_[b].a_chunk, k * L.c1, L.c1});
        q_own_.emplace_back(L_[b].a_chunk, L.c1);
      }
      k_ = Mat<float>(L.b, L.c1);
      q_ = Mat<float>(max_a, L.c1);
      s_ = Mat<float>(max_a, L.b);
      max_.resize(max_a / p.grid.p3);
    } else {
      v_blk_ = {j * L.b + i * L.b_chunk, L.b_chunk, k * L.c2, L.c2};
      v_own_ = Mat<float>(v_blk_.rows, v_blk_.cols);
      v_ = Mat<float>(L.b, L.c2);
      s_ = Mat<float>(max_a, L.b);
      o_partial_ = Mat<float>(max_a, L.c2);
      for (int b = 0; b < T_; ++b) {
        o_own_.emplace_back(L_[b].a_chunk, L.c2);
        out_.push_back({row0_[b] + i * L_[b].a + j * L_[b].a_chunk, L_[b].a_chunk, k * L.c2, L.c2});
      }
      sum_.resize(max_a);
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
  int64_t shard_size(int b) const { return L_[b].s_rows * L_[b].b; }

  void run_producer() {
    g_.ci.allgather(k_own_.ptr(), k_.ptr(), k_own_.size(), Phase::K);
    for (int b = 0; b < T_; ++b) {
      timers_.block = b;
      const Layout3D& L = L_[b];
      auto& shard = shard_[b % 2];
      g_.cj.allgather(q_own_[b].ptr(), q_.ptr(), q_own_[b].size(), Phase::Q);
      {
        Timed _(timers_, "gemm_s");
        gemm_nt(L.a, L.b, L.c1, 1.f / std::sqrt(float(plan().shape.d1)), q_.ptr(), L.c1, k_.ptr(), L.c1, 0.f,
                s_.ptr(), L.b);
      }
      pair_.wait(req_[b % 2]);  // shard buffer from block b-2 must be sent before reuse
      g_.ck.reduce_scatter(s_.ptr(), shard.ptr(), shard_size(b), Phase::S_rs);
      softmax_exp(g_.cj, shard.ptr(), L.s_rows, L.b, max_, timers_);  // the producer sends exp(S - max)
      req_[b % 2] = pair_.isend(shard.ptr(), shard_size(b), partner_, b, Phase::S_handoff);
    }
    for (auto& r : req_) pair_.wait(r);
  }

  void run_consumer() {
    req_[0] = pair_.irecv(shard_[0].ptr(), shard_size(0), partner_, 0, Phase::S_handoff, true);
    g_.ci.allgather(v_own_.ptr(), v_.ptr(), v_own_.size(), Phase::V);
    for (int b = 0; b < T_; ++b) {
      timers_.block = b;
      auto& shard = shard_[b % 2];
      pair_.wait(req_[b % 2]);
      if (b + 1 < T_)
        req_[(b + 1) % 2] =
            pair_.irecv(shard_[(b + 1) % 2].ptr(), shard_size(b + 1), partner_, b + 1, Phase::S_handoff, true);
      g_.ck.allgather(shard.ptr(), s_.ptr(), shard_size(b), Phase::S_ag);
      softmax_v_reduce(g_.cj, s_.ptr(), L_[b].a, L_[b].b, v_, o_partial_.ptr(), sum_, o_own_[b].ptr(), timers_);
    }
  }

  int T_;
  std::vector<Layout3D> L_;
  std::vector<int64_t> row0_;
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
    .validate =
        [](const Shape& s, Dims g, std::span<const int64_t> rows) {
          require(!rows.empty(), "pipeline blocks do not divide n");
          for (int64_t r : rows) make_layout3d(s, g, r);
        },
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
