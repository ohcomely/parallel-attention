#include <array>
#include <cmath>

#include "attn/algorithm.hpp"
#include "attn/kernels.hpp"
#include "attn/rng.hpp"
#include "steps.hpp"

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

int64_t checked_block_rows(const Config& c, MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  require(c.T >= 1, "T must be positive");
  require(c.grid.size() * 2 == size, "split grid " + c.grid.str() + " must have P/2 ranks");
  return exact_div(c.prob.n, c.T, "n / T");
}

class AttnSplit final : public Algorithm {
 public:
  AttnSplit(const Config& c, MPI_Comm comm, Timers& t)
      : p_(c.prob),
        t_(t),
        T_(c.T),
        block_rows_(checked_block_rows(c, comm)),
        L_(make_layout3d(c.prob, c.grid, block_rows_)),
        pair_(Comm::dup(comm, t)),
        role_(split_role(pair_.rank())),
        partner_(split_partner(pair_.rank())),
        g_(make_group_grid(comm, role_, c.grid, c.map, t)) {
    const auto [i, j, k] = g_.at;
    for (auto& s : shard_) s = Mat<float>(L_.s_rows, L_.b);
    if (role_ == Role::producer) {
      k_own_ = generate(ArrayId::K, {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c1, L_.c1}, p_.seed);
      for (int b = 0; b < T_; ++b)
        q_own_.push_back(generate(
            ArrayId::Q, {b * block_rows_ + i * L_.a + j * L_.a_chunk, L_.a_chunk, k * L_.c1, L_.c1}, p_.seed));
      k_ = Mat<float>(L_.b, L_.c1);
      q_ = Mat<float>(L_.a, L_.c1);
      s_ = Mat<float>(L_.a, L_.b);
    } else {
      v_own_ = generate(ArrayId::V, {j * L_.b + i * L_.b_chunk, L_.b_chunk, k * L_.c2, L_.c2}, p_.seed);
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

  void run() override {
    role_ == Role::producer ? run_producer() : run_consumer();
    t_.block = -1;
  }

  std::vector<OutputPiece> output() const override {
    std::vector<OutputPiece> pieces;
    for (size_t b = 0; b < out_.size(); ++b) pieces.push_back({out_[b], o_own_[b].ptr()});
    return pieces;
  }

  Role role() const override { return role_; }

 private:
  void run_producer() {
    g_.ci.allgather(k_own_.ptr(), k_.ptr(), k_own_.size(), Phase::K);
    for (int b = 0; b < T_; ++b) {
      t_.block = b;
      auto& shard = shard_[b % 2];
      g_.cj.allgather(q_own_[b].ptr(), q_.ptr(), q_own_[b].size(), Phase::Q);
      {
        Timed _(t_, "gemm_s");
        gemm_nt(L_.a, L_.b, L_.c1, 1.f / std::sqrt(float(p_.d1)), q_.ptr(), L_.c1, k_.ptr(), L_.c1, 0.f, s_.ptr(),
                L_.b);
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
      t_.block = b;
      auto& shard = shard_[b % 2];
      pair_.wait(req_[b % 2]);
      if (b + 1 < T_)
        req_[(b + 1) % 2] =
            pair_.irecv(shard_[(b + 1) % 2].ptr(), shard.size(), partner_, b + 1, Phase::S_handoff, true);
      g_.ck.allgather(shard.ptr(), s_.ptr(), shard.size(), Phase::S_ag);
      softmax_v_reduce(g_.cj, s_, v_, o_partial_, max_, sum_, o_own_[b].ptr(), t_);
    }
  }

  Problem p_;
  Timers& t_;
  int T_;
  int64_t block_rows_;
  Layout3D L_;
  Comm pair_;
  Role role_;
  int partner_;
  Grid g_;
  std::array<Mat<float>, 2> shard_;
  std::array<Request, 2> req_;
  Mat<float> k_own_, k_, q_, s_, v_own_, v_, o_partial_;
  std::vector<Mat<float>> q_own_, o_own_;
  std::vector<Block> out_;
  std::vector<float> max_, sum_;
};

}  // namespace

std::unique_ptr<Algorithm> make_attn_split(const Config& c, MPI_Comm comm, Timers& t) {
  return std::make_unique<AttnSplit>(c, comm, t);
}

}  // namespace attn
