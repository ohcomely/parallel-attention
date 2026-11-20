#include <algorithm>
#include <cmath>
#include <limits>

#include "attn/algorithm.hpp"
#include "attn/kernels.hpp"
#include "attn/rng.hpp"

namespace attn {

namespace {

class AttnRing final : public Algorithm {
 public:
  AttnRing(const Config& c, MPI_Comm comm, Timers& t) : p_(c.prob), t_(t) {
    int size;
    MPI_Comm_size(comm, &size);
    rows_ = exact_div(p_.n, size, "n / P");
    comm_ = Comm::dup(comm, t);
    const int64_t r0 = comm_.rank() * rows_;
    q_ = generate(ArrayId::Q, {r0, rows_, 0, p_.d1}, p_.seed);
    k_own_ = generate(ArrayId::K, {r0, rows_, 0, p_.d1}, p_.seed);
    v_own_ = generate(ArrayId::V, {r0, rows_, 0, p_.d2}, p_.seed);
    k_ = k_next_ = Mat<float>(rows_, p_.d1);
    v_ = v_next_ = Mat<float>(rows_, p_.d2);
    s_ = Mat<float>(rows_, rows_);
    o_ = Mat<float>(rows_, p_.d2);
    max_.resize(rows_);
    sum_.resize(rows_);
    block_max_.resize(rows_);
    block_sum_.resize(rows_);
    out_ = {r0, rows_, 0, p_.d2};
  }

  void run() override {
    const int P = comm_.size(), next = (comm_.rank() + 1) % P, prev = (comm_.rank() + P - 1) % P;
    const float scale = 1.f / std::sqrt(float(p_.d1));
    {
      Timed _(t_, "init");
      k_.data = k_own_.data;
      v_.data = v_own_.data;
      std::fill(max_.begin(), max_.end(), -std::numeric_limits<float>::infinity());
      std::fill(sum_.begin(), sum_.end(), 0.f);
      std::fill(o_.data.begin(), o_.data.end(), 0.f);
    }
    for (int step = 0; step < P; ++step) {
      t_.block = step;
      {
        Timed _(t_, "gemm_s");
        gemm_nt(rows_, rows_, p_.d1, scale, q_.ptr(), p_.d1, k_.ptr(), p_.d1, 0.f, s_.ptr(), rows_);
      }
      {
        Timed _(t_, "softmax");
        row_max(rows_, rows_, s_.ptr(), rows_, block_max_.data());
        online_rescale(rows_, p_.d2, max_.data(), block_max_.data(), sum_.data(), o_.ptr(), p_.d2);
        exp_shift_rowsum(rows_, rows_, s_.ptr(), rows_, max_.data(), block_sum_.data());
        add_inplace(rows_, block_sum_.data(), sum_.data());
      }
      {
        Timed _(t_, "gemm_o");
        gemm_nn(rows_, p_.d2, rows_, 1.f, s_.ptr(), rows_, v_.ptr(), p_.d2, 1.f, o_.ptr(), p_.d2);
      }
      if (step + 1 < P) {
        comm_.sendrecv(k_.ptr(), k_.size(), next, k_next_.ptr(), k_next_.size(), prev, Phase::ring_kv);
        comm_.sendrecv(v_.ptr(), v_.size(), next, v_next_.ptr(), v_next_.size(), prev, Phase::ring_kv);
        std::swap(k_, k_next_);
        std::swap(v_, v_next_);
      }
    }
    t_.block = -1;
    { Timed _(t_, "normalize"); divide_rows(rows_, p_.d2, o_.ptr(), p_.d2, sum_.data()); }
  }

  std::vector<OutputPiece> output() const override { return {{out_, o_.ptr()}}; }

 private:
  Problem p_;
  Timers& t_;
  Comm comm_;
  int64_t rows_ = 0;
  Mat<float> q_, k_own_, v_own_, k_, v_, k_next_, v_next_, s_, o_;
  std::vector<float> max_, sum_, block_max_, block_sum_;
  Block out_;
};

}  // namespace

std::unique_ptr<Algorithm> make_attn_ring(const Config& c, MPI_Comm comm, Timers& t) {
  return std::make_unique<AttnRing>(c, comm, t);
}

}  // namespace attn
