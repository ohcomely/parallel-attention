// Ring Attention, non-overlapped: Q rows stay, K and V blocks travel around the ring, online softmax.
#include <algorithm>
#include <cmath>
#include <limits>

#include "algs.hpp"
#include "attn/costmodel.hpp"
#include "attn/kernels.hpp"

namespace attn {

namespace {

class AttnRing final : public Attention {
 public:
  AttnRing(const Plan& p, MPI_Comm comm)
      : Attention(p), comm_(Comm::dup(comm, timers_)), rows_(p.shape.n / comm_.size()) {
    const int64_t r0 = comm_.rank() * rows_, d1 = p.shape.d1, d2 = p.shape.d2;
    q_blk_ = {r0, rows_, 0, d1};
    k_blk_ = {r0, rows_, 0, d1};
    v_blk_ = {r0, rows_, 0, d2};
    out_ = {r0, rows_, 0, d2};
    for (auto* m : {&q_, &k_own_, &k_, &k_next_}) *m = DeviceMat<float>(rows_, d1);
    for (auto* m : {&v_own_, &v_, &v_next_}) *m = DeviceMat<float>(rows_, d2);
    s_ = DeviceMat<float>(rows_, rows_);
    o_ = DeviceMat<float>(rows_, d2);
    for (auto* m : {&max_, &sum_, &block_max_, &block_sum_}) *m = DeviceMat<float>(rows_, 1);
  }

  std::vector<InputSlot> inputs() override {
    return {{Array::Q, q_blk_, q_.ptr()}, {Array::K, k_blk_, k_own_.ptr()}, {Array::V, v_blk_, v_own_.ptr()}};
  }

  void run() override {
    auto& t = timers_;
    const int64_t d1 = plan().shape.d1, d2 = plan().shape.d2;
    const int P = comm_.size(), next = (comm_.rank() + 1) % P, prev = (comm_.rank() + P - 1) % P;
    const float scale = 1.f / std::sqrt(float(d1));
    {
      Timed _(t, "init");
      copy_device(k_.ptr(), k_own_.ptr(), k_.bytes());
      copy_device(v_.ptr(), v_own_.ptr(), v_.bytes());
      fill(rows_, -std::numeric_limits<float>::infinity(), max_.ptr());
      fill(rows_, 0.f, sum_.ptr());
      fill(o_.size(), 0.f, o_.ptr());
    }
    for (int step = 0; step < P; ++step) {
      t.block = step;
      {
        Timed _(t, "gemm_s");
        gemm_nt(rows_, rows_, d1, scale, q_.ptr(), d1, k_.ptr(), d1, 0.f, s_.ptr(), rows_);
      }
      {
        Timed _(t, "softmax");
        row_max(rows_, rows_, s_.ptr(), rows_, block_max_.ptr());
        online_rescale(rows_, d2, max_.ptr(), block_max_.ptr(), sum_.ptr(), o_.ptr(), d2);
        exp_shift_rowsum(rows_, rows_, s_.ptr(), rows_, max_.ptr(), block_sum_.ptr());
        add_inplace(rows_, block_sum_.ptr(), sum_.ptr());
      }
      {
        Timed _(t, "gemm_o");
        gemm_nn(rows_, d2, rows_, 1.f, s_.ptr(), rows_, v_.ptr(), d2, 1.f, o_.ptr(), d2);
      }
      if (step + 1 < P) {
        comm_.sendrecv(k_.ptr(), k_.size(), next, k_next_.ptr(), k_next_.size(), prev, Phase::ring_kv);
        comm_.sendrecv(v_.ptr(), v_.size(), next, v_next_.ptr(), v_next_.size(), prev, Phase::ring_kv);
        std::swap(k_, k_next_);
        std::swap(v_, v_next_);
      }
    }
    t.block = -1;
    { Timed _(t, "normalize"); divide_rows(rows_, d2, o_.ptr(), d2, sum_.ptr()); }
  }

  std::vector<OutputPiece> output() const override { return {{out_, o_.ptr()}}; }

 private:
  Comm comm_;
  int64_t rows_;
  Block q_blk_, k_blk_, v_blk_, out_;
  DeviceMat<float> q_, k_own_, v_own_, k_, v_, k_next_, v_next_, s_, o_;
  DeviceMat<float> max_, sum_, block_max_, block_sum_;
};

}  // namespace

namespace algs {

const AlgorithmSpec ring{
    .name = "ring",
    .description = "Ring Attention baseline (non-overlapped), rank r owns rows r n/P .. (r+1) n/P",
    .groups = 1,
    .pipelined = false,
    .allows = [](Dims g) { return g.p2 == 1 && g.p3 == 1; },
    .validate = [](const Shape& s, Dims g, int) { exact_div(s.n, g.p1, "n / P"); },
    .words = [](const Shape& s, Dims, int P) { return words_ring(s, P); },
    .make = [](const Plan& p, MPI_Comm comm) -> std::unique_ptr<Attention> {
      return std::make_unique<AttnRing>(p, comm);
    },
};

}  // namespace algs

}  // namespace attn
