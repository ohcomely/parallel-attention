#include "bench/reference.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "attn/device.hpp"
#include "attn/kernels.hpp"
#include "bench/inputs.hpp"

namespace attn::bench {

namespace {
DeviceMat<double> to_device(const Mat<double>& m) {
  DeviceMat<double> d(m.rows, m.cols);
  copy_to_device(d.ptr(), m.ptr(), d.bytes());
  return d;
}
}  // namespace

// Same kernels as the algorithms, in double, on the device.
Mat<double> reference_output(const Shape& p, uint64_t seed, Block b) {
  const auto q = to_device(generate<double>(Array::Q, {b.row0, b.rows, 0, p.d1}, seed));
  const auto k = to_device(generate<double>(Array::K, {0, p.n, 0, p.d1}, seed));
  const auto v = to_device(generate<double>(Array::V, {0, p.n, b.col0, b.cols}, seed));
  DeviceMat<double> s(b.rows, p.n), o(b.rows, b.cols), max(b.rows, 1), sum(b.rows, 1);

  gemm_nt(b.rows, p.n, p.d1, 1.0 / std::sqrt(double(p.d1)), q.ptr(), p.d1, k.ptr(), p.d1, 0.0, s.ptr(), p.n);
  row_max(b.rows, p.n, s.ptr(), p.n, max.ptr());
  exp_shift_rowsum(b.rows, p.n, s.ptr(), p.n, max.ptr(), sum.ptr());
  gemm_nn(b.rows, b.cols, p.n, 1.0, s.ptr(), p.n, v.ptr(), b.cols, 0.0, o.ptr(), b.cols);
  divide_rows(b.rows, b.cols, o.ptr(), b.cols, sum.ptr());
  Mat<double> out(b.rows, b.cols);
  copy_to_host(out.ptr(), o.ptr(), o.bytes());
  return out;
}

double max_rel_error(const Attention& alg, uint64_t seed, MPI_Comm comm, const Mat<double>* full) {
  double local[2] = {0, 0};  // max abs diff, max abs reference
  for (const auto& [b, device_data] : alg.output()) {
    std::vector<float> data(size_t(b.rows * b.cols));
    copy_to_host(data.data(), device_data, data.size() * sizeof(float));
    const Mat<double> ref = full ? Mat<double>{} : reference_output(alg.plan().shape, seed, b);
    for (int64_t r = 0; r < b.rows; ++r)
      for (int64_t c = 0; c < b.cols; ++c) {
        const double expected = full ? (*full)(b.row0 + r, b.col0 + c) : ref(r, c);
        const double diff = std::abs(double(data[r * b.cols + c]) - expected);
        local[0] = std::isfinite(diff) ? std::max(local[0], diff) : INFINITY;
        local[1] = std::max(local[1], std::abs(expected));
      }
  }
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, comm);
  return global[0] / global[1];
}

}  // namespace attn::bench
