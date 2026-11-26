// Every kernel in kernels.hpp against a naive double-precision host loop, for float and double, on random inputs
// with odd sizes and leading dimensions larger than the row length. Data goes host -> device -> kernel -> host,
// so on the GPU path this also checks the copies. Single process.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "attn/device.hpp"
#include "attn/kernels.hpp"

using namespace attn;

namespace {

int failures = 0, checks = 0;
std::mt19937_64 rng(42);

std::vector<double> random_vec(int64_t n, double lo = -1, double hi = 1) {
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<double> v(size_t(n), 0.0);
  for (double& x : v) x = u(rng);
  return v;
}

// A host array (double, the reference) and its device copy in T.
template <class T>
struct Buf {
  std::vector<double> host;
  DeviceMat<T> dev;

  explicit Buf(std::vector<double> h) : host(std::move(h)), dev(int64_t(host.size()), 1) {
    std::vector<T> t(host.begin(), host.end());
    host.assign(t.begin(), t.end());  // the reference starts from the same rounded values
    copy_to_device(dev.ptr(), t.data(), dev.bytes());
  }
  T* d() { return dev.ptr(); }
  std::vector<double> read() const {
    std::vector<T> t(host.size());
    copy_to_host(t.data(), dev.ptr(), dev.bytes());
    return {t.begin(), t.end()};
  }
};

// Max |device - reference| / max |reference| over the whole buffer (padding included: it must stay untouched).
template <class T>
void expect_close(const Buf<T>& b, const std::string& what) {
  const std::vector<double> got = b.read();
  double diff = 0, scale = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = std::abs(got[i] - b.host[i]);
    diff = std::isfinite(e) ? std::max(diff, e) : INFINITY;
    scale = std::max(scale, std::abs(b.host[i]));
  }
  const double err = diff / std::max(scale, 1e-300);
  const double tol = sizeof(T) == 4 ? 1e-5 : 1e-12;
  ++checks;
  const bool ok = err <= tol;
  failures += !ok;
  std::printf("%s %-6s %-44s rel err %.2e\n", ok ? "ok  " : "FAIL", sizeof(T) == 4 ? "float" : "double",
              what.c_str(), err);
}

template <class T>
void test_shape(int64_t m, int64_t n, int64_t k) {
  const std::string dims = " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
  const int64_t lda = k + 3, ldb_t = k + 5, ldb_n = n + 2, ldc = n + 7;
  const double alpha = 0.7, beta = 0.3;

  {  // gemm_nt: C = alpha A B^T + beta C
    Buf<T> A(random_vec(m * lda)), B(random_vec(n * ldb_t)), C(random_vec(m * ldc));
    gemm_nt<T>(m, n, k, T(alpha), A.d(), lda, B.d(), ldb_t, T(beta), C.d(), ldc);
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j) {
        double s = 0;
        for (int64_t c = 0; c < k; ++c) s += A.host[i * lda + c] * B.host[j * ldb_t + c];
        C.host[i * ldc + j] = alpha * s + beta * C.host[i * ldc + j];
      }
    expect_close(C, "gemm_nt" + dims);
  }
  {  // gemm_nn: C = alpha A B + beta C, B: k x n
    Buf<T> A(random_vec(m * lda)), B(random_vec(k * ldb_n)), C(random_vec(m * ldc));
    gemm_nn<T>(m, n, k, T(alpha), A.d(), lda, B.d(), ldb_n, T(beta), C.d(), ldc);
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j) {
        double s = 0;
        for (int64_t c = 0; c < k; ++c) s += A.host[i * lda + c] * B.host[c * ldb_n + j];
        C.host[i * ldc + j] = alpha * s + beta * C.host[i * ldc + j];
      }
    expect_close(C, "gemm_nn" + dims);
  }

  // Row kernels on an m x n matrix with leading dimension ldc.
  const auto row_ref = [&](const std::vector<double>& S, const std::function<double(double, double)>& op,
                           double init) {
    std::vector<double> out(static_cast<size_t>(m));
    for (int64_t r = 0; r < m; ++r) {
      double acc = init;
      for (int64_t c = 0; c < n; ++c) acc = op(acc, S[r * ldc + c]);
      out[r] = acc;
    }
    return out;
  };
  const auto max_op = [](double a, double b) { return std::max(a, b); };
  const auto sum_op = [](double a, double b) { return a + b; };
  {
    Buf<T> S(random_vec(m * ldc)), out(std::vector<double>(m, 0.0));
    row_max<T>(m, n, S.d(), ldc, out.d());
    out.host = row_ref(S.host, max_op, -INFINITY);
    expect_close(out, "row_max" + dims);
  }
  {
    Buf<T> S(random_vec(m * ldc)), out(std::vector<double>(m, 0.0));
    row_sum<T>(m, n, S.d(), ldc, out.d());
    out.host = row_ref(S.host, sum_op, 0.0);
    expect_close(out, "row_sum" + dims);
  }
  {
    Buf<T> S(random_vec(m * ldc)), mx(random_vec(m));
    exp_shift<T>(m, n, S.d(), ldc, mx.d());
    for (int64_t r = 0; r < m; ++r)
      for (int64_t c = 0; c < n; ++c) S.host[r * ldc + c] = std::exp(S.host[r * ldc + c] - mx.host[r]);
    expect_close(S, "exp_shift" + dims);
  }
  {
    Buf<T> S(random_vec(m * ldc)), mx(random_vec(m)), sum(std::vector<double>(m, 0.0));
    exp_shift_rowsum<T>(m, n, S.d(), ldc, mx.d(), sum.d());
    for (int64_t r = 0; r < m; ++r)
      for (int64_t c = 0; c < n; ++c) S.host[r * ldc + c] = std::exp(S.host[r * ldc + c] - mx.host[r]);
    sum.host = row_ref(S.host, sum_op, 0.0);
    expect_close(S, "exp_shift_rowsum (S)" + dims);
    expect_close(sum, "exp_shift_rowsum (sum)" + dims);
  }
  {
    Buf<T> X(random_vec(m * ldc)), denom(random_vec(m, 0.5, 2));
    divide_rows<T>(m, n, X.d(), ldc, denom.d());
    for (int64_t r = 0; r < m; ++r)
      for (int64_t c = 0; c < n; ++c) X.host[r * ldc + c] /= denom.host[r];
    expect_close(X, "divide_rows" + dims);
  }
  {
    Buf<T> mx(random_vec(m)), bmx(random_vec(m)), sum(random_vec(m, 0.5, 2)), O(random_vec(m * ldc));
    online_rescale<T>(m, n, mx.d(), bmx.d(), sum.d(), O.d(), ldc);
    for (int64_t r = 0; r < m; ++r) {
      const double new_max = std::max(mx.host[r], bmx.host[r]), f = std::exp(mx.host[r] - new_max);
      mx.host[r] = new_max;
      sum.host[r] *= f;
      for (int64_t c = 0; c < n; ++c) O.host[r * ldc + c] *= f;
    }
    expect_close(mx, "online_rescale (max)" + dims);
    expect_close(sum, "online_rescale (sum)" + dims);
    expect_close(O, "online_rescale (O)" + dims);
  }
  {
    Buf<T> x(random_vec(m * n)), y(random_vec(m * n));
    add_inplace<T>(m * n, x.d(), y.d());
    for (int64_t i = 0; i < m * n; ++i) y.host[i] += x.host[i];
    expect_close(y, "add_inplace" + dims);
  }
  {
    Buf<T> x(random_vec(m * n + 3));
    fill<T>(m * n, T(-0.25), x.d());  // the last 3 entries must stay
    std::fill(x.host.begin(), x.host.begin() + m * n, -0.25);
    expect_close(x, "fill" + dims);
  }
}

}  // namespace

int main() {
  for (const auto [m, n, k] : {std::array<int64_t, 3>{37, 300, 65}, {64, 1024, 128}, {3, 1, 2}}) {
    test_shape<float>(m, n, k);
    test_shape<double>(m, n, k);
  }
  std::printf("%s (%d of %d checks failed)\n", failures ? "FAILED" : "PASSED", failures, checks);
  return failures ? 1 : 0;
}
