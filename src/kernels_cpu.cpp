#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "attn/kernels.hpp"
#include "attn/common.hpp"

#ifdef ATTN_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace attn {

namespace {
void blas_gemm(CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha, const float* A, int lda, const float* B,
               int ldb, float beta, float* C, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
void blas_gemm(CBLAS_TRANSPOSE tb, int m, int n, int k, double alpha, const double* A, int lda, const double* B,
               int ldb, double beta, double* C, int ldc) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
}  // namespace

const char* blas_name() { return ATTN_BLAS_NAME; }

int blas_threads() {
#ifdef ATTN_ACCELERATE
  return -1;
#else
  return openblas_get_num_threads();
#endif
}

int kernel_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

template <class T>
void gemm_nt(int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const T* B, int64_t ldb, T beta,
             T* C, int64_t ldc) {
  blas_gemm(CblasTrans, to_int(m), to_int(n), to_int(k), alpha, A, to_int(lda), B, to_int(ldb), beta, C,
            to_int(ldc));
}

template <class T>
void gemm_nn(int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const T* B, int64_t ldb, T beta,
             T* C, int64_t ldc) {
  blas_gemm(CblasNoTrans, to_int(m), to_int(n), to_int(k), alpha, A, to_int(lda), B, to_int(ldb), beta, C,
            to_int(ldc));
}

// Rows are independent: split them over OpenMP threads when the library is built with OpenMP (ATTN_OPENMP).
// The inner loops are `omp simd` (needs only -fopenmp-simd) so they vectorise without -ffast-math.
#ifdef _OPENMP
#define ATTN_PARALLEL_ROWS _Pragma("omp parallel for schedule(static)")
#else
#define ATTN_PARALLEL_ROWS
#endif

namespace {

// exp(x) for float without libm, so the loop vectorises. Cephes expf polynomial on x - n ln2, |rel err| < 2e-7.
// Inputs below -87 (where expf underflows to denormals) give 0; inputs above 88 are clamped.
inline float exp_vec(float x) {
  const float xc = x < -87.f ? -87.f : (x > 88.f ? 88.f : x);
  const float t = xc * 1.44269504088896341f;                       // x / ln 2
  const int32_t ni = int32_t(t + (t < 0.f ? -0.5f : 0.5f));        // round to nearest
  const float n = float(ni);
  const float r = (xc - n * 0.693359375f) + n * 2.12194440e-4f;    // x - n ln2, ln2 split in two parts
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  const float er = p * r * r + r + 1.f;
  const float scale = std::bit_cast<float>(int32_t((ni + 127) << 23));  // 2^n, n in [-126, 127]
  return x < -87.f ? 0.f : er * scale;
}

template <class T>
T exp_t(T x) {
  if constexpr (std::is_same_v<T, float>) return exp_vec(x);
  else return std::exp(x);  // fp64 is only used by the reference: keep libm accuracy
}

}  // namespace

template <class T>
void row_max(int64_t m, int64_t n, const T* S, int64_t lds, T* max) {
  ATTN_PARALLEL_ROWS
  for (int64_t r = 0; r < m; ++r) {
    const T* row = S + r * lds;
    T mx = row[0];
    for (int64_t c = 0; c < n; ++c) mx = std::fmax(mx, row[c]);  // fmax vectorises; an omp simd max does not (clang)
    max[r] = mx;
  }
}

template <class T>
void row_sum(int64_t m, int64_t n, const T* S, int64_t lds, T* sum) {
  ATTN_PARALLEL_ROWS
  for (int64_t r = 0; r < m; ++r) {
    const T* row = S + r * lds;
    T s = 0;
#pragma omp simd reduction(+ : s)
    for (int64_t c = 0; c < n; ++c) s += row[c];
    sum[r] = s;
  }
}

template <class T>
void exp_shift(int64_t m, int64_t n, T* S, int64_t lds, const T* max) {
  ATTN_PARALLEL_ROWS
  for (int64_t r = 0; r < m; ++r) {
    T* row = S + r * lds;
    const T mx = max[r];
#pragma omp simd
    for (int64_t c = 0; c < n; ++c) row[c] = exp_t(row[c] - mx);
  }
}

template <class T>
void exp_shift_rowsum(int64_t m, int64_t n, T* S, int64_t lds, const T* max, T* sum) {
  ATTN_PARALLEL_ROWS
  for (int64_t r = 0; r < m; ++r) {
    T* row = S + r * lds;
    const T mx = max[r];
    T s = 0;
#pragma omp simd reduction(+ : s)
    for (int64_t c = 0; c < n; ++c) {
      row[c] = exp_t(row[c] - mx);
      s += row[c];
    }
    sum[r] = s;
  }
}

template <class T>
void divide_rows(int64_t m, int64_t n, T* X, int64_t ldx, const T* denom) {
  ATTN_PARALLEL_ROWS
  for (int64_t r = 0; r < m; ++r) {
    const T inv = T(1) / denom[r];
    T* row = X + r * ldx;
#pragma omp simd
    for (int64_t c = 0; c < n; ++c) row[c] *= inv;
  }
}

template <class T>
void online_rescale(int64_t m, int64_t n, T* max, const T* block_max, T* sum, T* O, int64_t ldo) {
  for (int64_t r = 0; r < m; ++r) {
    const T new_max = std::max(max[r], block_max[r]);
    const T factor = std::exp(max[r] - new_max);
    max[r] = new_max;
    sum[r] *= factor;
    for (int64_t c = 0; c < n; ++c) O[r * ldo + c] *= factor;
  }
}

template <class T>
void add_inplace(int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) y[i] += x[i];
}

#define ATTN_INSTANTIATE(T)                                                                                    \
  template void gemm_nt<T>(int64_t, int64_t, int64_t, T, const T*, int64_t, const T*, int64_t, T, T*, int64_t); \
  template void gemm_nn<T>(int64_t, int64_t, int64_t, T, const T*, int64_t, const T*, int64_t, T, T*, int64_t); \
  template void row_max<T>(int64_t, int64_t, const T*, int64_t, T*);                                           \
  template void row_sum<T>(int64_t, int64_t, const T*, int64_t, T*);                                           \
  template void exp_shift<T>(int64_t, int64_t, T*, int64_t, const T*);                                         \
  template void exp_shift_rowsum<T>(int64_t, int64_t, T*, int64_t, const T*, T*);                              \
  template void divide_rows<T>(int64_t, int64_t, T*, int64_t, const T*);                                       \
  template void online_rescale<T>(int64_t, int64_t, T*, const T*, T*, T*, int64_t);                            \
  template void add_inplace<T>(int64_t, const T*, T*);

ATTN_INSTANTIATE(float)
ATTN_INSTANTIATE(double)

}  // namespace attn
