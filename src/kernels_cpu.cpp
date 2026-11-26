#include <algorithm>
#include <cmath>
#include <limits>

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

template <class T>
void row_max(int64_t m, int64_t n, const T* S, int64_t lds, T* max) {
  for (int64_t r = 0; r < m; ++r) {
    const T* row = S + r * lds;
    max[r] = *std::max_element(row, row + n);
  }
}

template <class T>
void row_sum(int64_t m, int64_t n, const T* S, int64_t lds, T* sum) {
  for (int64_t r = 0; r < m; ++r) {
    const T* row = S + r * lds;
    T s = 0;
    for (int64_t c = 0; c < n; ++c) s += row[c];
    sum[r] = s;
  }
}

template <class T>
void exp_shift(int64_t m, int64_t n, T* S, int64_t lds, const T* max) {
  for (int64_t r = 0; r < m; ++r) {
    T* row = S + r * lds;
    for (int64_t c = 0; c < n; ++c) row[c] = std::exp(row[c] - max[r]);
  }
}

template <class T>
void exp_shift_rowsum(int64_t m, int64_t n, T* S, int64_t lds, const T* max, T* sum) {
  for (int64_t r = 0; r < m; ++r) {
    T* row = S + r * lds;
    T s = 0;
    for (int64_t c = 0; c < n; ++c) {
      row[c] = std::exp(row[c] - max[r]);
      s += row[c];
    }
    sum[r] = s;
  }
}

template <class T>
void divide_rows(int64_t m, int64_t n, T* X, int64_t ldx, const T* denom) {
  for (int64_t r = 0; r < m; ++r) {
    const T inv = T(1) / denom[r];
    for (int64_t c = 0; c < n; ++c) X[r * ldx + c] *= inv;
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
