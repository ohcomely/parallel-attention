#pragma once
#include <cstdint>

// Local kernels shared by every algorithm. Row-major, raw pointers, explicit leading dimensions.
namespace attn {

const char* blas_name();

template <class T>  // C = alpha * A * B^T + beta * C;  A: m x k, B: n x k
void gemm_nt(int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const T* B, int64_t ldb, T beta,
             T* C, int64_t ldc);

template <class T>  // C = alpha * A * B + beta * C;  A: m x k, B: k x n
void gemm_nn(int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const T* B, int64_t ldb, T beta,
             T* C, int64_t ldc);

template <class T>
void row_max(int64_t m, int64_t n, const T* S, int64_t lds, T* max);

template <class T>  // S := exp(S - max) in place; sum = row sums of the result
void exp_shift_rowsum(int64_t m, int64_t n, T* S, int64_t lds, const T* max, T* sum);

template <class T>
void divide_rows(int64_t m, int64_t n, T* X, int64_t ldx, const T* denom);

template <class T>  // online softmax: max := max(max, block_max); sum and rows of O rescaled by exp(old - new)
void online_rescale(int64_t m, int64_t n, T* max, const T* block_max, T* sum, T* O, int64_t ldo);

template <class T>
void add_inplace(int64_t n, const T* x, T* y);

}  // namespace attn
