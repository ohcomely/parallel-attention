// GPU backend of kernels.hpp and device.hpp: cuBLAS GEMMs and simple row kernels (one thread block per row), all on
// the default stream. Kernel launches are asynchronous; device_sync() (called by Timed and Comm) waits for them.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "attn/common.hpp"
#include "attn/device.hpp"
#include "attn/kernels.hpp"

namespace attn {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(s)));
}

#define ATTN_CHECK(call) check((call), #call)

// Created on first use, on the device selected by then; never destroyed (destroying it at exit can race with the
// CUDA runtime's own teardown).
cublasHandle_t handle() {
  static cublasHandle_t h = [] {
    cublasHandle_t x;
    ATTN_CHECK(cublasCreate(&x));
    ATTN_CHECK(cublasSetMathMode(x, CUBLAS_DEFAULT_MATH));  // strict fp32: no TF32 tensor-core rounding
    return x;
  }();
  return h;
}

// ---- GEMMs: row-major C = A op(B) is column-major C^T = op(B)^T A^T, so the operands are swapped. ----

cublasStatus_t gemm(cublasOperation_t tb, int m, int n, int k, const float* alpha, const float* A, int lda,
                    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cublasSgemm(handle(), tb, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
}

cublasStatus_t gemm(cublasOperation_t tb, int m, int n, int k, const double* alpha, const double* A, int lda,
                    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cublasDgemm(handle(), tb, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
}

// ---- Row kernels ----

constexpr int threads = 256;  // per block; a multiple of 32

__device__ float exp_(float x) { return expf(x); }
__device__ double exp_(double x) { return exp(x); }

template <class T>
struct Max {
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <class T>
struct Sum {
  __device__ T operator()(T a, T b) const { return a + b; }
};

// Reduces v over the block; every thread gets the result. At most one call per kernel (shared buffer reused).
template <class T, class Op>
__device__ T block_reduce(T v, Op op, T identity) {
  __shared__ T partial[threads / 32];
  for (int offset = 16; offset > 0; offset /= 2) v = op(v, __shfl_down_sync(0xffffffffu, v, offset));
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < threads / 32 ? partial[lane] : identity;
    for (int offset = 16; offset > 0; offset /= 2) v = op(v, __shfl_down_sync(0xffffffffu, v, offset));
    if (lane == 0) partial[0] = v;
  }
  __syncthreads();
  return partial[0];
}

template <class T>
__global__ void row_max_kernel(int64_t n, const T* S, int64_t lds, T* max) {
  const T* row = S + int64_t(blockIdx.x) * lds;
  T v = T(-INFINITY);
  for (int64_t c = threadIdx.x; c < n; c += threads) v = Max<T>{}(v, row[c]);
  v = block_reduce(v, Max<T>{}, T(-INFINITY));
  if (threadIdx.x == 0) max[blockIdx.x] = v;
}

template <class T>
__global__ void row_sum_kernel(int64_t n, const T* S, int64_t lds, T* sum) {
  const T* row = S + int64_t(blockIdx.x) * lds;
  T v = 0;
  for (int64_t c = threadIdx.x; c < n; c += threads) v += row[c];
  v = block_reduce(v, Sum<T>{}, T(0));
  if (threadIdx.x == 0) sum[blockIdx.x] = v;
}

template <class T>
__global__ void exp_shift_kernel(int64_t n, T* S, int64_t lds, const T* max) {
  T* row = S + int64_t(blockIdx.x) * lds;
  const T m = max[blockIdx.x];
  for (int64_t c = threadIdx.x; c < n; c += threads) row[c] = exp_(row[c] - m);
}

template <class T>
__global__ void exp_shift_rowsum_kernel(int64_t n, T* S, int64_t lds, const T* max, T* sum) {
  T* row = S + int64_t(blockIdx.x) * lds;
  const T m = max[blockIdx.x];
  T v = 0;
  for (int64_t c = threadIdx.x; c < n; c += threads) {
    const T e = exp_(row[c] - m);
    row[c] = e;
    v += e;
  }
  v = block_reduce(v, Sum<T>{}, T(0));
  if (threadIdx.x == 0) sum[blockIdx.x] = v;
}

template <class T>
__global__ void divide_rows_kernel(int64_t n, T* X, int64_t ldx, const T* denom) {
  T* row = X + int64_t(blockIdx.x) * ldx;
  const T inv = T(1) / denom[blockIdx.x];
  for (int64_t c = threadIdx.x; c < n; c += threads) row[c] *= inv;
}

template <class T>
__global__ void online_rescale_kernel(int64_t n, T* max, const T* block_max, T* sum, T* O, int64_t ldo) {
  const int64_t r = blockIdx.x;
  const T old_max = max[r];
  const T new_max = Max<T>{}(old_max, block_max[r]);
  const T factor = exp_(old_max - new_max);
  __syncthreads();  // every thread has read max[r] before thread 0 overwrites it
  if (threadIdx.x == 0) {
    max[r] = new_max;
    sum[r] *= factor;
  }
  T* row = O + r * ldo;
  for (int64_t c = threadIdx.x; c < n; c += threads) row[c] *= factor;
}

template <class T>
__global__ void add_inplace_kernel(int64_t n, const T* x, T* y) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    y[i] += x[i];
}

template <class T>
__global__ void fill_kernel(int64_t n, T value, T* x) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    x[i] = value;
}

// Grid for an element-wise kernel over n entries (grid-stride loop covers the rest).
unsigned elementwise_blocks(int64_t n) {
  const int64_t blocks = (n + threads - 1) / threads;
  return unsigned(blocks < (1 << 20) ? blocks : (1 << 20));
}

unsigned row_blocks(int64_t m) {
  require(m <= INT_MAX, "too many rows for one launch: " + std::to_string(m));
  return unsigned(m);
}

void check_launch(const char* what) { check(cudaGetLastError(), what); }

}  // namespace

const char* blas_name() { return ATTN_BLAS_NAME; }
int blas_threads() { return -1; }

template <class T>
void gemm_nt(int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const T* B, int64_t ldb, T beta,
             T* C, int64_t ldc) {
  ATTN_CHECK(gemm(CUBLAS_OP_T, to_int(m), to_int(n), to_int(k), &alpha, A, to_int(lda), B, to_int(ldb), &beta, C,
                  to_int(ldc)));
}

template <class T>
void gemm_nn(int64_t m, int64_t n, int64_t k, T alpha, const T* A, int64_t lda, const T* B, int64_t ldb, T beta,
             T* C, int64_t ldc) {
  ATTN_CHECK(gemm(CUBLAS_OP_N, to_int(m), to_int(n), to_int(k), &alpha, A, to_int(lda), B, to_int(ldb), &beta, C,
                  to_int(ldc)));
}

template <class T>
void row_max(int64_t m, int64_t n, const T* S, int64_t lds, T* max) {
  if (m == 0) return;
  row_max_kernel<<<row_blocks(m), threads>>>(n, S, lds, max);
  check_launch("row_max");
}

template <class T>
void row_sum(int64_t m, int64_t n, const T* S, int64_t lds, T* sum) {
  if (m == 0) return;
  row_sum_kernel<<<row_blocks(m), threads>>>(n, S, lds, sum);
  check_launch("row_sum");
}

template <class T>
void exp_shift(int64_t m, int64_t n, T* S, int64_t lds, const T* max) {
  if (m == 0) return;
  exp_shift_kernel<<<row_blocks(m), threads>>>(n, S, lds, max);
  check_launch("exp_shift");
}

template <class T>
void exp_shift_rowsum(int64_t m, int64_t n, T* S, int64_t lds, const T* max, T* sum) {
  if (m == 0) return;
  exp_shift_rowsum_kernel<<<row_blocks(m), threads>>>(n, S, lds, max, sum);
  check_launch("exp_shift_rowsum");
}

template <class T>
void divide_rows(int64_t m, int64_t n, T* X, int64_t ldx, const T* denom) {
  if (m == 0) return;
  divide_rows_kernel<<<row_blocks(m), threads>>>(n, X, ldx, denom);
  check_launch("divide_rows");
}

template <class T>
void online_rescale(int64_t m, int64_t n, T* max, const T* block_max, T* sum, T* O, int64_t ldo) {
  if (m == 0) return;
  online_rescale_kernel<<<row_blocks(m), threads>>>(n, max, block_max, sum, O, ldo);
  check_launch("online_rescale");
}

template <class T>
void add_inplace(int64_t n, const T* x, T* y) {
  if (n == 0) return;
  add_inplace_kernel<<<elementwise_blocks(n), threads>>>(n, x, y);
  check_launch("add_inplace");
}

template <class T>
void fill(int64_t n, T value, T* x) {
  if (n == 0) return;
  fill_kernel<<<elementwise_blocks(n), threads>>>(n, value, x);
  check_launch("fill");
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
  template void add_inplace<T>(int64_t, const T*, T*);                                                         \
  template void fill<T>(int64_t, T, T*);

ATTN_INSTANTIATE(float)
ATTN_INSTANTIATE(double)

// ---- Device memory ----

void* device_alloc(size_t bytes) {
  if (bytes == 0) return nullptr;
  void* p = nullptr;
  ATTN_CHECK(cudaMalloc(&p, bytes));
  return p;
}

void device_free(void* p) {
  if (p) cudaFree(p);  // no throw: called from destructors
}

void copy_to_device(void* dst, const void* src, size_t bytes) {
  if (bytes) ATTN_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void copy_to_host(void* dst, const void* src, size_t bytes) {
  if (bytes) ATTN_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void copy_device(void* dst, const void* src, size_t bytes) {
  if (bytes) ATTN_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0));
}

void device_sync() { ATTN_CHECK(cudaDeviceSynchronize()); }

void select_device(int node_local_rank) {
  int count = 0;
  ATTN_CHECK(cudaGetDeviceCount(&count));
  require(count > 0, "no CUDA device visible");
  ATTN_CHECK(cudaSetDevice(node_local_rank % count));
}

DeviceInfo device_info() {
  int dev = 0, runtime = 0, major = 0, minor = 0, patch = 0;
  ATTN_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  ATTN_CHECK(cudaGetDeviceProperties(&prop, dev));
  ATTN_CHECK(cudaRuntimeGetVersion(&runtime));
  ATTN_CHECK(cublasGetProperty(MAJOR_VERSION, &major));
  ATTN_CHECK(cublasGetProperty(MINOR_VERSION, &minor));
  ATTN_CHECK(cublasGetProperty(PATCH_LEVEL, &patch));
  return {"gpu", prop.name, std::to_string(runtime / 1000) + "." + std::to_string(runtime % 1000 / 10),
          std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch)};
}

}  // namespace attn
