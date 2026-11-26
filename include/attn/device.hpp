#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

// Memory the kernels run on: GPU memory with ATTN_GPU, plain host memory otherwise (copies are then memcpy).
// Implemented next to the kernels, in kernels_cpu.cpp or kernels_gpu.cu (neither depends on MPI).
namespace attn {

void* device_alloc(size_t bytes);
void device_free(void* p);
void copy_to_device(void* dst, const void* src, size_t bytes);  // host -> device
void copy_to_host(void* dst, const void* src, size_t bytes);    // device -> host
void copy_device(void* dst, const void* src, size_t bytes);     // device -> device
void device_sync();             // wait for all queued device work; no-op on CPU
void select_device(int node_local_rank);  // GPU: device node_local_rank % device count; no-op on CPU.
                                          // Called by attn::device_init (attention.hpp).

struct DeviceInfo {
  std::string device, gpu, cuda, cublas;  // "cpu", "na", "na", "na" on the CPU path
};
DeviceInfo device_info();

// A rows x cols row-major matrix in device memory. Move-only; no element access from the host.
template <class T>
class DeviceMat {
 public:
  int64_t rows = 0, cols = 0;

  DeviceMat() = default;
  DeviceMat(int64_t r, int64_t c) : rows(r), cols(c), p_(static_cast<T*>(device_alloc(size_t(r * c) * sizeof(T)))) {}
  DeviceMat(DeviceMat&& o) noexcept { *this = std::move(o); }
  DeviceMat& operator=(DeviceMat&& o) noexcept {
    if (this != &o) {
      device_free(p_);
      rows = std::exchange(o.rows, 0);
      cols = std::exchange(o.cols, 0);
      p_ = std::exchange(o.p_, nullptr);
    }
    return *this;
  }
  DeviceMat(const DeviceMat&) = delete;
  DeviceMat& operator=(const DeviceMat&) = delete;
  ~DeviceMat() { device_free(p_); }

  int64_t size() const { return rows * cols; }
  size_t bytes() const { return size_t(size()) * sizeof(T); }
  T* ptr() { return p_; }
  const T* ptr() const { return p_; }

 private:
  T* p_ = nullptr;
};

}  // namespace attn
