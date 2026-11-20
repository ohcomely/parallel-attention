#pragma once
#include <cstdint>
#include <vector>

namespace attn {

template <class T>
struct Mat {
  int64_t rows = 0, cols = 0;
  std::vector<T> data;

  Mat() = default;
  Mat(int64_t r, int64_t c) : rows(r), cols(c), data(size_t(r * c)) {}

  int64_t size() const { return rows * cols; }
  T* ptr() { return data.data(); }
  const T* ptr() const { return data.data(); }
  T& operator()(int64_t r, int64_t c) { return data[size_t(r * cols + c)]; }
  const T& operator()(int64_t r, int64_t c) const { return data[size_t(r * cols + c)]; }
};

}  // namespace attn
