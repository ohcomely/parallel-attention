#include "bench/inputs.hpp"

namespace attn::bench {

namespace {
uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
}  // namespace

float generated_value(Array a, int64_t row, int64_t col, uint64_t seed) {
  uint64_t h = splitmix64(seed);
  h = splitmix64(h ^ uint64_t(a));
  h = splitmix64(h ^ uint64_t(row));
  h = splitmix64(h ^ uint64_t(col));
  const double unit = double(h >> 11) * 0x1.0p-53;
  return float(0.2 * unit - 0.1);
}

}  // namespace attn::bench
