#pragma once
#include <array>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace attn {

struct Problem {
  int64_t n = 0, d1 = 0, d2 = 0;
  uint64_t seed = 1;
};

//all possible steps in the algorithm
enum class Phase { Q, K, V, S_rs, S_ag, S_handoff, rowstat, O, ring_kv, count };
constexpr int num_phases = int(Phase::count);
constexpr std::array<std::string_view, num_phases> phase_names{
    "Q", "K", "V", "S_rs", "S_ag", "S_handoff", "rowstat", "O", "ring_kv"};
inline std::string_view name(Phase p) { return phase_names[int(p)]; }

inline void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

//know limit of MPI calls
inline int to_int(int64_t x) {
  require(x >= 0 && x <= INT_MAX, "count exceeds int range: " + std::to_string(x));
  return int(x);
}

}  // namespace attn
