#pragma once
// One entry per algorithm. Everything that depends on which algorithm is used (grid rules, cost model,
// construction) goes through this table; adding an algorithm means one source file and one line in registry.cpp.
#include <mpi.h>

#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "attn/attention.hpp"

namespace attn {

struct AlgorithmSpec {
  std::string_view name;
  std::string_view description;
  int groups;      // 1, or 2 when ranks split into producer and consumer groups (each with its own grid)
  bool pipelined;  // accepts T > 1
  bool (*allows)(Dims g);                              // grid shapes the algorithm accepts
  void (*validate)(const Shape& s, Dims g, int T);     // throws std::invalid_argument if blocks do not divide
  double (*words)(const Shape& s, Dims g, int P);      // busiest rank, row statistics excluded
  std::unique_ptr<Attention> (*make)(const Plan& p, MPI_Comm comm);
};

std::span<const AlgorithmSpec* const> algorithms();
const AlgorithmSpec& algorithm(std::string_view name);  // throws std::invalid_argument if unknown

// Valid grid with the fewest predicted words; ties go to smaller p3, then smaller p1.
std::optional<Dims> best_grid(const AlgorithmSpec& spec, const Shape& s, int P, int T = 1);

}  // namespace attn
