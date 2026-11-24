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
  // Throws std::invalid_argument if the blocks do not divide. block_rows: query rows of each pipeline block.
  void (*validate)(const Shape& s, Dims g, std::span<const int64_t> block_rows);
  double (*words)(const Shape& s, Dims g, int P);      // busiest rank, row statistics excluded
  std::unique_ptr<Attention> (*make)(const Plan& p, MPI_Comm comm);
};

std::span<const AlgorithmSpec* const> algorithms();
const AlgorithmSpec& algorithm(std::string_view name);  // throws std::invalid_argument if unknown

// Valid grid with the fewest predicted words; ties go to smaller p3, then smaller p1.
std::optional<Dims> best_grid(const AlgorithmSpec& spec, const Shape& s, int P, std::span<const int64_t> block_rows);
std::optional<Dims> best_grid(const AlgorithmSpec& spec, const Shape& s, int P, int T = 1);  // T equal blocks

// Query rows of each pipeline block: T equal blocks, or proportional to `blocks` if given. Empty if they do not divide.
std::vector<int64_t> pipeline_rows(int64_t n, int T, const std::vector<int>& blocks);

}  // namespace attn
