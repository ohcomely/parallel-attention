#pragma once
// Public entry point: distributed exact attention O = softmax(Q K^T / sqrt(d1)) V over an MPI communicator.
//
//   auto a = attn::make("3d", comm, {n, d1, d2});        // grid chosen by the cost model unless given
//   attn::fill_inputs(*a, [&](attn::Array x, int64_t r, int64_t c) { return value_of(x, r, c); });
//   a->run();                                             // collective over `comm`; may be repeated
//   for (const auto& [block, data] : a->output()) ...     // this rank's part of O
#include <mpi.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "attn/common.hpp"
#include "attn/grid.hpp"
#include "attn/timers.hpp"

namespace attn {

struct Options {
  std::optional<Dims> grid;  // per group for two-group algorithms (split); unset = fewest predicted words
  int T = 1;                 // pipeline blocks, for pipelined algorithms (split)
  std::string map = "kji";   // grid coordinate order across consecutive ranks; first letter varies fastest
};

// Everything resolved before any MPI work: algorithm, shape, rank count, grid and the cost model's prediction.
struct Plan {
  std::string alg;
  Shape shape;
  int P = 1;
  Dims grid;     // per group for split; P x 1 x 1 for ring
  int T = 1;
  std::string map;
  double words;  // words received by the busiest rank, row statistics excluded
};

// Throws std::invalid_argument with a readable message if the configuration is not valid.
Plan plan(std::string_view alg, Shape shape, int P, const Options& opts = {});

enum class Role { all, producer, consumer };
const char* name(Role r);

enum class Array { Q = 1, K = 2, V = 3 };
const char* name(Array a);

struct InputSlot {
  Array array;
  Block block;  // which part of the global matrix this rank supplies
  float* data;  // block.rows x block.cols, row-major
};

struct OutputPiece {
  Block block;  // which part of the global O this rank holds
  const float* data;
};

class Attention {
 public:
  explicit Attention(Plan p) : plan_(std::move(p)) {}
  virtual ~Attention() = default;
  Attention(const Attention&) = delete;
  Attention& operator=(const Attention&) = delete;

  const Plan& plan() const { return plan_; }
  virtual Role role() const { return Role::all; }

  // Input blocks this rank must fill before run(). Every entry of Q, K, V is owned by exactly one rank.
  virtual std::vector<InputSlot> inputs() = 0;
  // The timed algorithm. Collective over the communicator given to make(). Inputs are not modified.
  virtual void run() = 0;
  virtual std::vector<OutputPiece> output() const = 0;

  // Instrumentation of the last run (compute / comm / idle time, per-phase words, optional timeline).
  Timers& timers() { return timers_; }
  const Timers& timers() const { return timers_; }

 protected:
  Timers timers_;

 private:
  Plan plan_;
};

// Builds the algorithm on `comm` (P = its size): creates sub-communicators and allocates buffers. Not timed.
std::unique_ptr<Attention> make(std::string_view alg, MPI_Comm comm, Shape shape, const Options& opts = {});

// Fills every input slot of `a` with value(array, global_row, global_col).
template <class F>
void fill_inputs(Attention& a, F&& value) {
  for (const InputSlot& s : a.inputs())
    for (int64_t r = 0; r < s.block.rows; ++r)
      for (int64_t c = 0; c < s.block.cols; ++c)
        s.data[r * s.block.cols + c] = float(value(s.array, s.block.row0 + r, s.block.col0 + c));
}

}  // namespace attn
