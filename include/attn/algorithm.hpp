#pragma once
#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

#include "attn/grid.hpp"
#include "attn/layout.hpp"
#include "attn/problem.hpp"
#include "attn/timers.hpp"

namespace attn {

enum class Role { all, producer, consumer };
const char* name(Role r);

struct Config {
  std::string alg = "3d";  // 2d | 3d | split | ring
  Problem prob;
  Dims grid;  // per group for split, ignored for ring
  int T = 1;  // pipeline blocks (split)
  std::string map = "kji";
};

struct OutputPiece {
  Block block;
  const float* data;  // block.rows x block.cols, row-major
};

// Construction lays out the inputs and allocates buffers (untimed); run() is the timed algorithm.
class Algorithm {
 public:
  virtual ~Algorithm() = default;
  virtual void run() = 0;
  virtual std::vector<OutputPiece> output() const = 0;
  virtual Role role() const { return Role::all; }
};

std::unique_ptr<Algorithm> make_attn3d(const Config& c, MPI_Comm comm, Timers& t);
std::unique_ptr<Algorithm> make_attn_split(const Config& c, MPI_Comm comm, Timers& t);
std::unique_ptr<Algorithm> make_attn_ring(const Config& c, MPI_Comm comm, Timers& t);
std::unique_ptr<Algorithm> make_algorithm(const Config& c, MPI_Comm comm, Timers& t);

}  // namespace attn
