#pragma once
#include <mpi.h>

#include <string>
#include <string_view>

#include "attn/comm.hpp"

namespace attn {

struct Dims {
  int p1 = 1, p2 = 1, p3 = 1;
  int size() const { return p1 * p2 * p3; }
  std::string str() const;
};

Dims parse_dims(std::string_view s);  // "p1xp2xp3" or "p1xp2"

struct Coords {
  int i = 0, j = 0, k = 0;
};

// `map` is a permutation of "ijk"; its first letter varies fastest across consecutive ranks.
Coords coords_of(int rank, Dims d, std::string_view map);

struct Grid {
  Dims dims;
  Coords at;
  Comm ci;  // varying i, fixed (j,k): K, V all-gathers
  Comm cj;  // varying j, fixed (i,k): Q all-gather, row statistics, O reduce-scatter
  Comm ck;  // varying k, fixed (i,j): S reduce-scatter and all-gather
};

Grid make_grid(MPI_Comm comm, Dims d, std::string_view map, Timers& t);

}  // namespace attn
