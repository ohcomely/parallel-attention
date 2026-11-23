#pragma once
#include <mpi.h>

#include "attn/attention.hpp"
#include "attn/matrix.hpp"

namespace attn::bench {

// Single-process fp64 attention on the generated inputs, for the O entries in `block`.
Mat<double> reference_output(const Shape& s, uint64_t seed, Block block);

// max |O - O_ref| / max |O_ref| over all ranks of `comm` (collective). The reference is computed per owned piece
// unless the full n x d2 reference is given.
double max_rel_error(const Attention& alg, uint64_t seed, MPI_Comm comm, const Mat<double>* full = nullptr);

}  // namespace attn::bench
