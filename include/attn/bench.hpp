#pragma once
#include <mpi.h>

#include "attn/algorithm.hpp"
#include "attn/matrix.hpp"

namespace attn {

// One repetition: opening barrier, run, own finish (t_total), closing barrier (t_idle_tail).
void run_timed(Algorithm& alg, Timers& t, MPI_Comm comm);

// max |O - O_ref| / max |O_ref| over all ranks; reference computed per owned piece unless `full` is given.
double max_rel_error(const Algorithm& alg, const Problem& p, MPI_Comm comm, const Mat<double>* full = nullptr);

}  // namespace attn
