#pragma once
#include "attn/layout.hpp"
#include "attn/matrix.hpp"
#include "attn/problem.hpp"

namespace attn {

// Single-process fp64 attention for the O entries in `block`.
Mat<double> reference_output(const Problem& p, Block block);

}  // namespace attn
