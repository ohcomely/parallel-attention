#pragma once
#include <vector>

#include "attn/comm.hpp"
#include "attn/timers.hpp"
#include "attn/matrix.hpp"

namespace attn {

// Second half shared by 3D and the Split consumer, once the complete S block is local:
// row stats over cj, O_partial = exp(S - max) V, reduce-scatter over cj into o_owned, normalise.
void softmax_v_reduce(Comm& cj, Mat<float>& s, const Mat<float>& v, Mat<float>& o_partial, std::vector<float>& max,
                      std::vector<float>& sum, float* o_owned, Timers& t);

}  // namespace attn
