#pragma once
#include <vector>

#include "attn/comm.hpp"
#include "attn/timers.hpp"
#include "attn/matrix.hpp"

namespace attn {

// The softmax is split in two so the exponentials can be taken on the 1/p3-size S shard (after the reduce-scatter
// over k, before the all-gather), instead of p3 times on the gathered block:
//
//   softmax_exp:      row max over cj, then shard := exp(shard - max). `max` needs shard.rows entries.
//   softmax_v_reduce: on the gathered exp(S) block: row sums over cj, O_partial = exp(S) V, reduce-scatter over cj
//                     into o_owned, divide by the row sums. `sum` needs s.rows entries.
void softmax_exp(Comm& cj, float* shard, int64_t rows, int64_t keys, std::vector<float>& max, Timers& t);
void softmax_v_reduce(Comm& cj, const Mat<float>& s, const Mat<float>& v, Mat<float>& o_partial,
                      std::vector<float>& sum, float* o_owned, Timers& t);

}  // namespace attn
