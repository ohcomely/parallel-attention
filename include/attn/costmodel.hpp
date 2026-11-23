#pragma once
#include <vector>

#include "attn/common.hpp"
#include "attn/grid.hpp"

// Words received by the busiest rank (research notes §5–6). Row statistics are excluded, as in the bounds.
// Pure formulas; mirrored by tools/costmodel.py. Which formula applies to which algorithm is in the registry.
namespace attn {

double thm_a_words(const Shape& s, int P);
double thm_b_words(const Shape& s, int P);  // NaN when d1 != d2 (not derived)

struct SplitWords {
  double producer, consumer;
};
double words_3d(const Shape& s, Dims g);
SplitWords words_split(const Shape& s, Dims g);
double words_ring(const Shape& s, int P);

std::vector<Dims> all_grids(int ranks);  // every p1 x p2 x p3 with product `ranks`

}  // namespace attn
