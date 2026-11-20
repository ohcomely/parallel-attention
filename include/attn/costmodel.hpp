#pragma once
#include <optional>
#include <string>
#include <vector>

#include "attn/algorithm.hpp"

// Words received by the busiest rank (research notes §5–6). Row statistics are excluded, as in the bounds.
// Mirrored by tools/costmodel.py.
namespace attn {

double thm_a_words(const Problem& p, int P);
double thm_b_words(const Problem& p, int P);  // NaN when d1 != d2 (not derived)

struct SplitWords {
  double producer, consumer;
};
double words_3d(const Problem& p, Dims g);
SplitWords words_split(const Problem& p, Dims g);
double words_ring(const Problem& p, int P);
double predicted_words(const Config& c, int P);

std::vector<Dims> all_grids(int ranks);
bool grid_fits(const Config& c, int P);  // the layouts divide exactly

// Fewest predicted words; ties go to smaller p3, then smaller p1. "2d" means p3 = 1, "3d" means p3 > 1.
std::optional<Dims> best_grid(const std::string& alg, const Problem& p, int P, int T = 1);

}  // namespace attn
