#pragma once
#include <string_view>

#include "attn/grid.hpp"
#include "attn/problem.hpp"

namespace attn {

struct Block {
  int64_t row0 = 0, rows = 0, col0 = 0, cols = 0;
};

int64_t exact_div(int64_t a, int64_t b, std::string_view what);

// Block sizes of a p1 x p2 x p3 grid acting on `query_rows` query rows (n for 3D, n/T for a Split block).
struct Layout3D {
  int64_t a;        // query rows per i
  int64_t b;        // key rows per j
  int64_t c1, c2;   // columns of Q,K / V,O per k
  int64_t a_chunk;  // owned Q,O rows: a / p2
  int64_t b_chunk;  // owned K,V rows: b / p1
  int64_t s_rows;   // S rows after reduce-scatter over k: a / p3
};

Layout3D make_layout3d(const Problem& p, Dims g, int64_t query_rows);

}  // namespace attn
