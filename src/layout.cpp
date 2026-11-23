#include "attn/layout.hpp"

namespace attn {

Layout3D make_layout3d(const Shape& s, Dims g, int64_t query_rows) {
  Layout3D L;
  L.a = exact_div(query_rows, g.p1, "query rows / p1");
  L.b = exact_div(s.n, g.p2, "n / p2");
  L.c1 = exact_div(s.d1, g.p3, "d1 / p3");
  L.c2 = exact_div(s.d2, g.p3, "d2 / p3");
  L.a_chunk = exact_div(L.a, g.p2, "query rows per i / p2");
  L.b_chunk = exact_div(L.b, g.p1, "key rows per j / p1");
  L.s_rows = exact_div(L.a, g.p3, "query rows per i / p3");
  return L;
}

}  // namespace attn
