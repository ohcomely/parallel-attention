#include "attn/layout.hpp"

namespace attn {

int64_t exact_div(int64_t a, int64_t b, std::string_view what) {
  require(b > 0 && a % b == 0,
          std::string(what) + ": " + std::to_string(a) + " not divisible by " + std::to_string(b));
  return a / b;
}

Layout3D make_layout3d(const Problem& p, Dims g, int64_t query_rows) {
  Layout3D L;
  L.a = exact_div(query_rows, g.p1, "query rows / p1");
  L.b = exact_div(p.n, g.p2, "n / p2");
  L.c1 = exact_div(p.d1, g.p3, "d1 / p3");
  L.c2 = exact_div(p.d2, g.p3, "d2 / p3");
  L.a_chunk = exact_div(L.a, g.p2, "query rows per i / p2");
  L.b_chunk = exact_div(L.b, g.p1, "key rows per j / p1");
  L.s_rows = exact_div(L.a, g.p3, "query rows per i / p3");
  return L;
}

}  // namespace attn
