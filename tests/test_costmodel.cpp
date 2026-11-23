#include <cmath>
#include <cstdio>
#include <string>

#include "attn/attention.hpp"
#include "attn/costmodel.hpp"
#include "attn/registry.hpp"
#include "bench/stats.hpp"

using namespace attn;
using attn::bench::median_ci;
using attn::bench::MedianCI;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
  }
}

void expect_near(double got, double want, double tol, const std::string& what) {
  expect(std::abs(got - want) <= tol, what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

void expect_grid(const std::string& alg, const Shape& s, int P, Dims want) {
  const Dims g = plan(alg, s, P).grid;
  expect(g.str() == want.str(), alg + " P=" + std::to_string(P) + ": got " + g.str() + ", want " + want.str());
}

double best_words(const std::string& alg, const Shape& s, int P) { return plan(alg, s, P).words; }

// plan() must reject the options with std::invalid_argument whose message contains `needle`.
void expect_rejected(const std::string& alg, const Shape& s, int P, const Options& o, const std::string& needle) {
  try {
    plan(alg, s, P, o);
    expect(false, alg + ": expected rejection containing '" + needle + "'");
  } catch (const std::invalid_argument& e) {
    expect(std::string(e.what()).find(needle) != std::string::npos,
           alg + ": message '" + e.what() + "' lacks '" + needle + "'");
  }
}

}  // namespace

int main() {
  // Research notes §9: n = 2^15, d = 2^7 (values there are rounded to integers).
  const Shape notes{1 << 15, 1 << 7, 1 << 7};
  const struct {
    int P;
    double a, b;
  } table[] = {{1 << 14, 130048, 130048}, {1 << 16, 65280, 61384}, {1 << 18, 30756, 26804},
               {1 << 20, 13418, 11248},   {1 << 22, 5628, 4616},    {1 << 24, 2309, 1870},
               {1 << 26, 935, 752}};
  for (const auto& row : table) {
    expect_near(thm_a_words(notes, row.P), row.a, 1, "Theorem A at P=" + std::to_string(row.P));
    expect_near(thm_b_words(notes, row.P), row.b, 1, "Theorem B at P=" + std::to_string(row.P));
  }
  expect_near(best_words("2d", notes, 1 << 14), 130048, 0.5, "notes 2D at 2^14");
  expect_near(best_words("split", notes, 1 << 14), 228352, 0.5, "notes Split at 2^14");
  expect_near(best_words("split", notes, 1 << 16), 81664, 0.5, "notes Split at 2^16");

  // Implementation brief §6 and §7.2: n = 16384, d = 4096.
  const Shape main_shape{16384, 4096, 4096};
  expect_grid("2d", main_shape, 16, {4, 4, 1});
  expect_grid("3d", main_shape, 128, {8, 8, 2});
  expect_grid("3d", main_shape, 1024, {16, 16, 4});
  expect_grid("split", main_shape, 256, {8, 8, 2});
  // The brief lists 8x16x4 here; 16x16x2 predicts exactly the same words and wins the smaller-p3 tie-break.
  expect_grid("split", main_shape, 1024, {16, 16, 2});
  expect_near(words_split(main_shape, {8, 16, 4}).consumer, words_split(main_shape, {16, 16, 2}).consumer, 0.5,
              "Split 8x16x4 and 16x16x2 tie");
  expect_near(1 - best_words("3d", main_shape, 128) / best_words("2d", main_shape, 128), 0.18, 0.005,
              "3D vs 2D at P=128");
  expect_near(1 - best_words("3d", main_shape, 1024) / best_words("2d", main_shape, 1024), 0.32, 0.005,
              "3D vs 2D at P=1024");
  expect_near(words_ring(main_shape, 1024), 2.0 * 16384 * 4096 * (1 - 1.0 / 1024), 0.5, "Ring words");
  expect(std::isnan(thm_b_words({16384, 4096, 1024}, 64)), "Theorem B undefined for d1 != d2");
  expect(!best_grid(algorithm("split"), main_shape, 7), "Split needs even P");

  // plan(): defaults, explicit grids and the error messages.
  const Plan p = plan("split", {16384, 4096}, 1024);
  expect(p.shape.d2 == 4096 && p.T == 1 && p.map == "kji", "plan fills defaults (d2 = d1, T = 1, map kji)");
  expect_near(p.words, 4980736, 0.5, "Split words at P=1024");
  expect(plan("ring", main_shape, 64).grid.str() == "64x1x1", "ring grid is P x 1 x 1");
  expect(plan("3d", main_shape, 128, {.grid = Dims{4, 8, 4}}).grid.str() == "4x8x4", "explicit grid is kept");
  expect_rejected("4d", main_shape, 8, {}, "unknown algorithm");
  expect_rejected("3d", main_shape, 4, {.grid = Dims{2, 2, 1}}, "does not accept grid");
  expect_rejected("2d", main_shape, 8, {.grid = Dims{2, 2, 1}}, "needs 4 ranks");
  expect_rejected("2d", main_shape, 8, {.T = 2}, "not pipelined");
  expect_rejected("split", main_shape, 7, {}, "multiple of 2");
  expect_rejected("split", {12, 4, 4}, 8, {.T = 5}, "no valid split grid");
  expect_rejected("2d", main_shape, 4, {.map = "kjj"}, "permutation");
  expect_rejected("2d", {0, 4}, 4, {}, "positive");

  // Order-statistics CI of the median: for 20 samples the 95% interval is [x_(6), x_(15)] (1-based).
  std::vector<double> x;
  for (int i = 20; i >= 1; --i) x.push_back(i);
  const MedianCI m = median_ci(x);
  expect(m.exact && m.median == 10.5 && m.lo == 6 && m.hi == 15, "median CI of 1..20");
  expect(!median_ci({1, 2, 3, 4, 5}).exact, "5 samples cannot give a 95% CI");
  expect(median_ci({1, 2, 3, 4, 5, 6}).exact, "6 samples give a 95% CI");

  std::printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
  return failures ? 1 : 0;
}
