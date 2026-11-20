"""Checks tools/costmodel.py and tools/stats.py against the same values as tests/test_costmodel.cpp."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import costmodel as cm  # noqa: E402
import stats  # noqa: E402

failures = []


def expect(ok, what):
    if not ok:
        failures.append(what)


def near(got, want, tol, what):
    expect(abs(got - want) <= tol, f"{what}: got {got}, want {want}")


n, d = 1 << 15, 1 << 7
for P, a, b in [(1 << 14, 130048, 130048), (1 << 16, 65280, 61384), (1 << 18, 30756, 26804),
                (1 << 20, 13418, 11248), (1 << 22, 5628, 4616), (1 << 24, 2309, 1870), (1 << 26, 935, 752)]:
    near(cm.thm_a(n, d, d, P), a, 1, f"Theorem A at P={P}")
    near(cm.thm_b(n, d, d, P), b, 1, f"Theorem B at P={P}")


def best_words(alg, n, d, P):
    return cm.predicted_words(alg, n, d, d, P, cm.best_grid(alg, n, d, d, P))


near(best_words("2d", n, d, 1 << 14), 130048, 0.5, "notes 2D at 2^14")
near(best_words("split", n, d, 1 << 14), 228352, 0.5, "notes Split at 2^14")
near(best_words("split", n, d, 1 << 16), 81664, 0.5, "notes Split at 2^16")

n, d = 16384, 4096
for alg, P, want in [("2d", 16, (4, 4, 1)), ("3d", 128, (8, 8, 2)), ("3d", 1024, (16, 16, 4)),
                     ("split", 256, (8, 8, 2)), ("split", 1024, (16, 16, 2))]:
    got = cm.best_grid(alg, n, d, d, P)
    expect(got == want, f"{alg} P={P}: got {got}, want {want}")
near(1 - best_words("3d", n, d, 128) / best_words("2d", n, d, 128), 0.18, 0.005, "3D vs 2D at P=128")
near(1 - best_words("3d", n, d, 1024) / best_words("2d", n, d, 1024), 0.32, 0.005, "3D vs 2D at P=1024")
expect(math.isnan(cm.thm_b(16384, 4096, 1024, 64)), "Theorem B undefined for d1 != d2")
expect(cm.best_grid("split", n, d, d, 7) is None, "Split needs even P")

expect(stats.median_ci(range(20, 0, -1)) == (10.5, 6, 15, True), "median CI of 1..20")
expect(not stats.median_ci([1, 2, 3, 4, 5])[3], "5 samples cannot give a 95% CI")
expect(stats.median_ci([1, 2, 3, 4, 5, 6])[3], "6 samples give a 95% CI")

for f in failures:
    print("FAIL", f)
print(f"FAILED ({len(failures)})" if failures else "PASSED")
sys.exit(1 if failures else 0)
