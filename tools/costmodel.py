"""Words received by the busiest rank: lower bounds (Theorems A, B) and the algorithms. Mirrors src/costmodel.cpp."""
import argparse
import math


def c(r, N):
    return (1 - 1 / r) * N


def G(t, d):
    return 3 * t ** (2 / 3) if t <= d ** 3 else t / d + 2 * math.sqrt(t * d)


def thm_a(n, d1, d2, P):
    return G(n * n * d1 / P, d1) + G(n * n * d2 / P, d2) - (2 * n * d1 + 2 * n * d2 + 2 * n * n) / P


def thm_b(n, d1, d2, P):
    if d1 != d2:
        return math.nan  # not derived for unequal head dimensions
    d, F = d1, n * n * d1 / P
    if P <= n * n / (4 * d * d):
        return 4 * n * d / math.sqrt(P) - 4 * n * d / P
    return 3 * (2 * F) ** (2 / 3) - (4 * n * d + n * n) / P


def words_3d(n, d1, d2, g):
    p1, p2, p3 = g
    return (c(p2, n * d1 / (p1 * p3)) + c(p2, n * d2 / (p1 * p3)) + c(p1, n * d1 / (p2 * p3))
            + c(p1, n * d2 / (p2 * p3)) + 2 * c(p3, n * n / (p1 * p2)))


def words_split_roles(n, d1, d2, g):
    p1, p2, p3 = g
    producer = c(p2, n * d1 / (p1 * p3)) + c(p1, n * d1 / (p2 * p3)) + c(p3, n * n / (p1 * p2))
    consumer = (n * n / (p1 * p2 * p3) + c(p3, n * n / (p1 * p2)) + c(p1, n * d2 / (p2 * p3))
                + c(p2, n * d2 / (p1 * p3)))
    return producer, consumer


def words_ring(n, d1, d2, P):
    return c(P, n * (d1 + d2))


def predicted_words(alg, n, d1, d2, P, g=None):
    if alg == "ring":
        return words_ring(n, d1, d2, P)
    if alg == "split":
        return max(words_split_roles(n, d1, d2, g))
    return words_3d(n, d1, d2, g)


def valid(n, d1, d2, g, query_rows):
    p1, p2, p3 = g
    if query_rows % p1 or n % p2 or d1 % p3 or d2 % p3:
        return False
    a, b = query_rows // p1, n // p2
    return a % p2 == 0 and b % p1 == 0 and a % p3 == 0


def all_grids(P):
    return [(p1, p2, P // (p1 * p2)) for p1 in range(1, P + 1) for p2 in range(1, P // p1 + 1)
            if P % (p1 * p2) == 0]


def best_grid(alg, n, d1, d2, P, T=1):
    """Fewest predicted words; ties: smaller p3, then smaller p1. 3d means p3 > 1, 2d means p3 = 1."""
    ranks, rows = (P // 2, n // T) if alg == "split" else (P, n)
    if alg == "split" and (P % 2 or n % T):
        return None
    keep = {"2d": lambda g: g[2] == 1, "3d": lambda g: g[2] > 1}.get(alg, lambda g: True)
    cands = [g for g in all_grids(ranks) if keep(g) and valid(n, d1, d2, g, rows)]
    if not cands:
        return None
    fewest = min(predicted_words(alg, n, d1, d2, P, g) for g in cands)
    ties = [g for g in cands if predicted_words(alg, n, d1, d2, P, g) <= fewest * (1 + 1e-9)]
    return min(ties, key=lambda g: (g[2], g[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--d2", type=int)
    ap.add_argument("--P", type=int, nargs="+", required=True)
    args = ap.parse_args()
    n, d1 = args.n, args.d
    d2 = args.d2 or d1
    print(f"n={n} d1={d1} d2={d2}   words received by the busiest rank")
    print(f"{'P':>6} {'thmA':>12} {'thmB':>12} {'2d':>22} {'3d':>22} {'split':>22} {'ring':>12}")
    for P in args.P:
        cells = []
        for alg in ("2d", "3d", "split"):
            g = best_grid(alg, n, d1, d2, P)
            cells.append(f"{predicted_words(alg, n, d1, d2, P, g):11.0f} {'x'.join(map(str, g)):>10}" if g else
                         f"{'-':>22}")
        print(f"{P:>6} {thm_a(n, d1, d2, P):12.0f} {thm_b(n, d1, d2, P):12.0f} {' '.join(cells)} "
              f"{words_ring(n, d1, d2, P):12.0f}")


if __name__ == "__main__":
    main()
