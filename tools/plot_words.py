"""Words received by the busiest rank vs P: algorithms on their best grids against Theorems A and B."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import costmodel as cm  # noqa: E402

INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ALGS = [("2d", "2D", "#2a78d6", "o"), ("3d", "3D", "#eb6834", "s"), ("split", "Split 3D", "#1baf7a", "D"),
        ("ring", "Ring", "#eda100", "^")]


def series(n, d, Ps):
    out = {}
    for alg, *_ in ALGS:
        pts = []
        for P in Ps:
            if alg == "3d" and P < n * n / (d * d):
                continue  # below n²/d² the best grid is 2D; forcing p3 > 1 only adds clutter
            g = None if alg == "ring" else cm.best_grid(alg, n, d, d, P)
            if alg == "ring" or g:
                pts.append((P, cm.predicted_words(alg, n, d, d, P, g)))
        out[alg] = pts
    out["A"] = [(P, cm.thm_a(n, d, d, P)) for P in Ps]
    out["B"] = [(P, cm.thm_b(n, d, d, P)) for P in Ps]
    return out


def style(ax, Ps):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(Ps))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel("P (processors per head)", color=INK_2, fontsize=10)


def regimes(ax, n, d, top):
    r = n * n / (d * d)
    for x, label in [(r, "n²/d²"), (4.2 * r, "4.2·n²/d²")]:
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(x * 1.06, top, label, color=MUTED, fontsize=8.5, va="top")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--Pmax", type=int, default=4096)
    ap.add_argument("--out", default="figures/words_vs_P.png")
    args = ap.parse_args()
    n, d = args.n, args.d
    Ps = [2 ** k for k in range(2, args.Pmax.bit_length())]
    s = series(n, d, Ps)

    plt.rcParams["font.family"] = ["Helvetica Neue", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.5, 5.4), facecolor=SURFACE)
    style(ax, Ps)

    def plot(key, **kw):
        ax.plot([p for p, _ in s[key]], [w for _, w in s[key]], linewidth=2, **kw)

    for alg, label, color, marker in ALGS:
        plot(alg, color=color, marker=marker, markersize=6, label=label, markeredgecolor=SURFACE,
             markeredgewidth=1.2, zorder=4 if alg == "3d" else 3)
    plot("A", color=INK, linestyle=(0, (6, 3)), label="Theorem A (per-loop balance)")
    plot("B", color=INK_2, linestyle=(0, (1.5, 2.5)), label="Theorem B (total balance)")

    ax.yaxis.set_major_locator(FixedLocator([2 ** k * 1e6 for k in range(0, 9)]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_ylabel("words received by the busiest rank", color=INK_2, fontsize=10)
    ax.set_title(f"Communication per rank   (n = {n}, d = {d})", loc="left", color=INK, fontsize=12,
                 fontweight="bold", pad=34)
    regimes(ax, n, d, ax.get_ylim()[1])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK_2,
              handlelength=2.6, columnspacing=1.4, borderaxespad=0.2)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
