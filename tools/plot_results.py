"""Analysis figures for one experiment directory (results.csv, optional timeline_*.csv).

    python tools/plot_results.py results/merlin_cpu2                              # -> figures/merlin_cpu2/
    python tools/plot_results.py results/merlin_cpu2 --compare results/merlin_cpu  # + placement figure

Each figure is written as .png (review) and .pdf (paper):
  1_scaling      time vs P, and speedup over 2D at the same P (Split at its best T)
  2_breakdown    slowest rank's compute / communication / idle, one panel per P
  3_comm_model   measured comm time vs bytes the model predicts, with a bandwidth fit for 2D and 3D
  4_split_T      Split's pipeline depth T: idle vs compute trade-off, against 2D and 3D
  5_variation    per-allocation medians relative to the pooled median
  6_placement    same P, different ranks per node (needs --compare)
  7_timeline     per-rank phase timeline (only if the directory has timeline files)
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stats import median_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Identity follows the algorithm in every figure (categorical slots in fixed order).
ALG = {"2d": ("2D", "#2a78d6", "o"), "3d": ("3D", "#eb6834", "s"), "split": ("Split", "#1baf7a", "D"),
       "ring": ("Ring", "#c98500", "^")}
ALGS = list(ALG)
PART = {"compute": ("compute", "#bdbcb6"), "comm": ("communication", "#6250d6"), "idle": ("idle", "#e87ba4")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.8,
    "lines.markersize": 5.5, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


# ---------------------------------------------------------------------------------------------------- data

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows):
    """{(alg, T, P): stats of the slowest rank pooled over allocations, plus per-allocation medians}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["alg"], int(r["T"]), int(r["P"]))].append(r)
    out = {}
    for k, rs in groups.items():
        med, lo, hi, _ = median_ci([float(r["t_total_max"]) for r in rs])
        per_job = defaultdict(list)
        for r in rs:
            per_job[r["job_id"]].append(r)

        def m(field, sub=rs):
            return statistics.median(float(r[field]) for r in sub)

        out[k] = dict(
            t=med, lo=lo, hi=hi, words=float(rs[0]["pred_words"]), compute=m("crit_compute"), comm=m("crit_comm"),
            idle=statistics.median(float(r["crit_idle_handoff"]) + float(r["crit_idle_tail"]) for r in rs),
            rpn=int(rs[0]["ranks_per_node"]), nodes=int(rs[0]["nodes"]),
            job_t={j: m("t_total_max", v) for j, v in sorted(per_job.items())},
            job_comm={j: m("crit_comm", v) for j, v in sorted(per_job.items())})
    return out


def Ps_of(S):
    return sorted({k[2] for k in S})


def best_split(S, P):
    """(T, stats) of the fastest Split pipeline depth at P, or None."""
    c = [(k[1], s) for k, s in S.items() if k[0] == "split" and k[2] == P]
    return min(c, key=lambda c: c[1]["t"]) if c else None


def entry(S, alg, P):
    """Stats for an algorithm at P; Split at its best T. Returns (stats, T) or (None, None)."""
    if alg == "split":
        b = best_split(S, P)
        return (b[1], b[0]) if b else (None, None)
    s = S.get((alg, 1, P))
    return (s, 1) if s else (None, None)


# ---------------------------------------------------------------------------------------------------- helpers

def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def p_axis(ax, Ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ps)
    ax.set_xticklabels([str(p) for p in Ps])
    ax.minorticks_off()
    ax.set_xlim(Ps[0] / 1.35, Ps[-1] * 1.35)


def end_labels(ax, items, gap, log=True, dx=7):
    """Direct labels right of the last point, pushed apart vertically. items: (x, y, text)."""
    f = math.log10 if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    items = sorted(items, key=lambda t: f(t[1]))
    pos = [f(t[1]) for t in items]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    for (x, _, text), y in zip(items, pos):
        ax.annotate(text, (x, inv(y)), xytext=(dx, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK)


def stacked(ax, x, s, width, clip=None):
    """One stacked compute / comm / idle bar; clipped at `clip`. Returns the drawn height."""
    bottom = 0.0
    for part, (_, color) in PART.items():
        v = s[part]
        if clip is not None:
            v = max(0.0, min(v, clip - bottom))
        ax.bar(x, v, width, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.0)
        bottom += v
    return bottom


def part_legend(fig, y=1.0, extra=()):
    handles = [Patch(color=c, label=l) for l, c in PART.values()] + list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, y), fontsize=8.5,
               handlelength=1.4, columnspacing=1.4)


def nodes_label(P, rpn):
    n = math.ceil(P / rpn)
    return f"{n} node{'s' * (n > 1)}"


# ---------------------------------------------------------------------------------------------------- figures

def fig_scaling(S, shape, out):
    Ps = Ps_of(S)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    labels_a, labels_b = [], []
    for alg in ALGS:
        pts = [(P, *entry(S, alg, P)) for P in Ps]
        pts = [(P, s, T) for P, s, T in pts if s]
        if not pts:
            continue
        name, color, marker = ALG[alg]
        x = [P for P, _, _ in pts]
        y = [s["t"] for _, s, _ in pts]
        a.errorbar(x, y, yerr=[[s["t"] - s["lo"] for _, s, _ in pts], [s["hi"] - s["t"] for _, s, _ in pts]],
                   color=color, marker=marker, capsize=2, elinewidth=1)
        labels_a.append((x[-1], y[-1], name + (" (best T)" if alg == "split" else "")))
        ratio = [(P, S[("2d", 1, P)]["t"] / s["t"], S[("2d", 1, P)]["t"] / s["hi"], S[("2d", 1, P)]["t"] / s["lo"])
                 for P, s, _ in pts if ("2d", 1, P) in S]
        if alg != "2d" and ratio:
            b.errorbar([r[0] for r in ratio], [r[1] for r in ratio],
                       yerr=[[r[1] - r[2] for r in ratio], [r[3] - r[1] for r in ratio]], color=color,
                       marker=marker, capsize=2, elinewidth=1)
            labels_b.append((ratio[-1][0], ratio[-1][1], name + (" (best T)" if alg == "split" else "")))
            if alg == "split":
                for (P, r, _, _), (_, _, T) in zip(ratio, pts):
                    b.annotate(f"T={T}", (P, r), xytext=(0, -12), textcoords="offset points", ha="center",
                               fontsize=6.5, color=INK2)
    base = S.get(("2d", 1, Ps[0]))
    if base:
        a.plot(Ps, [base["t"] * Ps[0] / p for p in Ps], color=INK2, lw=0.9, ls=":", zorder=0)
        labels_a.append((Ps[-1], base["t"] * Ps[0] / Ps[-1], "ideal"))
    a.set_yscale("log")
    end_labels(a, labels_a, gap=0.07)
    p_axis(a, Ps)
    a.set_xlim(Ps[0] / 1.35, Ps[-1] * 3.4)
    a.set_xlabel("P (ranks)")
    a.set_ylabel("median time (s)")
    a.set_title(f"(a) Strong scaling, {shape}")
    b.axhline(1, color=ALG["2d"][1], lw=1.2, ls="--")
    labels_b.append((Ps[-1], 1.0, "2D"))
    end_labels(b, labels_b, gap=0.075, log=False)
    p_axis(b, Ps)
    b.set_xlim(Ps[0] / 1.35, Ps[-1] * 3.0)
    b.set_ylim(0, max(1.3, b.get_ylim()[1]))
    b.set_xlabel("P (ranks)")
    b.set_ylabel("speedup over 2D (same P)")
    b.set_title("(b) Relative to 2D, with 95% CI")
    fig.tight_layout(w_pad=2)
    save(fig, out, "1_scaling")


def fig_breakdown(S, shape, out):
    Ps = Ps_of(S)
    fig, axes = plt.subplots(1, len(Ps), figsize=(1.8 * len(Ps) + 0.4, 2.8))
    axes = np.atleast_1d(axes)
    for ax, P in zip(axes, Ps):
        items = [(alg, *entry(S, alg, P)) for alg in ALGS]
        items = [(alg, s, T) for alg, s, T in items if s]
        others = [s["t"] for alg, s, _ in items if alg != "ring"]
        clip = 1.45 * max(others) if others else None
        for i, (alg, s, T) in enumerate(items):
            top = stacked(ax, i, s, 0.7, clip if alg == "ring" else None)
            clipped = alg == "ring" and clip is not None and s["t"] > clip
            ax.annotate(f'{s["t"]:.2f}' + (" ↑" if clipped else ""), (i, top), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([ALG[a][0] + (f"\nT={T}" if a == "split" else "") for a, _, T in items], fontsize=7.5)
        ax.grid(axis="x", visible=False)
        if clip:
            ax.set_ylim(0, clip * 1.12)
        ax.set_title(f"P = {P}  ({nodes_label(P, items[0][1]['rpn'])})", fontsize=8.5)
    axes[0].set_ylabel("time, slowest rank (s)")
    part_legend(fig, y=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "2_breakdown")


def fig_comm_model(S, shape, out):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xs, ys = [], []
    for k, s in sorted(S.items(), key=lambda kv: kv[0][2]):
        alg, T, P = k
        if s["words"] <= 0 or s["comm"] <= 0:
            continue
        mb = 4 * s["words"] / 1e6
        name, color, marker = ALG[alg]
        hollow = alg == "split" and T != best_split(S, P)[0]
        ax.scatter(mb, s["comm"], color=SURFACE if hollow else color, edgecolors=color, marker=marker, s=26,
                   linewidths=1.2, zorder=3)
        if alg in ("2d", "3d"):
            xs.append(mb), ys.append(s["comm"])
        if alg in ("2d", "3d", "ring") or not hollow:
            ax.annotate(str(P), (mb, s["comm"]), xytext=(4, -1), textcoords="offset points", fontsize=6.5,
                        color=INK2)
    if xs:
        x, y = np.array(xs), np.array(ys)
        b = (x @ y) / (x @ x)  # least squares through the origin: comm = bytes / bandwidth
        r2 = 1 - ((y - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mbs = [4 * s["words"] / 1e6 for s in S.values() if s["words"] > 0]
        grid = np.geomspace(min(mbs) / 1.3, max(mbs) * 1.3, 50)
        ax.plot(grid, b * grid, color=INK2, lw=1, ls="--", zorder=1)
        ax.annotate(f"2D/3D fit: {1 / b / 1e3:.2f} GB/s per rank, R² = {r2:.2f}", (0.98, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted data received per rank (MB) = 4 B × words")
    ax.set_ylabel("measured comm time, slowest rank (s)")
    ax.set_title(f"Communication time vs the word model\n{shape}; numbers = P")
    handles = [Line2D([], [], color=ALG[a][1], marker=ALG[a][2], ls="", label=ALG[a][0] + (" (best T)" if a == "split"
                                                                                          else "")) for a in ALGS]
    handles.append(Line2D([], [], color=ALG["split"][1], marker="D", ls="", mfc=SURFACE, label="Split, other T"))
    handles.append(Line2D([], [], color=INK2, ls="--", label="comm = bytes / bandwidth"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, handletextpad=0.3)
    fig.tight_layout()
    save(fig, out, "3_comm_model")


def fig_split_T(S, shape, out):
    Ps = [P for P in Ps_of(S) if len({k[1] for k in S if k[0] == "split" and k[2] == P}) > 2]
    if not Ps:
        return
    fig, axes = plt.subplots(1, len(Ps), figsize=(3.4 * len(Ps), 2.8))
    axes = np.atleast_1d(axes)
    for ax, P in zip(axes, Ps):
        Ts = sorted(k[1] for k in S if k[0] == "split" and k[2] == P)
        for i, T in enumerate(Ts):
            s = S[("split", T, P)]
            top = stacked(ax, i, s, 0.62)
            ax.annotate(f'{s["t"]:.3f}', (i, max(top, s["t"])), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
        refs = []
        for alg in ("2d", "3d"):
            if (alg, 1, P) in S:
                t = S[(alg, 1, P)]["t"]
                ax.axhline(t, color=ALG[alg][1], lw=1.2, ls="--")
                refs.append((len(Ts) - 0.5, t, f"{ALG[alg][0]} {t:.3f}"))
        end_labels(ax, refs, gap=0.075 * ax.get_ylim()[1], log=False, dx=3)
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([f"T={T}" for T in Ts])
        ax.set_xlim(-0.5, len(Ts) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"Split, P = {P}")
    axes[0].set_ylabel("time, slowest rank (s)")
    part_legend(fig, y=1.0, extra=[Line2D([], [], color=INK2, ls="--", label="2D / 3D median total")])
    fig.tight_layout(w_pad=5)
    save(fig, out, "4_split_T")


def fig_variation(S, out):
    items = [(k, s) for k, s in S.items() if len(s["job_t"]) >= 2]
    if not items:
        return
    order = {a: i for i, a in enumerate(ALGS)}
    items.sort(key=lambda ks: (ks[0][2], order.get(ks[0][0], 9), ks[0][1]))
    jobs = sorted({j for _, s in items for j in s["job_t"]})
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for i, (k, s) in enumerate(items):
        rel = [s["job_t"][j] / s["t"] for j in jobs if j in s["job_t"]]
        ax.plot([i, i], [min(rel), max(rel)], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([i] * len(rel), rel, color=ALG[k[0]][1], s=12, zorder=3)
        for j in jobs:
            if j in s["job_t"] and s["job_t"][j] / s["t"] > 1.08:
                ax.annotate(f'alloc {j.rsplit("-", 1)[-1]}', (i, s["job_t"][j] / s["t"]), xytext=(4, 0),
                            textcoords="offset points", fontsize=6.5, color=INK2, va="center")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([ALG[k[0]][0] + (f" T={k[1]}" if k[0] == "split" else "") for k, _ in items], rotation=90,
                       fontsize=6.5)
    ax.grid(axis="x", visible=False)
    prev = None
    for i, (k, _) in enumerate(items):
        if k[2] != prev:
            if prev is not None:
                ax.axvline(i - 0.5, color=INK2, lw=0.6)
            ax.annotate(f"P = {k[2]}", (i - 0.3, 0.03), xycoords=("data", "axes fraction"), fontsize=7.5,
                        fontweight="bold", va="bottom")
            prev = k[2]
    ax.set_xlim(-0.6, len(items) - 0.4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.06 * (hi - lo), hi)
    ax.set_ylabel("allocation median /\npooled median")
    ax.set_title(f"Run-to-run variation over {len(jobs)} allocations (labelled: more than 8% slower)")
    fig.tight_layout()
    save(fig, out, "5_variation")


def fig_placement(S, C, out, name_s, name_c):
    """2D at each P present in both experiments: compute and comm of the slowest rank, two placements."""
    Ps = [P for P in Ps_of(S) if ("2d", 1, P) in S and ("2d", 1, P) in C]
    if not Ps:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    w = 0.36
    for i, P in enumerate(Ps):
        for dx, src, hatch in ((-w / 2 - 0.02, C, "////"), (w / 2 + 0.02, S, None)):
            s = src[("2d", 1, P)]
            bottom = 0
            for part in ("compute", "comm"):
                ax.bar(i + dx, s[part], w, bottom=bottom, color=PART[part][1], edgecolor=SURFACE, linewidth=0.8,
                       hatch=hatch)
                bottom += s[part]
            ax.annotate(f'{s["rpn"]}/node', (i + dx, bottom), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(Ps)))
    ax.set_xticklabels([f"P = {P}" for P in Ps])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("2D, slowest rank (s)")
    ax.set_title("Same P, different ranks per node (2D)")
    handles = [Patch(color=PART["compute"][1], label="compute"), Patch(color=PART["comm"][1], label="communication"),
               Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////", label=f"{name_c}: spread over 8 nodes"),
               Patch(facecolor=SURFACE, edgecolor=INK2, label=f"{name_s}: 16 ranks/node")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save(fig, out, "6_placement")


def fig_timeline(dir_, S, out, ranks=16):
    files = [f for f in sorted(dir_.glob("timeline_*_job1.csv")) if not f.name.startswith("timeline_ring")]
    if not files:
        return
    order = {"2d": 0, "3d": 1, "split": 2}
    files.sort(key=lambda f: (order.get(f.name.split("_")[1], 9), f.name))
    fig, axes = plt.subplots(len(files), 1, figsize=(7.2, 0.4 + 1.15 * len(files)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, files):
        _, alg, p, t, _ = f.stem.split("_")
        P, T = int(p[1:]), int(t[1:])
        for r in load(f):
            if int(r["rank"]) >= ranks:
                continue
            phase, role = r["phase"], r["role"]
            kind = ("compute" if phase in {"gemm_s", "gemm_o", "exp", "rowmax", "normalize", "softmax", "init"}
                    else "idle" if phase == "S_handoff" and role == "consumer" else "comm")
            t0, t1 = float(r["t_start"]), float(r["t_end"])
            ax.barh(int(r["rank"]), t1 - t0, left=t0, height=0.85, color=PART[kind][1], linewidth=0)
        s = S.get((alg, T, P))
        ax.set_title(f"{ALG[alg][0]}{f' T={T}' if alg == 'split' else ''}, P = {P}"
                     + (f"  (median {s['t']:.3f} s)" if s else "")
                     + ("  even ranks produce S, odd ranks consume it" if alg == "split" else ""), fontsize=8.5)
        ax.set_ylim(ranks - 0.5, -0.5)
        ax.set_yticks([0, ranks - 1])
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("time from the opening barrier (s); ranks 0-15 = first node")
    part_legend(fig, y=1.0)
    fig.tight_layout(h_pad=0.6)
    save(fig, out, "7_timeline")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="experiment directory with results.csv")
    ap.add_argument("--compare", type=Path, help="second experiment for the placement figure")
    ap.add_argument("--out", type=Path, help="output directory (default figures/<dir name>)")
    a = ap.parse_args()
    rows = load(a.dir / "results.csv")
    S = summarize(rows)
    out = a.out or ROOT / "figures" / a.dir.name
    out.mkdir(parents=True, exist_ok=True)
    shape = f'n = {rows[0]["n"]}, d = {rows[0]["d1"]}'
    fig_scaling(S, shape, out)
    fig_breakdown(S, shape, out)
    fig_comm_model(S, shape, out)
    fig_split_T(S, shape, out)
    fig_variation(S, out)
    if a.compare:
        fig_placement(S, summarize(load(a.compare / "results.csv")), out, a.dir.name, a.compare.name)
    fig_timeline(a.dir, S, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
