"""Analysis figures for one experiment directory (results.csv + timeline_*.csv), e.g. results/merlin_cpu.

    python tools/plot_results.py results/merlin_cpu            # -> figures/merlin_cpu/*.png

Figures:
  scaling.png      median time vs P per algorithm, 95% CI bars, ideal-scaling guide
  breakdown.png    critical-rank compute / comm / idle per algorithm, one panel per P
  comm_words.png   measured comm time vs predicted words, both relative to 2D at the same P
  bandwidth.png    effective receive bandwidth per rank (4 * words / comm time)
  timeline.png     per-rank phase timeline at the largest P with timelines (first node)
  jobs.png         run-to-run variation: job 2 median / job 1 median per configuration
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stats import median_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Categorical slots in fixed order (identity follows the algorithm in every figure).
ALG_STYLE = {  # key: (label, color, linestyle, marker, filled)
    ("2d", 1): ("2D", "#2a78d6", "-", "o", True),
    ("3d", 1): ("3D", "#eb6834", "-", "s", True),
    ("split", 8): ("Split T=8", "#1baf7a", "-.", "D", True),
    ("split", 4): ("Split T=4", "#1baf7a", "-", "D", True),
    ("split", 2): ("Split T=2", "#1baf7a", ":", "D", False),
    ("split", 1): ("Split T=1", "#1baf7a", "--", "D", False),
    ("ring", 1): ("Ring", "#c98500", "-", "^", True),
}
ORDER = list(ALG_STYLE)
NON_2D = [k for k in ORDER if k[0] not in ("2d", "ring")]  # compared against 2D in comm_words.png
COMPONENT_STYLE = {"compute": ("compute", "#b9b8b2"), "comm": ("communication", "#6250d6"),
                   "idle": ("idle (wait for S, tail)", "#e87ba4")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def key(r):
    return (r["alg"], int(r["T"]))


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows):
    """{(alg, T, P): dict(median, lo, hi, compute, comm, idle, words, per-job medians)}"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["alg"], int(r["T"]), int(r["P"]))].append(r)
    out = {}
    for k, rs in groups.items():
        t = [float(r["t_total_max"]) for r in rs]
        med, lo, hi, _ = median_ci(t)
        jobs, jobs_comm = defaultdict(list), defaultdict(list)
        for r in rs:
            jobs[r["job_id"]].append(float(r["t_total_max"]))
            jobs_comm[r["job_id"]].append(float(r["crit_comm"]))
        out[k] = dict(
            median=med, lo=lo, hi=hi, words=float(rs[0]["pred_words"]),
            compute=statistics.median(float(r["crit_compute"]) for r in rs),
            comm=statistics.median(float(r["crit_comm"]) for r in rs),
            idle=statistics.median(float(r["crit_idle_handoff"]) + float(r["crit_idle_tail"]) for r in rs),
            rpn=int(rs[0]["ranks_per_node"]), nodes=int(rs[0]["nodes"]),
            jobs={j: statistics.median(v) for j, v in sorted(jobs.items())},
            jobs_comm={j: statistics.median(v) for j, v in sorted(jobs_comm.items())},
            grid=f'{rs[0]["p1"]}x{rs[0]["p2"]}x{rs[0]["p3"]}')
    return out


def end_labels(ax, items, gap=0.055, log=True):
    """Direct labels at the right end of lines, pushed apart vertically. items: (x, y, text)."""
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    items = sorted(items, key=lambda t: f(t[1]))
    pos = [f(y) for _, y, _ in items]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    for (x, _, text), y in zip(items, pos):
        ax.annotate(text, (x, inv(y)), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK)


def p_axis(ax, Ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ps)
    ax.set_xticklabels([str(p) for p in Ps])
    ax.minorticks_off()


def fig_scaling(S, title, path):
    Ps = sorted({k[2] for k in S})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = []
    for k in ORDER:
        pts = sorted((p, S[(*k, p)]) for p in Ps if (*k, p) in S)
        if not pts:
            continue
        label, color, ls, marker, filled = ALG_STYLE[k]
        x = [p for p, _ in pts]
        y = [s["median"] for _, s in pts]
        err = [[s["median"] - s["lo"] for _, s in pts], [s["hi"] - s["median"] for _, s in pts]]
        ax.errorbar(x, y, yerr=err, color=color, ls=ls, marker=marker, mfc=color if filled else SURFACE,
                    mec=color, mew=1.6, capsize=3, elinewidth=1.2, label=label)
        labels.append((x[-1], y[-1], label))
    base = S.get(("2d", 1, Ps[0]))
    if base:
        ax.plot(Ps, [base["median"] * Ps[0] / p for p in Ps], color=INK2, lw=1, ls=":", zorder=0)
        ax.annotate("ideal scaling from 2D", (Ps[-1], base["median"] * Ps[0] / Ps[-1]), xytext=(8, -2),
                    textcoords="offset points", color=INK2, fontsize=8, va="top")
    ax.set_yscale("log")
    end_labels(ax, labels)
    p_axis(ax, Ps)
    ax.set_xlim(Ps[0] / 1.3, Ps[-1] * 2.2)
    ax.set_xlabel("P (ranks)")
    ax.set_ylabel("median time per run, slowest rank (s)")
    ax.set_title(title)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_breakdown(S, title, path):
    Ps = sorted({k[2] for k in S})
    cols = 3
    rows = math.ceil(len(Ps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.4 * rows), squeeze=False)
    for ax, P in zip(axes.flat, Ps):
        ks = [k for k in ORDER if (*k, P) in S]
        x = range(len(ks))
        bottom = [0.0] * len(ks)
        for comp, (label, color) in COMPONENT_STYLE.items():
            v = [S[(*k, P)][comp] for k in ks]
            ax.bar(x, v, bottom=bottom, color=color, width=0.68, edgecolor=SURFACE, linewidth=1.5, label=label)
            bottom = [b + vi for b, vi in zip(bottom, v)]
        for i, k in enumerate(ks):
            s = S[(*k, P)]
            ax.annotate(f'{s["median"]:.2f}', (i, s["median"]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([ALG_STYLE[k][0].replace(" ", "\n") for k in ks], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        s0 = S[(*ks[0], P)]
        ax.set_title(f"P = {P}   ({s0['nodes']} node{'s' if s0['nodes'] > 1 else ''}, "
                     f"{s0['rpn']} rank{'s' if s0['rpn'] > 1 else ''}/node)", fontsize=10)
        ax.set_ylim(0, max(bottom + [S[(*k, P)]["median"] for k in ks]) * 1.15)
    for ax in axes.flat[len(Ps):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("time (s)")
    handles = [Patch(color=c, label=l) for l, c in COMPONENT_STYLE.values()]
    fig.suptitle(title + " (slowest rank, medians; number = median total time)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_comm_words(S, title, path):
    """Per P: predicted words and measured comm time of 3D / Split, both relative to 2D. Ring: see bandwidth.png."""
    Ps = [p for p in sorted({k[2] for k in S}) if ("2d", 1, p) in S and any((a, t, p) in S for a, t in NON_2D)]
    fig, axes = plt.subplots(1, len(Ps), figsize=(2.6 * len(Ps) + 1, 4.2), sharey=True)
    w = 0.38
    top = 0
    for ax, P in zip(axes, Ps):
        ks = [k for k in NON_2D if (*k, P) in S]
        base = S[("2d", 1, P)]
        for i, k in enumerate(ks):
            s = S[(*k, P)]
            pw, mc = s["words"] / base["words"], s["comm"] / base["comm"]
            ax.bar(i - w / 2, pw, w, color="#c9c8c2", edgecolor=SURFACE, linewidth=1.5)
            ax.bar(i + w / 2, mc, w, color=ALG_STYLE[k][1], edgecolor=SURFACE, linewidth=1.5,
                   hatch="///" if not ALG_STYLE[k][4] else None)
            per_job = [s["jobs_comm"][j] / base["jobs_comm"][j] for j in s["jobs_comm"] if j in base["jobs_comm"]]
            top = max([top, mc, pw] + per_job)
            if len(per_job) > 1:
                ax.plot([i + w / 2] * 2, [min(per_job), max(per_job)], color=INK, lw=1.2)
                ax.scatter([i + w / 2] * len(per_job), per_job, color=INK, s=10, zorder=4)
            ax.annotate(f"{pw:.2f}", (i - w / 2, pw), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK2)
            ax.annotate(f"{mc:.2f}", (i + w / 2, max([mc] + per_job)), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
        ax.axhline(1, color=ALG_STYLE[("2d", 1)][1], lw=1.2, ls="--")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([ALG_STYLE[k][0].replace(" ", "\n") for k in ks], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"P = {P}  ({base['rpn']}/node)", fontsize=10)
    axes[0].set_ylabel("relative to 2D at the same P")
    axes[0].set_ylim(0, max(1.6, top * 1.1))
    handles = [Patch(color="#c9c8c2", label="predicted words"),
               Patch(color=INK2, label="measured comm time, slowest rank (algorithm colour)"),
               Line2D([], [], color=INK, marker="o", ms=3, lw=1.2, label="per allocation (job 1, job 2)"),
               Line2D([], [], color=ALG_STYLE[("2d", 1)][1], ls="--", lw=1.2, label="2D = 1")]
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93), fontsize=8.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_bandwidth(S, title, path):
    Ps = sorted({k[2] for k in S})
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = []
    for k in ORDER:
        pts = [(p, S[(*k, p)]) for p in Ps if (*k, p) in S and S[(*k, p)]["comm"] > 0 and S[(*k, p)]["words"] > 0]
        if not pts:
            continue
        label, color, ls, marker, filled = ALG_STYLE[k]
        x = [p for p, _ in pts]
        y = [4 * s["words"] / s["comm"] / 1e9 for _, s in pts]
        ax.plot(x, y, color=color, ls=ls, marker=marker, mfc=color if filled else SURFACE, mec=color, mew=1.6,
                label=label)
        labels.append((x[-1], y[-1], label))
    ax.set_yscale("log")
    end_labels(ax, labels, gap=0.06)
    rpn = {p: next(S[k]["rpn"] for k in S if k[2] == p) for p in Ps}
    p_axis(ax, Ps)
    ax.set_xticklabels([f"{p}\n{rpn[p]}/node" for p in Ps])
    ax.set_xlim(Ps[0] / 1.3, Ps[-1] * 2.2)
    ax.set_xlabel("P (ranks), ranks per node")
    ax.set_ylabel("effective bandwidth per rank (GB/s)\n4 B × predicted words / comm time")
    ax.set_title(title)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


COMPUTE_PHASES = {"gemm_s", "gemm_o", "exp", "rowmax", "normalize", "softmax", "init"}


def phase_kind(role, phase):
    if phase in COMPUTE_PHASES:
        return "compute"
    if phase == "S_handoff" and role == "consumer":
        return "idle"
    return "comm"


def fig_timeline(dir_, S, title, path, ranks=16):
    files = sorted(dir_.glob("timeline_*_job1.csv"))
    files = [f for f in files if not f.name.startswith("timeline_ring")]  # ring: thousands of tiny steps
    if not files:
        return False
    order = {"2d": 0, "3d": 1, "split": 2}
    files.sort(key=lambda f: (order.get(f.name.split("_")[1], 9), f.name))
    fig, axes = plt.subplots(len(files), 1, figsize=(11, 1.0 + 1.55 * len(files)), sharex=True)
    for ax, f in zip(axes, files):
        rows = [r for r in load(f) if int(r["rank"]) < ranks]
        alg, P, T = f.name.split("_")[1], int(f.name.split("_")[2][1:]), int(f.name.split("_")[3][1:])
        for r in rows:
            kind = phase_kind(r["role"], r["phase"])
            t0, t1 = float(r["t_start"]), float(r["t_end"])
            ax.barh(int(r["rank"]), t1 - t0, left=t0, height=0.8, color=COMPONENT_STYLE[kind][1], linewidth=0)
        s = S.get((alg, T, P))
        name = ALG_STYLE.get((alg, T), (alg,))[0]
        ax.set_title(f"{name}, P = {P}" + (f" (median total {s['median']:.3f} s)" if s else ""), fontsize=10,
                     loc="left")
        ax.set_ylim(ranks - 0.5, -0.5)
        ax.set_yticks([0, ranks - 1])
        ax.set_ylabel("rank")
        ax.grid(axis="y", visible=False)
        if alg == "split":
            ax.text(1.0, 0.5, "even = producer\nodd = consumer", transform=ax.transAxes, fontsize=7.5,
                    color=INK2, va="center", ha="left")
    axes[-1].set_xlabel("time since the opening barrier (s), first timed repetition")
    handles = [Patch(color=c, label=l) for l, c in COMPONENT_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{title}: ranks 0-{ranks - 1} (one node)", y=1.04, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_jobs(S, title, path):
    items = [(k, s) for k, s in S.items() if len(s["jobs"]) >= 2]
    if not items:
        return False
    items.sort(key=lambda ks: (ks[0][2], ORDER.index(ks[0][:2]) if ks[0][:2] in ORDER else 9))
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(items) + 1.2))
    for i, (k, s) in enumerate(items):
        j = list(s["jobs"].values())
        ratio = j[1] / j[0]
        label, color, _, marker, filled = ALG_STYLE.get(k[:2], (k[0], INK2, "-", "o", True))
        ax.plot([1, ratio], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([ratio], [i], color=color if filled else SURFACE, edgecolors=color, linewidths=1.5,
                   marker=marker, s=36, zorder=3)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"P={k[2]:>3}  {ALG_STYLE.get(k[:2], (k[0],))[0]}" for k, _ in items], fontsize=8,
                       family="monospace")
    ax.axvline(1, color=INK2, lw=1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("median time, job 2 / job 1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="experiment directory with results.csv")
    ap.add_argument("--out", type=Path, help="output directory (default figures/<dir name>)")
    a = ap.parse_args()
    rows = load(a.dir / "results.csv")
    S = summarize(rows)
    r0 = rows[0]
    out = a.out or ROOT / "figures" / a.dir.name
    out.mkdir(parents=True, exist_ok=True)
    shape = f'n = {r0["n"]}, d = {r0["d1"]}'
    fig_scaling(S, f"Strong scaling, {shape}", out / "scaling.png")
    fig_breakdown(S, f"Where the time goes, {shape}", out / "breakdown.png")
    fig_comm_words(S, f"Does communication time follow the predicted words? ({shape})", out / "comm_words.png")
    fig_bandwidth(S, "Effective bandwidth per rank", out / "bandwidth.png")
    fig_timeline(a.dir, S, "Phase timeline", out / "timeline.png")
    fig_jobs(S, "Run-to-run variation between allocations", out / "jobs.png")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
