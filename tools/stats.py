"""Summarise attnbench results.csv files: per configuration the median of t_total_max with a nonparametric 95% CI
(Hoefler & Belli rules 5-8), medians of the critical rank's components, and CI-based comparisons."""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict

CONFIG = ("alg", "n", "d1", "d2", "P", "p1", "p2", "p3", "T", "blocks", "map", "dtype")
COMPONENTS = ("crit_compute", "crit_comm", "crit_idle_handoff", "crit_idle_tail", "crit_other")


def median_ci(xs):
    """(median, lo, hi, exact). [x_j, x_(n-1-j)] with the largest j such that coverage >= 95%. Mirrors stats.cpp."""
    x = sorted(xs)
    n = len(x)
    j = -1
    while j + 1 < n // 2 and 2 * sum(math.comb(n, i) for i in range(j + 2)) / 2 ** n <= 0.05:
        j += 1
    med = statistics.median(x)
    return (med, x[0], x[-1], False) if j < 0 else (med, x[j], x[n - 1 - j], True)


def overlap(a, b):
    return a[1] <= b[2] and b[1] <= a[2]


def load(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "uniform") for k in CONFIG)].append(r)  # blocks: missing in older CSVs
    out = []
    for key, rs in groups.items():
        s = dict(zip(CONFIG, key))
        s["median"], s["lo"], s["hi"], s["exact"] = median_ci([float(r["t_total_max"]) for r in rs])
        s["reps"] = len(rs)
        s["pred_words"] = rs[0]["pred_words"]
        for c in COMPONENTS:
            s[c] = statistics.median(float(r[c]) for r in rs)
        by_job = defaultdict(list)
        for r in rs:
            by_job[r["job_id"]].append(float(r["t_total_max"]))
        cis = [median_ci(v) for v in by_job.values()]
        s["jobs"] = len(by_job)
        s["jobs_agree"] = all(overlap(a, b) for a in cis for b in cis)
        out.append(s)
    return sorted(out, key=lambda s: (int(s["n"]), int(s["d1"]), int(s["d2"]), int(s["P"]), s["median"]))


def mark_fastest(summaries):
    """Per (n, d1, d2, P): 'best' for the fastest, 'tie' when its CI overlaps the runner-up (rule 7)."""
    groups = defaultdict(list)
    for s in summaries:
        groups[(s["n"], s["d1"], s["d2"], s["P"])].append(s)
    for g in groups.values():
        g.sort(key=lambda s: s["median"])
        for s in g:
            s["rank"] = ""
        if len(g) > 1:
            g[0]["rank"] = "tie" if overlap((0, g[0]["lo"], g[0]["hi"]), (0, g[1]["lo"], g[1]["hi"])) else "best"


def print_table(summaries, out=sys.stdout):
    head = (f"{'n':>6} {'d1':>5} {'d2':>5} {'P':>5} {'alg':>6} {'grid':>9} {'T':>3} {'reps':>4} {'median s':>10} "
            f"{'95% CI':>23} {'compute':>9} {'comm':>9} {'idle':>9} {'other':>9} {'pred_words':>11} {'':>4} jobs")
    print(head, file=out)
    for s in summaries:
        grid = f"{s['p1']}x{s['p2']}x{s['p3']}"
        ci = f"[{s['lo']:.4g}, {s['hi']:.4g}]" + ("" if s["exact"] else "?")
        idle = s["crit_idle_handoff"] + s["crit_idle_tail"]
        jobs = f"{s['jobs']}" + ("" if s["jobs_agree"] else " DISAGREE")
        print(f"{s['n']:>6} {s['d1']:>5} {s['d2']:>5} {s['P']:>5} {s['alg']:>6} {grid:>9} {s['T']:>3} {s['reps']:>4} "
              f"{s['median']:10.4g} {ci:>23} {s['crit_compute']:9.3g} {s['crit_comm']:9.3g} {idle:9.3g} "
              f"{s['crit_other']:9.2g} {float(s['pred_words']):11.0f} {s.get('rank', ''):>4} {jobs}", file=out)


def write_csv(summaries, path):
    keys = list(CONFIG) + ["reps", "median", "lo", "hi", "exact", *COMPONENTS, "pred_words", "jobs", "jobs_agree",
                           "rank"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="results.csv files (pooled)")
    ap.add_argument("--csv", help="write the summary table here")
    args = ap.parse_args()
    summaries = summarize(load(args.results))
    mark_fastest(summaries)
    print_table(summaries)
    print("\nCI: nonparametric 95% CI of the median ('?' = fewer than 6 reps). 'best' = fastest with a CI disjoint "
          "from the runner-up; 'tie' = overlapping CIs, no claim. DISAGREE = job CIs do not overlap.")
    if args.csv:
        write_csv(summaries, args.csv)


if __name__ == "__main__":
    main()
