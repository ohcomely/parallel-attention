#!/usr/bin/env python3
"""Run an experiment described by a TOML spec (see experiments/*.toml).

    python3 experiments/run.py experiments/sanity_mac.toml            # run locally
    python3 experiments/run.py experiments/e1_strong.toml --dry-run   # show runs, grids and commands only

A spec has top-level settings, a [defaults] table and one or more [[run]] tables:

    name     = "e1_strong"                 # output directory results/<name>/ unless --out is given
    jobs     = 2                           # independent repetitions of the whole experiment (allocations)
    shuffle  = true                        # randomise the order of P values and of runs within each P
    launcher = ["mpirun", "-np", "{P}"]    # command prefix; {P} is replaced by the rank count
    env      = { VECLIB_MAXIMUM_THREADS = "1" }

    [defaults]                              # applies to every run
    n = 16384
    d = 4096

    [[run]]                                 # list values expand as a Cartesian product
    alg = ["ring", "2d"]
    P   = [8, 16, 32]

Every key other than P becomes an attnbench option (`max_reps = 60` -> `--max-reps 60`, `check = true` -> `--check`).
`timeline = true` writes timeline_<alg>_P<P>_T<T>_job<j>.csv next to results.csv. All runs with the same P in one job
go into a single `attnbench --batch` call. Each job's rows are labelled <name>-<job> in the job_id column.
"""
import argparse
import itertools
import os
import random
import shlex
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL = {"name", "jobs", "shuffle", "launcher", "env", "defaults", "run"}


def expand(spec):
    """All run configurations (dicts) of the spec, in spec order."""
    defaults = spec.get("defaults", {})
    runs = []
    for block in spec.get("run", []):
        merged = {**defaults, **block}
        require("P" in merged, f"run {block} has no P")
        keys = list(merged)
        values = [v if isinstance(v, list) else [v] for v in merged.values()]
        runs += [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return runs


def require(ok, msg):
    if not ok:
        sys.exit(f"error: {msg}")


def options(run, out_dir, job):
    """attnbench options for one run (without P, --out and --batch)."""
    args = []
    for key, value in run.items():
        if key == "P":
            continue
        flag = "--" + key.replace("_", "-")
        if key == "timeline" and value is True:
            value = out_dir / f"timeline_{run['alg']}_P{run['P']}_T{run.get('T', 1)}_job{job}.csv"
        if value is True:
            args.append(flag)
        elif value is not False:
            args += [flag, str(value)]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--out", type=Path, help="output directory (default: results/<name>)")
    ap.add_argument("--bench", type=Path, default=ROOT / "build" / "attnbench")
    ap.add_argument("--seed", type=int, default=0, help="seed for the run order")
    ap.add_argument("--dry-run", action="store_true", help="print plans and commands, run nothing")
    a = ap.parse_args()

    spec = tomllib.loads(a.spec.read_text())
    unknown = set(spec) - TOP_LEVEL
    require(not unknown, f"unknown top-level keys {sorted(unknown)} in {a.spec}")
    name = spec.get("name", a.spec.stem)
    jobs = spec.get("jobs", 1)
    launcher = spec.get("launcher", ["mpirun", "-np", "{P}"])
    out_dir = (a.out or ROOT / "results" / name).resolve()
    runs = expand(spec)
    require(runs, f"{a.spec} defines no runs")
    require(a.bench.exists() or a.dry_run, f"{a.bench} not found; build first (cmake --build build)")

    if a.dry_run and a.bench.exists():  # show what the cost model picks for each run
        for run in runs:
            plan = subprocess.run([a.bench, "--plan", "--P", str(run["P"]), *options(run, out_dir, 1)],
                                  capture_output=True, text=True)
            print(("  " if plan.returncode == 0 else "! ") + (plan.stdout or plan.stderr).strip())

    batch_dir = out_dir / "batches"
    if not a.dry_run:
        batch_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{k: str(v) for k, v in spec.get("env", {}).items()}}
    rng = random.Random(a.seed)
    failed = []
    for job in range(1, jobs + 1):
        by_P = defaultdict(list)
        for run in runs:
            by_P[run["P"]].append(run)
        order = list(by_P)
        if spec.get("shuffle", False):
            rng.shuffle(order)
            for P in order:
                rng.shuffle(by_P[P])
        job_id = f"{name}-{job}"
        for P in order:
            batch = batch_dir / f"job{job}_P{P}.txt"
            lines = [shlex.join(options(run, out_dir, job)) for run in by_P[P]]
            cmd = [part.replace("{P}", str(P)) for part in launcher]
            cmd += [str(a.bench), "--batch", str(batch), "--out", str(out_dir / "results.csv")]
            print(f"[{job_id}] P={P}: {len(lines)} run(s)\n  $ {shlex.join(cmd)}")
            if a.dry_run:
                print("".join(f"    {line}\n" for line in lines), end="")
                continue
            batch.write_text("".join(line + "\n" for line in lines))
            if subprocess.run(cmd, env={**env, "ATTN_JOB_ID": job_id}, stdin=subprocess.DEVNULL).returncode:
                failed.append(f"{job_id} P={P}")

    if not a.dry_run:
        print(f"results: {out_dir / 'results.csv'}")
        print(f"summary: python3 {ROOT / 'tools' / 'stats.py'} {out_dir / 'results.csv'}")
    require(not failed, f"failed batches: {', '.join(failed)}")


if __name__ == "__main__":
    main()
