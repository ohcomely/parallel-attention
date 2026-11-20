#!/usr/bin/env bash
# Curiosity run on the MacBook (not for the paper): Ring vs 2D vs 3D at P = 8, n = d = 4096 (3D regime: P > n²/d²),
# plus a single-rank baseline. One BLAS thread per rank. Two "jobs", each in random order.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bench="$root/build/attnbench"
out="${1:-$root/build/mac_timing}"
mkdir -p "$out"
rm -f "$out/results.csv"
export VECLIB_MAXIMUM_THREADS=1

configs=("8 ring" "8 2d" "8 3d" "1 2d")
for job in mac1 mac2; do
  printf '%s\n' "${configs[@]}" | sort -R | while read -r P alg; do
    SLURM_JOB_ID=$job mpirun -np "$P" -x VECLIB_MAXIMUM_THREADS -x SLURM_JOB_ID "$bench" --alg "$alg" \
      --n 4096 --d 4096 --grid auto --warmup 3 --reps 20 --max-reps 60 --out "$out/results.csv" < /dev/null
  done
done
python3 "$root/tools/stats.py" "$out/results.csv"
