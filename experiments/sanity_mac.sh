#!/usr/bin/env bash
# Sanity check (brief §7.1): correctness, word counts and timers on the CPU path for P in {1,2,4,8}.
# Extra CMake flags can be passed through ATTN_CMAKE_ARGS, e.g. a -DCMAKE_OSX_SYSROOT override.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
build="$root/build"
out="$build/sanity"

cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release ${ATTN_CMAKE_ARGS:-}
cmake --build "$build" -j
ctest --test-dir "$build" --output-on-failure

mkdir -p "$out"
rm -f "$out/results.csv"
run() { mpirun -np "$1" --oversubscribe "$build/attnbench" --n 2048 --d 512 --warmup 1 --reps 3 --check \
          --out "$out/results.csv" "${@:2}"; }

run 1 --alg 2d --grid 1x1x1
run 4 --alg 2d --grid 2x2x1
run 8 --alg 3d --grid 2x2x2
run 8 --alg 3d --grid 2x2x2 --d2 128
run 8 --alg ring
run 8 --alg split --grid 2x2x1 --T 1
run 8 --alg split --grid 2x2x1 --T 4 --timeline "$out/timeline.csv"

echo "sanity outputs in $out"
