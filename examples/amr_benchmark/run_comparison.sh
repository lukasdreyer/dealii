#!/bin/bash
# Run amr_benchmark with deal.II built with t8code and with p4est (Release
# builds) on 1, 2 and 4 processes, and summarize the timings of each phase
# (sums over the cycles of the maxima over the processes, the minimum over
# the repetitions).
#
# Usage: ./run_comparison.sh [output directory] [benchmark arguments...]
# The deal.II build directories are taken from T8_BUILD and P4EST_BUILD.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/results}; shift || true
args=${@:-"n=4 global=2 cycles=6 max_level=5"}
T8_BUILD=${T8_BUILD:-$here/../../build_t8_release}
P4EST_BUILD=${P4EST_BUILD:-$here/../../build_p4est_release}
MPIEXEC=${MPIEXEC:-mpiexec}
REPETITIONS=${REPETITIONS:-3}
mkdir -p "$out"

for backend in t8 p4est; do
  build=$here/build_$backend
  dealii=$([ $backend = t8 ] && echo "$T8_BUILD" || echo "$P4EST_BUILD")
  mkdir -p "$build"
  (cd "$build" && cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR="$dealii" .. > cmake.log && ninja > build.log)
  for np in 1 2 4; do
    for rep in $(seq 1 $REPETITIONS); do
      "$MPIEXEC" -np $np "$build/amr_benchmark" $args csv="$out/${backend}_np${np}_rep${rep}.csv" > "$out/${backend}_np${np}_rep${rep}.log" 2>&1
    done
  done
done

python3 "$here/summarize.py" "$out"
