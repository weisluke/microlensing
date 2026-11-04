#!/usr/bin/env bash
# Nsight Systems profile of one rung build: usage profile_rung.sh <build_dir> <rung> [config ...]
# Default configurations: nb_f hm100_f. Writes results/<rung>/nsys/<config>.nsys-rep and CSV reports.
set -euo pipefail
build=$1; rung=$2; shift 2
if [ $# -eq 0 ]; then set -- nb_f hm100_f; fi
root=${IPM_BENCH_ROOT:?set IPM_BENCH_ROOT to the directory that holds builds/, ref/ and results/}
scratch=${IPM_BENCH_TMP:-/tmp/ipm_bench}/$rung/nsys
out=$root/results/$rung/nsys
mkdir -p "$out" "$scratch"
here=$(cd "$(dirname "$0")" && pwd)
for cfg in "$@"; do
  read -r exe args <<<"$(python "$here/configs.py" --command "$cfg" --build "$build" --ref-root "$root/ref")"
  mkdir -p "$scratch/$cfg"
  nsys profile -t cuda,osrt --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
       --force-overwrite true -o "$out/$cfg" "$exe" $args -v 1 -o "$scratch/$cfg/"
  for rep in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum; do
    nsys stats -r "$rep" --format csv --force-export true -o "$out/$cfg" "$out/$cfg.nsys-rep" > /dev/null
  done
  rm -rf "$scratch/$cfg"
done
ls -la "$out"
