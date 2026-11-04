#!/bin/bash
# usage: tests/run_cuda_tests.sh <out_dir> <file.cu>...
# Compiles each test program with the project flags and runs it.
# Exits non-zero if any compile or run fails.
if [ $# -lt 2 ]; then
	echo "usage: $0 <out_dir> <file.cu>..." >&2
	exit 2
fi
out_dir=$1; shift
mkdir -p "$out_dir" || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
status=0
for src in "$@"; do
	name=$(basename "${src%.cu}")
	exe="$out_dir/$name"
	echo "== compile $src"
	if ! nvcc -std=c++20 -arch=compute_80 -code=sm_80,compute_80 --use_fast_math -O3 -I "$root/include" -o "$exe" "$src"; then
		echo "COMPILE FAILED: $src"; status=1; continue
	fi
	echo "== run $name"
	if ! "$exe"; then
		echo "RUN FAILED: $name"; status=1
	fi
done
exit $status
