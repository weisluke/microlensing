#!/usr/bin/env bash
# Out-of-tree build of the four IPM targets with the flags of ./compile plus -Xptxas -v.
# usage: build_rung.sh <src_dir> <out_dir> [extra nvcc flags...]
# Needs nvcc on PATH. Extra host-compiler or linker flags can be passed through NVCC_PREPEND_FLAGS.
set -euo pipefail
src=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
shift 2
extra=("$@")
command -v nvcc >/dev/null || { echo "nvcc not on PATH" >&2; exit 1; }
flags=(-std=c++20 -arch=compute_80 -code="sm_80,compute_80" --use_fast_math -I "$src/include" -O3 -Xptxas -v)
# provenance first: a source that is not a git checkout fails here, before the long builds
rev=$(git -C "$src" rev-parse HEAD)
state=clean
if [ -n "$(git -C "$src" status --porcelain --untracked-files=no)" ]; then state=dirty; fi
echo "$rev $state" > "$out/git_rev.txt"
nvcc --version | tail -1 > "$out/nvcc_version.txt"
printf '%s\n' "${flags[@]}" ${extra[@]+"${extra[@]}"} "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}" > "$out/flags.txt"
# drop the outputs of earlier runs so that a failed rebuild cannot leave a stale mix
rm -rf "$out/pyroot" "$out/ipm_gpu" "$out/ipm_gpu_double" "$out/lib_ipm.so" "$out/lib_ipm_double.so"
: > "$out/build.log"
build() {   # name define shared source
  echo "=== $1 ===" >> "$out/build.log"
  if [ "$3" = shared ]; then
    if ! nvcc "${flags[@]}" ${extra[@]+"${extra[@]}"} --compiler-options '-fPIC' -D "$2" -shared -o "$out/$1" "$src/src/$4" 2>> "$out/build.log"; then
      echo "build of $1 failed, see $out/build.log" >&2; tail -n 20 "$out/build.log" >&2; exit 1
    fi
  else
    if ! nvcc "${flags[@]}" ${extra[@]+"${extra[@]}"} -D "$2" -o "$out/$1" "$src/src/$4" 2>> "$out/build.log"; then
      echo "build of $1 failed, see $out/build.log" >&2; tail -n 20 "$out/build.log" >&2; exit 1
    fi
  fi
}
build ipm_gpu is_float exe main_ipm.cu
build ipm_gpu_double is_double exe main_ipm.cu
build lib_ipm.so is_float shared lib_ipm.cu
build lib_ipm_double.so is_double shared lib_ipm.cu
rm -rf "$out/pyroot"
mkdir -p "$out/pyroot"
cp -r "$src/microlensing" "$out/pyroot/microlensing"
find "$out/pyroot/microlensing" -name __pycache__ -type d -exec rm -rf {} +
rm -f "$out/pyroot/microlensing/lib/"*.so "$out/pyroot/microlensing/lib/"*.dll
cp "$out/lib_ipm.so" "$out/lib_ipm_double.so" "$out/pyroot/microlensing/lib/"
touch "$out/pyroot/microlensing/lib/__init__.py"
echo "built $out from $rev ($state)"
grep -c "Compiling entry function" "$out/build.log" | sed 's/^/kernels compiled: /'
