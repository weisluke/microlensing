#!/usr/bin/env python3
"""GPU busy and kernel-overlap fractions from an Nsight Systems trace:

    nsys stats -r cuda_gpu_trace --format csv myrun.nsys-rep > trace.csv
    python tests/batch/gpu_busy.py trace.csv [--after-gap 0.5]

Prints wall_s (first kernel start to last kernel end), busy_fraction (the union of the kernel intervals divided by
wall_s) and overlap_fraction (time with two or more kernels running divided by the busy time). Memory copies and
sets are ignored: only rows with a grid size count as kernels. With several devices the whole and every device
are reported. --after-gap S drops everything up to the first idle gap of S seconds (run the benchmark child with
BENCH_PROFILE_GAP=1, which sleeps 1 s before the timed region, and use --after-gap 0.5)."""
import argparse
import csv
import io
import sys
from pathlib import Path

GRID_COLUMNS = ("GrdX", "Grid X")


def read_trace_devices(path):
    """{device: [(start_s, end_s)]} of the kernels of a cuda_gpu_trace CSV; text before the header line is skipped.
    Without a "Device" column everything is one device "-". Raises ValueError when the trace has no grid-size
    column, because then kernels cannot be told from memory copies and sets."""
    lines = Path(path).read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"Start (ns)"') or ln.startswith("Start (ns)")), None)
    if start is None:
        raise ValueError(f"{path}: no 'Start (ns)' header line")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    grid = next((c for c in GRID_COLUMNS if c in (reader.fieldnames or [])), None)
    if grid is None:
        raise ValueError(f"{path}: none of the kernel-identifying columns {GRID_COLUMNS} in the header "
                         f"{reader.fieldnames}; was the trace made with `nsys stats -r cuda_gpu_trace`?")
    devices = {}
    for row in reader:
        if not (row[grid] or "").strip():
            continue
        t0 = float(row["Start (ns)"]) * 1e-9
        dev = (row.get("Device") or "-").strip() if "Device" in row else "-"
        devices.setdefault(dev, []).append((t0, t0 + float(row["Duration (ns)"]) * 1e-9))
    return devices


def read_trace(path):
    """All kernel intervals (start_s, end_s) of the trace, in file order."""
    return [iv for ivs in read_trace_devices(path).values() for iv in ivs]


def after_gap(intervals, gap):
    """The intervals that start at or after the end of the first idle period of at least `gap` seconds (no kernel
    running); ValueError when there is no such gap."""
    ivs = sorted(intervals)
    reach = ivs[0][1]
    for k, (s, e) in enumerate(ivs[1:], 1):
        if s - reach >= gap:
            return ivs[k:]
        reach = max(reach, e)
    raise ValueError(f"no idle gap of at least {gap} s between kernels")


def busy_stats(intervals):
    """wall_s, busy_fraction and overlap_fraction of (start, end) intervals in seconds."""
    if not intervals:
        raise ValueError("no kernel intervals")
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals], key=lambda x: (x[0], x[1]))
    wall0, wall1 = min(s for s, _ in intervals), max(e for _, e in intervals)
    busy = overlap = 0.0
    depth, last = 0, wall0
    for t, step in events:
        if depth >= 1:
            busy += t - last
        if depth >= 2:
            overlap += t - last
        depth += step
        last = t
    wall = wall1 - wall0
    return {"wall_s": wall, "busy_fraction": busy / wall if wall > 0 else 0.0,
            "overlap_fraction": overlap / busy if busy > 0 else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="CSV from `nsys stats -r cuda_gpu_trace --format csv`")
    ap.add_argument("--after-gap", type=float, default=None, metavar="SECONDS",
                    help="analyse only the kernels after the first idle gap (on all devices) of at least SECONDS")
    args = ap.parse_args(argv)
    devices = read_trace_devices(args.trace)
    if args.after_gap is not None:
        every = after_gap([iv for ivs in devices.values() for iv in ivs], args.after_gap)
        t_start = every[0][0]
        devices = {d: [iv for iv in ivs if iv[0] >= t_start] for d, ivs in devices.items()}
        devices = {d: ivs for d, ivs in devices.items() if ivs}
    everything = [iv for ivs in devices.values() for iv in ivs]
    for name, ivs in [("all", everything)] + (sorted(devices.items()) if len(devices) > 1 else []):
        stats = busy_stats(ivs)
        for k in ("wall_s", "busy_fraction", "overlap_fraction"):
            print(f"{k} {stats[k]:.6f}" if name == "all" else f"device {name}: {k} {stats[k]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
