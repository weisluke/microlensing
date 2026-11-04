#!/usr/bin/env python3
"""Checks of the A2 perimeter deposit that the regression does not cover (spec sections 5.2 and 8.4).

    python tests/gpu/a2_checks.py --build $IPM_BENCH_ROOT/builds/A2 --rung A2

- natural_fallback: fallback_f (54000x54000 with parities) with the build's ipm_gpu at -v 2 must succeed, report that the
  step buffers do not fit ("Perimeter deposit disabled: step buffers need ..."), and give finite, non-negative maps with
  pixels = minima + saddles within tol["parity_consistency"].
- nonsquare_double: a 3000x2000 double map with unequal half-lengths, perimeter deposit (ipm_gpu_double) against the
  forced direct deposit (ipm_gpu_double_direct): mean <= 1e-12, p99.9 <= 1e-9, |sum_ratio| <= 1e-12; max reported.
- tiny_maps_double: the same comparison on 1x1 and 7x3 maps, whose rows are shorter than a warp: |sum_ratio| <= 1e-12,
  max <= 1e-9.
- clamp_counts: each ladder configuration at -v 2 with ipm_gpu / ipm_gpu_double; the number of pixels and the total that
  the row sums clamped at 0 are recorded.

Every perimeter run must report the perimeter deposit and every direct run that it was disabled for testing, so that a
comparison cannot pass by running one path twice. Writes results/<rung>/a2_checks.json and exits non-zero on any failure.
A build without all four executables (including the _direct test drivers) fails before any GPU work.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402
from regress import MAP_FILES  # noqa: E402

CHECKS = ["tiny_maps_double", "nonsquare_double", "clamp_counts", "natural_fallback"]   # cheapest first
EXECUTABLES = ["ipm_gpu", "ipm_gpu_double", "ipm_gpu_direct", "ipm_gpu_double_direct"]

PERIMETER_MESSAGE = "Perimeter deposit for bounding boxes above"
FORCED_MESSAGE = "Perimeter deposit disabled for testing."
FALLBACK_MESSAGE = "Perimeter deposit disabled: step buffers need"
FALLBACK_RE = re.compile(r"Perimeter deposit disabled: step buffers need (\d+) MiB, (\d+) MiB free\.")
CLAMP_RE = re.compile(r"Row sums clamped (\d+) pixels, total (\S+)\.")

NONSQUARE = ("-k", "0.4", "-y", "0.4", "-ks", "0.2", "-hly1", "15", "-hly2", "10", "-npy1", "3000", "-npy2", "2000",
             "-rs", cf.SEED, "-ws", "0", "-wm", "1", "-wh", "0")
_LENS, _TAIL = NONSQUARE[:6], NONSQUARE[14:]
TINY = {"1x1": _LENS + ("-hly1", "0.5", "-hly2", "0.5", "-npy1", "1", "-npy2", "1") + _TAIL,
        "7x3": _LENS + ("-hly1", "3.5", "-hly2", "1.5", "-npy1", "7", "-npy2", "3") + _TAIL}
# upper limits on map_metrics(perimeter, direct); sum_ratio bounds |sum_ratio|
NONSQUARE_LIMITS = {"mean": 1e-12, "p999": 1e-9, "sum_ratio": 1e-12}
TINY_LIMITS = {"sum_ratio": 1e-12, "max": 1e-9}
RUN_KEYS = ("exe", "args", "ok", "error", "returncode", "wall_s", "t_shoot_cells", "stderr_tail")


def missing_executables(build):
    paths = [Path(build) / name for name in EXECUTABLES]
    return [str(p) for p in paths if not (p.is_file() and os.access(p, os.X_OK))]


def parse_clamp(stdout):
    """(pixel count, total) from the last "Row sums clamped" line of a run's stdout, or None if there is none."""
    found = CLAMP_RE.findall(stdout or "")
    if not found:
        return None
    count, total = found[-1]
    return int(count), float(total)


def reports_fallback(stdout):
    return FALLBACK_MESSAGE in (stdout or "")


def parse_fallback(stdout):
    """{"need_mib", "free_mib"} from the fallback message, or None."""
    m = FALLBACK_RE.search(stdout or "")
    return {"need_mib": int(m.group(1)), "free_mib": int(m.group(2))} if m else None


def limit_failures(m, limits):
    """Failures of map_metrics() output m against upper limits (NaN fails); the candidate must be finite and non-negative."""
    fails = []
    if not m["finite"]:
        fails.append("non-finite pixels")
    elif not m["nonneg"]:
        fails.append("negative pixels")
    for key, limit in limits.items():
        value = abs(m[key]) if key == "sum_ratio" else m[key]
        if not value <= limit:
            fails.append(f"{'|sum_ratio|' if key == 'sum_ratio' else key}={value:.3e} > {limit:.1e}")
    return fails


def judge_pair(edge_stdout, direct_stdout, edge_map, direct_map, limits, floor_frac=1e-3):
    """Perimeter-deposit map against the forced direct deposit of the same input; returns (metrics, failures, clamp)."""
    fails = []
    if PERIMETER_MESSAGE not in (edge_stdout or ""):
        fails.append(f"perimeter run: no '{PERIMETER_MESSAGE}' message (it fell back or was forced to the direct deposit)")
    if FORCED_MESSAGE not in (direct_stdout or ""):
        fails.append(f"direct run: no '{FORCED_MESSAGE}' message (the direct deposit was not forced)")
    m = cm.map_metrics(edge_map, direct_map, floor_frac)
    fails += limit_failures(m, limits)
    if not (np.isfinite(direct_map).all() and (direct_map >= 0).all()):
        fails.append("direct map: non-finite or negative pixels")
    return m, fails, parse_clamp(edge_stdout)


def judge_natural_fallback(stdout, maps, tol):
    """maps: {"pixels", "minima", "saddles"} arrays of the fallback run; returns (info, failures)."""
    fails = []
    if not reports_fallback(stdout):
        fails.append(f"no '{FALLBACK_MESSAGE}' message: the step buffers fitted, so the fallback was not exercised")
    info = {"fallback": parse_fallback(stdout), "maps": {}}
    for which, arr in maps.items():
        finite = bool(np.isfinite(arr).all())
        nonneg = bool(finite and (arr >= 0).all())
        info["maps"][which] = {"finite": finite, "nonneg": nonneg}
        if not finite:
            fails.append(f"{which}: non-finite pixels")
        elif not nonneg:
            fails.append(f"{which}: negative pixels")
    pc = cm.parity_consistency(maps["pixels"], maps["minima"], maps["saddles"])
    info["parity_consistency"] = pc
    if not pc <= tol["parity_consistency"]:
        fails.append(f"pixels != minima + saddles: {pc:.3e} > {tol['parity_consistency']:.1e}")
    return info, fails


def run(exe, args, outdir, **kw):
    """One run at -v 2 (the messages are printed at that level) with the whole stdout kept."""
    return bl.run_ipm(exe, args, outdir, verbose=2, keep_stdout=True, timeout=3600, **kw)


def run_summary(rec):
    return {k: rec[k] for k in RUN_KEYS}


def check_pairs(build, cases, limits, scratch, tol, keep):
    """Each case (label: arguments) with ipm_gpu_double and ipm_gpu_double_direct, judged by judge_pair()."""
    res = {"pass": True, "failures": [], "limits": limits, "maps": {}}
    for label, args in cases.items():
        edge_dir, direct_dir = Path(scratch) / f"{label}_edge", Path(scratch) / f"{label}_direct"
        try:
            edge = run(cf.exe_for(build, "double"), args, edge_dir)
            direct = run(cf.exe_for(build, "double", "_direct"), args, direct_dir)
            entry = {"edge": run_summary(edge), "direct": run_summary(direct)}
            fails = [f"{name} run failed: {r['error']}" for name, r in (("edge", edge), ("direct", direct)) if not r["ok"]]
            if not fails:
                m, fails, clamp = judge_pair(edge["stdout"], direct["stdout"], bl.read_map(edge_dir / MAP_FILES["pixels"]),
                                             bl.read_map(direct_dir / MAP_FILES["pixels"]), limits, tol["floor_frac"])
                entry.update(metrics=m, clamp=clamp)
        finally:
            if not keep:
                bl.clear_outputs(edge_dir)
                bl.clear_outputs(direct_dir)
        entry["pass"], entry["failures"] = not fails, fails
        res["maps"][label] = entry
        res["failures"] += [f"{label}: {f}" for f in fails]
    res["pass"] = not res["failures"]
    return res


def check_clamp_counts(build, scratch, ref_root, keep):
    res = {"pass": True, "failures": [], "configs": []}
    for cfg in cf.ordered(cf.ladder_configs()):
        outdir = Path(scratch) / cfg.name
        try:
            rec = run(cf.exe_for(build, cfg.build), cf.full_args(cfg, ref_root), outdir)
        finally:
            if not keep:
                bl.clear_outputs(outdir)
        entry = {"config": cfg.name, "build": cfg.build, "run": run_summary(rec), "count": None, "total": None}
        clamp = parse_clamp(rec["stdout"]) if rec["ok"] else None
        if not rec["ok"]:
            res["failures"].append(f"{cfg.name}: run failed: {rec['error']}")
        elif clamp is None:
            res["failures"].append(f"{cfg.name}: no 'Row sums clamped' line (the perimeter deposit was not used)")
        else:
            entry["count"], entry["total"] = clamp
        res["configs"].append(entry)
    res["pass"] = not res["failures"]
    return res


def check_natural_fallback(build, scratch, ref_root, tol, keep):
    cfg = cf.CONFIGS["fallback_f"]
    outdir = Path(scratch) / cfg.name
    try:
        rec = run(cf.exe_for(build, cfg.build), cf.full_args(cfg, ref_root), outdir, sample_gpu=True)
        res = {"config": cfg.name, "run": {**run_summary(rec), "peak_gpu_mem_mib": rec["peak_gpu_mem_mib"]}}
        if not rec["ok"]:
            fails = [f"run failed: {rec['error']}"]
        else:
            maps = {which: bl.read_map(outdir / MAP_FILES[which]) for which in ("pixels", "minima", "saddles")}
            info, fails = judge_natural_fallback(rec["stdout"], maps, tol)
            del maps
            res.update(info)
    finally:
        if not keep:
            bl.clear_outputs(outdir)
    res.update({"pass": not fails, "failures": fails})
    return res


def run_check(name, a, tol, scratch):
    if name == "natural_fallback":
        return check_natural_fallback(a.build, scratch, a.ref_root, tol, a.keep_outputs)
    if name == "nonsquare_double":
        return check_pairs(a.build, {"3000x2000": NONSQUARE}, NONSQUARE_LIMITS, scratch, tol, a.keep_outputs)
    if name == "tiny_maps_double":
        return check_pairs(a.build, TINY, TINY_LIMITS, scratch, tol, a.keep_outputs)
    return check_clamp_counts(a.build, scratch, a.ref_root, a.keep_outputs)


def summarize(name, res):
    parts = []
    if name == "clamp_counts":
        parts = [f"{c['config']} {c['count']} px total {c['total']}" for c in res.get("configs", []) if c["count"] is not None]
    elif name == "natural_fallback" and "parity_consistency" in res:
        fb = res.get("fallback") or {}
        parts = [f"need {fb.get('need_mib')} MiB, free {fb.get('free_mib')} MiB, parity {res['parity_consistency']:.2e}"]
    else:
        for label, e in res.get("maps", {}).items():
            if "metrics" in e:
                m = e["metrics"]
                parts.append(f"{label} mean {m['mean']:.2e} p99.9 {m['p999']:.2e} max {m['max']:.2e} sum {m['sum_ratio']:+.1e}")
    txt = "; ".join(parts)
    if res["failures"]:
        txt += " | " + "; ".join(res["failures"])
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--only", nargs="+", choices=CHECKS, metavar="CHECK", help=f"run only these checks ({', '.join(CHECKS)})")
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--tolerances", help="default: $IPM_BENCH_ROOT/results/tolerances.json")
    ap.add_argument("--results", help="default: $IPM_BENCH_ROOT/results/<rung>")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--keep-outputs", action="store_true")
    a = ap.parse_args(argv)
    problem = bl.check_gpu_allocation()
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    a.tolerances = a.tolerances or str(bl.bench_path("results", "tolerances.json"))
    results_path = Path(a.results or bl.bench_path("results", a.rung)) / "a2_checks.json"
    names = [n for n in CHECKS if a.only is None or n in a.only]
    out = {"rung": a.rung, "build": bl.build_meta(a.build), "selected": names, "checks": {}, "pass": False, "complete": False}
    missing = missing_executables(a.build)
    if missing:   # before any GPU work
        out["missing_executables"] = missing
        bl.write_json(results_path, out)
        print("missing executable(s): " + ", ".join(missing), file=sys.stderr)
        print("A2 CHECKS FAIL")
        return 1
    tol = cm.load_tolerances(a.tolerances)
    out["env"] = bl.environment_info()
    out["parity_consistency_limit"] = tol["parity_consistency"]
    bl.write_json(results_path, out)   # an earlier result file must not outlive an abort
    scratch = Path(a.scratch or bl.SCRATCH / a.rung / "a2_checks")
    for name in names:
        try:
            res = run_check(name, a, tol, scratch / name)
        except Exception as exc:   # recorded as a failure so the remaining checks still run
            res = {"pass": False, "failures": [f"check failed: {type(exc).__name__}: {exc}"]}
        out["checks"][name] = res
        print(f"{'PASS' if res['pass'] else 'FAIL':4s} {name:18s} {summarize(name, res)}", flush=True)
        bl.write_json(results_path, out)
    out["pass"] = all(c["pass"] for c in out["checks"].values())
    out["complete"] = True
    bl.write_json(results_path, out)
    print("A2 CHECKS", "PASS" if out["pass"] else "FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
