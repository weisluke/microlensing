#!/usr/bin/env python3
"""Compare a rung's maps with the R0 references (spec section 8).

Double build, two comparisons that must both pass: the rung's map from a run with the source centre shifted
(configs.SHIFT) against R0's ref/<config>/shift1 under the signed-off tol["double"] thresholds (equivalence check; no
lattice corner lies on a quadtree dividing line), and the map of the real configuration against R0's run1 under
tol["double_actual"] (R0's own shift sensitivity, calibrated by calibrate.py). Float build: error against the double truth
must stay within tol["float"]["ratio"] of R0-float's own error (from tolerances.json), and the
total deposited area must agree with the double truth to within max(tol["float"]["sum_ratio"], ratio x R0-float's own
total-area error against the truth). The sum against R0-float is reported for information only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402

MAP_FILES = {"pixels": "ipm_magnifications.bin",
             "minima": "ipm_magnifications_minima.bin",
             "saddles": "ipm_magnifications_saddles.bin"}


def reference_map(ref_root, name, k, which="pixels"):
    """k is a run number (directory run<k>) or a directory name such as "shift1"."""
    p = Path(ref_root) / name / (f"run{k}" if isinstance(k, int) else k) / MAP_FILES[which]
    if not p.exists():
        raise FileNotFoundError(f"reference map missing: {p} (run make_reference.py first)")
    return bl.read_map(p)


def check_config(cfg, outdir, ref_root, tol, rung, outdir_shift=None):
    result = {"config": cfg.name, "build": cfg.build, "rung": rung, "maps": {}, "pass": True, "failures": [], "flags": []}
    which_list = ["pixels", "minima", "saddles"] if cfg.parities else ["pixels"]
    loaded = {}
    if cfg.build == "double" and outdir_shift is None:
        raise ValueError(f"{cfg.name}: a double configuration needs the output directory of its shifted-centre run")
    for which in which_list:
        cand = bl.read_map(Path(outdir) / MAP_FILES[which])
        loaded[which] = cand
        ref = reference_map(ref_root, cfg.name, 1, which)
        m_r0 = cm.map_metrics(cand, ref, tol["floor_frac"])
        entry = {"vs_r0": m_r0}
        if cfg.build == "double":
            verdict = cm.evaluate_double_actual(m_r0, tol, cfg.name)
            cand_shift = bl.read_map(Path(outdir_shift) / MAP_FILES[which])
            m_shift = cm.map_metrics(cand_shift, reference_map(ref_root, cfg.name, "shift1", which), tol["floor_frac"])
            verdict_shift = cm.evaluate_double(m_shift, tol)
            entry["vs_r0_shift"] = m_shift
            entry["verdict_shift"] = verdict_shift
        else:
            truth = reference_map(ref_root, cfg.truth, 1, which)
            m_truth = cm.map_metrics(cand, truth, tol["floor_frac"])
            try:
                ref_truth = tol["float_truth_error"][cfg.name][which]
            except KeyError:
                raise KeyError(f"tolerances.json has no float_truth_error for {cfg.name}/{which}; run calibrate.py") from None
            entry["vs_truth"] = m_truth
            verdict = cm.evaluate_float(m_truth, ref_truth, m_r0, tol)
        entry["verdict"] = verdict
        result["maps"][which] = entry
        if cfg.build == "double":
            result["pass"] = result["pass"] and verdict["pass"] and verdict_shift["pass"]
            for label, v in (("vs_r0_shift", verdict_shift), ("vs_r0", verdict)):
                result["failures"] += [f"{which} {label}: {f}" for f in v["failures"]]
                result["flags"] += [f"{which} {label}: {f}" for f in v["flags"]]
        else:
            result["pass"] = result["pass"] and verdict["pass"]
            result["failures"] += [f"{which}: {f}" for f in verdict["failures"]]
            result["flags"] += [f"{which}: {f}" for f in verdict["flags"]]
    if cfg.parities:
        pc = cm.parity_consistency(loaded["pixels"], loaded["minima"], loaded["saddles"])
        result["parity_consistency"] = pc
        if pc > tol["parity_consistency"]:
            result["pass"] = False
            result["failures"].append(f"pixels != minima + saddles: {pc:.3e} > {tol['parity_consistency']:.1e}")
    if cfg.starfile_from:   # information only: the same stars read back from a file
        result["vs_source_field"] = cm.map_metrics(loaded["pixels"], reference_map(ref_root, cfg.starfile_from, 1), tol["floor_frac"])
    return result


def summarize(res):
    m = res.get("maps", {}).get("pixels")
    if not m:
        return "; ".join(res["failures"])
    key = "vs_truth" if "vs_truth" in m else "vs_r0"
    s = m[key]
    txt = f"{key} mean {s['mean']:.2e} p99.9 {s['p999']:.2e} p99.99 {s['p9999']:.2e} max {s['max']:.2e}" if "vs_r0_shift" in m else \
        f"{key} mean {s['mean']:.2e} p99.9 {s['p999']:.2e} max {s['max']:.2e} sum {m['vs_r0']['sum_ratio']:+.1e}"
    if "vs_r0_shift" in m:
        t = m["vs_r0_shift"]
        txt += f" | shift mean {t['mean']:.2e} p99.9 {t['p999']:.2e} max {t['max']:.2e}"
    if res["failures"]:
        txt += " | " + "; ".join(res["failures"])
    if res["flags"]:
        txt += " | flags: " + "; ".join(res["flags"])
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--set", default="ladder", choices=["ladder", "full"])
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--tolerances", help="default: $IPM_BENCH_ROOT/results/tolerances.json")
    ap.add_argument("--results", default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep-outputs", action="store_true")
    a = ap.parse_args(argv)
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    a.tolerances = a.tolerances or str(bl.bench_path("results", "tolerances.json"))
    tol = cm.load_tolerances(a.tolerances)
    results_dir = Path(a.results or bl.bench_path("results", a.rung))
    scratch = Path(a.scratch or bl.SCRATCH / a.rung / "regress")
    cfgs = cf.regression_configs(a.rung, a.set)
    names = {c.name for c in cfgs}
    unknown = [n for n in (a.only or []) if n not in names]
    if a.only:
        cfgs = [c for c in cfgs if c.name in a.only]
    if unknown:
        print(f"unknown configuration(s): {', '.join(unknown)}", file=sys.stderr)
    if not cfgs:
        print("no configurations selected", file=sys.stderr)
    if unknown or not cfgs:
        return 2
    out = {"rung": a.rung, "set": a.set, "env": bl.environment_info(), "build": bl.build_meta(a.build),
           "tolerances": {k: tol[k] for k in ("floor_frac", "double", "double_actual", "float", "parity_consistency")},
           "configs": [], "pass": True, "complete": False, "n_configs": len(cfgs)}
    bl.write_json(results_dir / f"regression_{a.set}.json", out)   # an earlier result file must not outlive an abort
    for cfg in cf.ordered(cfgs):
        outdir = scratch / cfg.name
        exe = cf.exe_for(a.build, cfg.build)
        rec = bl.run_ipm(exe, cf.full_args(cfg, a.ref_root), outdir, timeout=3600)
        outdir_shift, rec_shift = None, None
        if cfg.build == "double":
            outdir_shift = scratch / f"{cfg.name}_shift"
            rec_shift = bl.run_ipm(exe, cf.shifted_args(cfg, a.ref_root), outdir_shift, timeout=3600)
        if not rec["ok"] or (rec_shift is not None and not rec_shift["ok"]):
            fails = [f"run failed: {rec['error']}"] if not rec["ok"] else []
            if rec_shift is not None and not rec_shift["ok"]:
                fails.append(f"shift run failed: {rec_shift['error']}")
            res = {"config": cfg.name, "build": cfg.build, "rung": a.rung, "pass": False,
                   "failures": fails, "flags": [], "maps": {}}
        else:
            try:
                res = check_config(cfg, outdir, a.ref_root, tol, a.rung, outdir_shift)
            except Exception as exc:   # recorded as a failure so the remaining configurations still run
                res = {"config": cfg.name, "build": cfg.build, "rung": a.rung, "pass": False,
                       "failures": [f"check failed: {type(exc).__name__}: {exc}"], "flags": [], "maps": {}}
        res["run"] = {k: rec[k] for k in ("wall_s", "t_shoot_cells", "returncode", "stderr_tail")}
        if rec_shift is not None:
            res["run_shift"] = {k: rec_shift[k] for k in ("wall_s", "t_shoot_cells", "returncode", "stderr_tail")}
        out["configs"].append(res)
        out["pass"] = out["pass"] and res["pass"]
        print(f"{'PASS' if res['pass'] else 'FAIL':4s} {cfg.name:12s} {summarize(res)}", flush=True)
        if not a.keep_outputs:
            bl.clear_outputs(outdir)
            if outdir_shift is not None:
                bl.clear_outputs(outdir_shift)
        bl.write_json(results_dir / f"regression_{a.set}.json", out)
    out["complete"] = True
    bl.write_json(results_dir / f"regression_{a.set}.json", out)
    print("REGRESSION", "PASS" if out["pass"] else "FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
