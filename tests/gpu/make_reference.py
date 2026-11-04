#!/usr/bin/env python3
"""Run the R0 build over the matrix and store the reference maps (spec sections 5 and 8).

Three runs per full-set configuration (run1 is the reference, run2/run3 give the noise floor),
one run per truth-only configuration. Runs execute in the scratch directory and are moved to
ref/<config>/run<k>/ afterwards, so their wall times double as R0's full-matrix timing.

With --shift, the double regression configurations are instead run once with the source centre shifted (configs.SHIFT)
into ref/<config>/shift1/, for the shifted-centre equivalence check; those runs are recorded in shift_runs.json.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import configs as cf  # noqa: E402


def runs_for(cfg):
    return 3 if "full" in cfg.sets else 1


def reference_dir(ref_root, cfg, k):
    return Path(ref_root) / cfg.name / f"run{k}"


def is_complete(d):
    return (d / "ipm_magnifications.bin").exists() and (d / "ipm_parameter_info.txt").exists()


def publish(scratch, dest):
    """Stage a copy beside the destination, then rename it into place."""
    staging = dest.with_name(dest.name + ".partial")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        shutil.move(str(scratch), str(staging))
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(staging, dest)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return str(exc)
    return None


def shift_main(a):
    names = {c.name for c in cf.double_regression_configs()}
    wanted = [c for c in cf.double_regression_configs() if not a.only or c.name in a.only]
    unknown = [n for n in (a.only or []) if n not in names]
    if unknown or not wanted:
        print(f"--shift covers only {', '.join(sorted(names))}; unknown: {', '.join(unknown) or 'none selected'}", file=sys.stderr)
        return 2
    path = Path(a.results) / "shift_runs.json"
    out = bl.read_json(path) if path.exists() else {"rung": a.rung, "shift": list(cf.SHIFT), "env": bl.environment_info(),
                                                    "build": bl.build_meta(a.build), "records": []}
    failures = []
    for cfg in wanted:
        dest = Path(a.ref_root) / cfg.name / "shift1"
        if is_complete(dest) and not a.force:
            print(f"skip {cfg.name} shift1 (exists)", flush=True)
            continue
        scratch = Path(a.scratch) / cfg.name / "shift1"
        print(f"run  {cfg.name} shift1", flush=True)
        rec = bl.run_ipm(cf.exe_for(a.build, cfg.build), cf.shifted_args(cfg, a.ref_root), scratch, timeout=3600)
        rec.update({"config": cfg.name, "repeat": "shift1", "rung": a.rung, "outdir": str(dest)})
        if not rec["ok"]:
            failures.append(f"{cfg.name} shift1: {rec['error']}\n{rec['stderr_tail']}")
            print(failures[-1], flush=True)
            continue
        out["records"] = [r for r in out["records"] if r["config"] != cfg.name] + [rec]
        bl.write_json(path, out)
        err = publish(scratch, dest)
        if err:
            failures.append(f"{cfg.name} shift1: publish failed: {err}")
            print(failures[-1], flush=True)
            continue
        print(f"     wall {rec['wall_s']:.2f} s", flush=True)
    if failures:
        print("\nFAILED:\n" + "\n".join(failures))
        return 1
    print("all shifted references present")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="default: $IPM_BENCH_ROOT/builds/R0")
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--scratch", default=str(bl.SCRATCH / "ref"))
    ap.add_argument("--results", required=True,
                    help="directory for timing_full.json or shift_runs.json (round 1's references: results/R0)")
    ap.add_argument("--only", nargs="*", help="configuration names (default: every full and truth configuration)")
    ap.add_argument("--shift", action="store_true",
                    help="instead of the matrix, run the double regression configurations with the shifted centre into ref/<config>/shift1")
    ap.add_argument("--rung", default="R0", help="label recorded for the build that makes the references")
    ap.add_argument("--force", action="store_true", help="re-run even if the reference exists")
    a = ap.parse_args(argv)
    a.build = a.build or str(bl.bench_path("builds", "R0"))
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    if a.shift:
        return shift_main(a)
    wanted = [cf.CONFIGS[n] for n in a.only] if a.only else cf.configs_in("full") + cf.configs_in("truth")
    timing_path = Path(a.results) / "timing_full.json"
    if timing_path.exists():
        timing = bl.read_json(timing_path)
    else:
        timing = {"rung": a.rung, "set": "full", "env": bl.environment_info(), "build": bl.build_meta(a.build), "records": []}
    done = {(r["config"], r["repeat"]): i for i, r in enumerate(timing["records"])}
    failures = []
    for cfg in cf.ordered(wanted):
        exe = cf.exe_for(a.build, cfg.build)
        for k in range(1, runs_for(cfg) + 1):
            dest = reference_dir(a.ref_root, cfg, k)
            if is_complete(dest) and not a.force:
                print(f"skip {cfg.name} run{k} (exists)", flush=True)
                continue
            scratch = Path(a.scratch) / cfg.name / f"run{k}"
            print(f"run  {cfg.name} run{k}", flush=True)
            rec = bl.run_ipm(exe, cf.full_args(cfg, a.ref_root), scratch, timeout=3600)
            rec.update({"config": cfg.name, "repeat": k, "rung": a.rung, "outdir": str(dest)})
            if not rec["ok"]:
                failures.append(f"{cfg.name} run{k}: {rec['error']}\n{rec['stderr_tail']}")
                print(failures[-1], flush=True)
                continue
            if "full" in cfg.sets:
                key = (cfg.name, k)
                if key in done:
                    timing["records"][done[key]] = rec
                else:
                    done[key] = len(timing["records"])
                    timing["records"].append(rec)
                bl.write_json(timing_path, timing)
            err = publish(scratch, dest)
            if err:
                failures.append(f"{cfg.name} run{k}: publish failed: {err}")
                print(failures[-1], flush=True)
                continue
            print(f"     wall {rec['wall_s']:.2f} s, t_shoot_cells {rec['t_shoot_cells']:.3f} s", flush=True)
    if failures:
        print("\nFAILED:\n" + "\n".join(failures))
        return 1
    print("all references present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
