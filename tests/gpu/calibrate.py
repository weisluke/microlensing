#!/usr/bin/env python3
"""Noise floors and float truth errors from the R0 references; writes tolerances.json (spec section 8).

An existing tolerances.json keeps its thresholds (double, float, floor_frac, parity_consistency), so hand edits survive
a re-run; floors, float_truth_error, parity_floor, double_shift_sensitivity, double_actual and warnings are always
measured afresh. double_shift_sensitivity is R0's run1 against its shift1 run for each double regression configuration;
double_actual is derived from it, per configuration and as "default" (from the largest sensitivities): 2x mean/p999/p9999,
floored at the signed-off double value, and 3x max for flag_max, floored at 1e-3, each rounded up to one significant
figure; offmask_max and sum_ratio are copied from double. Without shift1 references it keeps its defaults.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402
from regress import MAP_FILES, reference_map  # noqa: E402


def round_up_1sf(x):
    """Round a non-negative number up to one significant figure: 1.76e-6 -> 2e-6."""
    if x <= 0:
        return 0.0
    exp = math.floor(math.log10(x))
    return float(f"{math.ceil(round(x / 10 ** exp, 9))}e{exp}")


def derive_double_actual(sens, double):
    """Limits for one configuration from its shift sensitivity (mean, p999, p9999, max)."""
    actual = {key: round_up_1sf(max(2 * sens[key], double[key])) for key in ("mean", "p999", "p9999")}
    actual.update(offmask_max=double["offmask_max"], sum_ratio=double["sum_ratio"],
                  flag_max=round_up_1sf(max(3 * sens["max"], 1e-3)))
    return actual


def sanity_warnings(tol):
    """Double thresholds and the parity threshold must be at least 10x the measured noise floor."""
    warnings = []
    for name, per_map in tol["floors"].items():
        if cf.CONFIGS[name].build != "double":
            continue
        for which, floors in per_map.items():
            for key in ("mean", "p999", "p9999", "offmask_max"):
                worst = max(f[key] for f in floors)
                if worst * 10 > tol["double"][key]:
                    warnings.append(f"{name} {which}: double floor {key}={worst:.2e} is within 10x of the threshold {tol['double'][key]:.1e}")
    sens = tol.get("double_shift_sensitivity", {})
    for key in ("mean", "p999", "p9999"):
        worst = max((per[key] for per in sens.values()), default=0.0)
        if tol["double"][key] < worst:
            warnings.append(f"double threshold {key}={tol['double'][key]:.1e} is below R0's own shift sensitivity {worst:.2e}: "
                            "it can only be met by the shifted equivalence check")
    for name, values in tol.get("parity_floor", {}).items():
        worst = max(values)
        if worst * 10 > tol["parity_consistency"]:
            warnings.append(f"{name} parity: consistency floor {worst:.2e} is within 10x of the threshold {tol['parity_consistency']:.1e}")
    return warnings


def parity_floors(ref_root, cfg):
    """pixels vs minima + saddles for run1, run2 and run3 of a parity configuration, then for run1 of its double twin."""
    runs = [(cfg.name, k) for k in (1, 2, 3)] + ([(cfg.truth, 1)] if cfg.truth else [])
    return [cm.parity_consistency(*(reference_map(ref_root, name, k, which) for which in ("pixels", "minima", "saddles")))
            for name, k in runs]


def calibrate(ref_root, tol):
    tol["floors"] = {}
    tol["float_truth_error"] = {}
    tol["parity_floor"] = {}
    lines = []
    for cfg in cf.configs_in("full"):
        which_list = ["pixels", "minima", "saddles"] if cfg.parities else ["pixels"]
        for which in which_list:
            ref1 = reference_map(ref_root, cfg.name, 1, which)
            floors = [cm.map_metrics(reference_map(ref_root, cfg.name, k, which), ref1, tol["floor_frac"]) for k in (2, 3)]
            tol["floors"].setdefault(cfg.name, {})[which] = floors
            line = (f"{cfg.name:12s} {which:7s} floor mean {max(f['mean'] for f in floors):.2e} "
                    f"p99.9 {max(f['p999'] for f in floors):.2e} max {max(f['max'] for f in floors):.2e}")
            if cfg.build == "float":
                err = cm.map_metrics(ref1, reference_map(ref_root, cfg.truth, 1, which), tol["floor_frac"])
                tol["float_truth_error"].setdefault(cfg.name, {})[which] = err
                line += (f" | vs truth mean {err['mean']:.2e} p99.9 {err['p999']:.2e} "
                         f"p99.99 {err['p9999']:.2e} max {err['max']:.2e}")
            lines.append(line)
            print(line, flush=True)
        if cfg.parities:
            values = parity_floors(ref_root, cfg)
            tol["parity_floor"][cfg.name] = values
            line = f"{cfg.name:12s} {'parity':7s} consistency " + " ".join(f"run{k} {v:.2e}" for k, v in enumerate(values[:3], start=1))
            if cfg.truth:
                line += f" | {cfg.truth} run1 {values[3]:.2e}"
            lines.append(line)
            print(line, flush=True)
    tol["double_shift_sensitivity"] = {}
    for cfg in cf.double_regression_configs():
        shift = Path(ref_root) / cfg.name / "shift1" / MAP_FILES["pixels"]
        if not shift.exists():
            continue
        sens = cm.map_metrics(reference_map(ref_root, cfg.name, "shift1"), reference_map(ref_root, cfg.name, 1), tol["floor_frac"])
        tol["double_shift_sensitivity"][cfg.name] = sens
        line = (f"{cfg.name:12s} pixels  shift sensitivity mean {sens['mean']:.2e} p99.9 {sens['p999']:.2e} "
                f"p99.99 {sens['p9999']:.2e} max {sens['max']:.2e}")
        lines.append(line)
        print(line, flush=True)
    sens = tol["double_shift_sensitivity"]
    extra = []
    if sens:
        extra += [f"{cfg.name}: no shift1 reference (the others have one); run make_reference.py --shift"
                  for cfg in cf.double_regression_configs() if cfg.name not in sens]
    if sens:
        worst = {key: max(s[key] for s in sens.values()) for key in ("mean", "p999", "p9999", "max")}
        tol["double_actual"] = {name: derive_double_actual(s, tol["double"]) for name, s in sens.items()}
        tol["double_actual"]["default"] = derive_double_actual(worst, tol["double"])
        tol["double_actual_note"] = "calibrated from R0 shift sensitivity; pending user sign-off"
        for name, actual in tol["double_actual"].items():
            line = f"double_actual {name:9s} " + " ".join(f"{k} {v:.2e}" for k, v in actual.items())
            lines.append(line)
            print(line, flush=True)
    else:
        extra.append("no shift1 references found: double_actual keeps its defaults (run make_reference.py --shift, then calibrate.py again)")
    warnings = sanity_warnings(tol) + extra
    tol["warnings"] = warnings
    for w in warnings:
        print("WARNING", w)
    return tol, lines, warnings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--out", help="default: $IPM_BENCH_ROOT/results/tolerances.json")
    a = ap.parse_args(argv)
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    a.out = a.out or str(bl.bench_path("results", "tolerances.json"))
    tol, lines, warnings = calibrate(a.ref_root, cm.load_tolerances(a.out))
    bl.write_json(a.out, tol)
    summary = Path(a.out).with_name("tolerances_summary.md")
    summary.write_text("```\n" + "\n".join(lines + warnings) + "\n```\n")
    print("wrote", a.out, "and", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
