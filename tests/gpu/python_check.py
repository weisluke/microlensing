#!/usr/bin/env python3
"""Python-wrapper check and timing for one rung (spec sections 5 and 7).

Run with PYTHONPATH=<build>/pyroot so that microlensing.IPM loads the rung's libraries:
    PYTHONPATH=$IPM_BENCH_ROOT/builds/R5/pyroot python tests/gpu/python_check.py --build $IPM_BENCH_ROOT/builds/R5 --rung R5
A double configuration is judged by both parts of the double rule (see regress.py): a second run with the shifted centre
against R0's shift1 reference, and the real run against run1. run() wall time includes the copy of each map into numpy that
the wrapper performs. Each configuration gets one untimed warm-up run first (spec section 7); it is not in "runs", and the
verdict is made on the first timed run.
"""
import argparse
import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402
from regress import reference_map  # noqa: E402

KW = {
    "nb_f": dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25.0, half_length_y2=25.0,
                 num_pixels_y1=5000, num_pixels_y2=5000),
    "nb_d": dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25.0, half_length_y2=25.0,
                 num_pixels_y1=5000, num_pixels_y2=5000, is_double=True),
    "par_f": dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25.0, half_length_y2=25.0,
                  num_pixels_y1=5000, num_pixels_y2=5000, write_parities=True),
    "px40k_f": dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25.0, half_length_y2=25.0,
                    num_pixels_y1=40000, num_pixels_y2=40000),
    "hm100_f": dict(kappa_tot=0.5, shear=0.49, kappa_star=0.1, rectangular=True, approx=True,
                    half_length_y1=12.5, half_length_y2=12.5, num_pixels_y1=1000, num_pixels_y2=1000),
}


def select(names):
    """The configurations to run: all of KW when names is empty, else the named ones (unknown name: SystemExit)."""
    if not names:
        return list(KW)
    unknown = [n for n in names if n not in KW]
    if unknown:
        raise SystemExit(f"unknown configuration(s) {unknown}; choose from {list(KW)}")
    return list(names)


def judge_double(cand, cand_shift, ref_root, name, tol):
    """Both parts of the double rule; returns (metrics vs run1, metrics vs shift1, combined verdict)."""
    m_r0 = cm.map_metrics(cand, reference_map(ref_root, name, 1), tol["floor_frac"])
    m_shift = cm.map_metrics(cand_shift, reference_map(ref_root, name, "shift1"), tol["floor_frac"])
    v_shift = cm.evaluate_double(m_shift, tol)
    v_r0 = cm.evaluate_double_actual(m_r0, tol, name)
    verdict = {"pass": v_shift["pass"] and v_r0["pass"], "failures": [], "flags": []}
    for label, v in (("vs_r0_shift", v_shift), ("vs_r0", v_r0)):
        verdict["failures"] += [f"{label}: {f}" for f in v["failures"]]
        verdict["flags"] += [f"{label}: {f}" for f in v["flags"]]
    return m_r0, m_shift, verdict


def judge_parities(cands, ref_root, cfg, tol):
    """Float rule for each of the three maps, plus pixels == minima + saddles; returns (verdict, maps, consistency)."""
    maps = {}
    verdict = {"pass": True, "failures": [], "flags": []}
    for which, cand in cands.items():
        m_r0 = cm.map_metrics(cand, reference_map(ref_root, cfg.name, 1, which), tol["floor_frac"])
        m_truth = cm.map_metrics(cand, reference_map(ref_root, cfg.truth, 1, which), tol["floor_frac"])
        v = cm.evaluate_float(m_truth, tol["float_truth_error"][cfg.name][which], m_r0, tol)
        maps[which] = {"vs_r0": m_r0, "vs_truth": m_truth, "verdict": v}
        verdict["pass"] = verdict["pass"] and v["pass"]
        verdict["failures"] += [f"{which}: {f}" for f in v["failures"]]
        verdict["flags"] += [f"{which}: {f}" for f in v["flags"]]
    pc = cm.parity_consistency(cands["pixels"], cands["minima"], cands["saddles"])
    if pc > tol["parity_consistency"]:
        verdict["pass"] = False
        verdict["failures"].append(f"pixels != minima + saddles: {pc:.3e} > {tol['parity_consistency']:.1e}")
    return verdict, maps, pc


def save_check(ipm, save_dir):
    """Call save() and compare the written map with the in-memory one; returns "pass" or the failure text. Removes save_dir."""
    try:
        ipm.save()
        written = bl.read_map(Path(save_dir) / "ipm_magnifications.bin")
        if np.array_equal(written, ipm.magnifications.astype(np.float64)):
            return "pass"
        return "ipm_magnifications.bin differs from ipm.magnifications"
    except Exception as e:  # noqa: BLE001 - reported in the result
        return f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)


def load_wrapper(build):
    """Import the Python wrapper; returns (IPM class, [float library, double library]), or (None, message) when the
    libraries it loaded are not the build's."""
    from microlensing.IPM.ipm import IPM
    import microlensing.IPM.lib_ipm as lib_f
    import microlensing.IPM.lib_ipm_double as lib_d
    pyroot = str(Path(build).resolve() / "pyroot")
    for lib in (lib_f.lib, lib_d.lib):
        if not str(lib._name).startswith(pyroot):
            return None, f"wrong library loaded: {lib._name}; run with PYTHONPATH={pyroot}"
    return IPM, [str(lib_f.lib._name), str(lib_d.lib._name)]


def warm_up(IPM, kw):
    """One untimed run with the configuration's keywords, so that the first timed repeat is not a cold one."""
    ipm = IPM(verbose=0, random_seed=int(cf.SEED), **kw)
    ipm.run()
    del ipm
    gc.collect()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME", help="run only these configurations")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--tolerances", help="default: $IPM_BENCH_ROOT/results/tolerances.json")
    ap.add_argument("--results", help="default: $IPM_BENCH_ROOT/results/<rung>")
    a = ap.parse_args(argv)
    names = select(a.only)
    problem = bl.check_gpu_allocation()
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    a.tolerances = a.tolerances or str(bl.bench_path("results", "tolerances.json"))
    a.results = a.results or str(bl.bench_path("results", a.rung))
    IPM, libraries = load_wrapper(a.build)
    if IPM is None:
        print(libraries)
        return 2
    tol = cm.load_tolerances(a.tolerances)
    out = {"rung": a.rung, "env": bl.environment_info(), "build": bl.build_meta(a.build),
           "libraries": libraries, "warmup": True, "configs": {}, "pass": True}
    for name in names:
        kw = KW[name]
        cfg = cf.CONFIGS[name]
        runs = []
        entry = {}
        warm_up(IPM, kw)
        for k in range(a.repeats):
            save_dir = tempfile.mkdtemp(prefix="python_check_") if (k == 0 and name == "nb_f") else None
            extra = dict(write_maps=True, outfile_prefix=save_dir + "/") if save_dir else {}
            ipm = IPM(verbose=0, random_seed=int(cf.SEED), **kw, **extra)
            t0 = time.perf_counter()
            ipm.run()
            wall = time.perf_counter() - t0
            runs.append({"run_wall_s": wall, "t_shoot_cells": ipm.t_shoot_cells})
            if k == 0:
                cand = np.asarray(ipm.magnifications, dtype=np.float64)
                entry = {"shape": list(cand.shape)}
                if save_dir:
                    entry["save_check"] = save_check(ipm, save_dir)
                if cfg.build == "double":
                    cy1, cy2 = cf.shift_center()
                    ipm_shift = IPM(verbose=0, random_seed=int(cf.SEED), center_y1=cy1, center_y2=cy2, **kw)
                    t0 = time.perf_counter()
                    ipm_shift.run()
                    entry["shift_run"] = {"run_wall_s": time.perf_counter() - t0, "t_shoot_cells": ipm_shift.t_shoot_cells}
                    cand_shift = np.asarray(ipm_shift.magnifications, dtype=np.float64)
                    del ipm_shift
                    gc.collect()
                    m_r0, m_shift, verdict = judge_double(cand, cand_shift, a.ref_root, name, tol)
                    entry["vs_r0_shift"] = m_shift
                    entry["vs_r0"] = m_r0
                elif cfg.parities:
                    cands = {"pixels": cand, "minima": np.asarray(ipm.magnifications_minima, dtype=np.float64),
                             "saddles": np.asarray(ipm.magnifications_saddles, dtype=np.float64)}
                    verdict, entry["maps"], entry["parity_consistency"] = judge_parities(cands, a.ref_root, cfg, tol)
                    entry["vs_r0"] = entry["maps"]["pixels"]["vs_r0"]
                    entry["vs_truth"] = entry["maps"]["pixels"]["vs_truth"]
                else:
                    m_r0 = cm.map_metrics(cand, reference_map(a.ref_root, name, 1), tol["floor_frac"])
                    entry["vs_r0"] = m_r0
                    m_truth = cm.map_metrics(cand, reference_map(a.ref_root, cfg.truth, 1), tol["floor_frac"])
                    entry["vs_truth"] = m_truth
                    verdict = cm.evaluate_float(m_truth, tol["float_truth_error"][name]["pixels"], m_r0, tol)
                if save_dir:
                    if entry["save_check"] != "pass":
                        verdict = {**verdict, "pass": False, "failures": verdict["failures"] + [f"save(): {entry['save_check']}"]}
                entry["verdict"] = verdict
            del ipm
            gc.collect()
        entry["runs"] = runs
        entry["median_run_wall_s"] = bl.median([r["run_wall_s"] for r in runs])
        entry["median_t_shoot_cells"] = bl.median([r["t_shoot_cells"] for r in runs])
        out["configs"][name] = entry
        out["pass"] = out["pass"] and entry["verdict"]["pass"]
        status = "PASS" if entry["verdict"]["pass"] else "FAIL " + "; ".join(entry["verdict"]["failures"])
        print(f"{name:8s} run() median {entry['median_run_wall_s']:.2f} s, t_shoot_cells "
              f"{entry['median_t_shoot_cells']:.3f} s, {status}", flush=True)
    bl.write_json(Path(a.results) / "python.json", out)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
