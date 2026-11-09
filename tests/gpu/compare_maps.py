#!/usr/bin/env python3
"""Map comparison metrics and pass/fail rules (spec section 8).

Metrics of candidate A against reference B, in float64:
  mask M = pixels with B >= floor_frac * mean(B); rel = |A - B| / B on M
  mean, p999 (99.9th percentile), p9999, max of rel; offmask_max = max |A - B| off M / mean(B)
  sum_ratio = sum(A) / sum(B) - 1; finite; nonneg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def default_tolerances():
    return {
        "floor_frac": 1e-3,
        "parity_consistency": 1e-5,
        "double": {"mean": 1e-7, "p999": 1e-5, "p9999": 3e-5, "offmask_max": 1e-5, "sum_ratio": 1e-6},
        "double_actual": {"default": {"mean": 1e-7, "p999": 1e-5, "p9999": 3e-5, "offmask_max": 1e-5, "sum_ratio": 1e-6,
                                      "flag_max": 1e-3}},
        "double_shift_sensitivity": {},
        "float": {"ratio": 1.5, "flag_ratio_max": 3.0, "sum_ratio": 1e-5},
        "floors": {},
        "float_truth_error": {},
    }


def load_tolerances(path):
    tol = default_tolerances()
    path = Path(path)
    if path.exists():
        for key, value in json.loads(path.read_text()).items():
            if isinstance(value, dict) and isinstance(tol.get(key), dict):
                tol[key].update(value)
            else:
                tol[key] = value
    return tol


def map_metrics(a, b, floor_frac=1e-3):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: candidate {a.shape} vs reference {b.shape}")
    finite = bool(np.isfinite(a).all())
    nonneg = bool(finite and (a >= 0).all())
    mean_b = float(b.mean())
    if not np.isfinite(mean_b) or mean_b <= 0:
        raise ValueError("reference map mean is not positive")
    mask = b >= floor_frac * mean_b
    n_mask = int(mask.sum())
    if n_mask == 0:
        raise ValueError("no reference pixel above the floor")
    rel = np.abs(a[mask] - b[mask]) / b[mask]
    off = np.abs(a[~mask] - b[~mask]) if n_mask < b.size else np.zeros(1)
    return {
        "mean": float(np.nanmean(rel)),
        "p999": float(np.nanpercentile(rel, 99.9)),
        "p9999": float(np.nanpercentile(rel, 99.99)),
        "max": float(np.nanmax(rel)),
        "offmask_max": float(np.nanmax(off) / mean_b),
        "sum_ratio": float(a.sum() / b.sum() - 1),
        "n_mask": n_mask,
        "n_total": int(b.size),
        "finite": finite,
        "nonneg": nonneg,
    }


def parity_consistency(total, minima, saddles):
    total = np.asarray(total, dtype=np.float64)
    diff = np.abs(total - (np.asarray(minima, dtype=np.float64) + np.asarray(saddles, dtype=np.float64)))
    return float(diff.max() / total.mean())


def _common_failures(m):
    fails = []
    if not m["finite"]:
        fails.append("non-finite pixels")
    elif not m["nonneg"]:
        fails.append("negative pixels")
    return fails


def evaluate_double(m, tol):
    d = tol["double"]
    fails = _common_failures(m)
    for key in ("mean", "p999", "p9999", "offmask_max"):
        if m[key] > d[key]:
            fails.append(f"{key}={m[key]:.3e} > {d[key]:.1e}")
    if abs(m["sum_ratio"]) > d["sum_ratio"]:
        fails.append(f"|sum_ratio|={abs(m['sum_ratio']):.3e} > {d['sum_ratio']:.1e}")
    return {"pass": not fails, "failures": fails, "flags": []}


def evaluate_double_actual(m, tol, config_name=None):
    """Real-configuration rule for double builds: R0's own shift sensitivity of the configuration bounds the limits
    (its entry in tol["double_actual"], else "default"); a large max is only flagged. Messages name the entry used."""
    entries = tol["double_actual"]
    entry = config_name if entries.get(config_name) else "default"
    d = entries[entry]
    fails = _common_failures(m)
    for key in ("mean", "p999", "p9999", "offmask_max"):
        if m[key] > d[key]:
            fails.append(f"{key}={m[key]:.3e} > {d[key]:.1e} [{entry}]")
    if abs(m["sum_ratio"]) > d["sum_ratio"]:
        fails.append(f"|sum_ratio|={abs(m['sum_ratio']):.3e} > {d['sum_ratio']:.1e} [{entry}]")
    flags = [f"max={m['max']:.3e} > {d['flag_max']:.1e} [{entry}]"] if m["max"] > d["flag_max"] else []
    return {"pass": not fails, "failures": fails, "flags": flags}


def evaluate_float(m_truth, ref_truth, m_r0, tol):
    f = tol["float"]
    fails = _common_failures(m_r0)
    flags = []
    for key in ("mean", "p999", "p9999"):
        limit = f["ratio"] * max(ref_truth[key], 1e-15)
        if m_truth[key] > limit:
            fails.append(f"{key} vs truth {m_truth[key]:.3e} > {f['ratio']} x R0's {ref_truth[key]:.3e}")
    if m_truth["max"] > f["flag_ratio_max"] * max(ref_truth["max"], 1e-15):
        flags.append(f"max vs truth {m_truth['max']:.3e} > {f['flag_ratio_max']} x R0's {ref_truth['max']:.3e}")
    # the total deposited area is judged against the double truth: R0-float's own deficit (~1e-4) sets the allowance
    ref_sum = abs(ref_truth.get("sum_ratio", 0.0))
    limit = max(f["sum_ratio"], f["ratio"] * ref_sum)
    if abs(m_truth["sum_ratio"]) > limit:
        fails.append(f"|sum_ratio| vs truth {abs(m_truth['sum_ratio']):.3e} > {limit:.1e} "
                     f"(max of {f['sum_ratio']:.1e} and {f['ratio']} x R0's {ref_sum:.3e})")
    return {"pass": not fails, "failures": fails, "flags": flags}


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_lib
    ap = argparse.ArgumentParser(description="metrics of map A against reference map B")
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--floor-frac", type=float, default=1e-3)
    a = ap.parse_args(argv)
    m = map_metrics(bench_lib.read_map(a.candidate), bench_lib.read_map(a.reference), a.floor_frac)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
