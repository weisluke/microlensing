#!/usr/bin/env python3
"""Time one rung on a configuration set: N repeats, median per configuration (spec section 7)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import configs as cf  # noqa: E402


def warm_up(build, ref_root, scratch):
    cfg = cf.CONFIGS["nb_f"]
    outdir = Path(scratch) / "warmup"
    rec = bl.run_ipm(cf.exe_for(build, cfg.build), cf.full_args(cfg, ref_root), outdir, timeout=3600)
    bl.clear_outputs(outdir)
    return rec


def summarize(records):
    per = {}
    for r in records:
        if r["ok"]:
            s = per.setdefault(r["config"], {"wall_s": [], "t_shoot_cells": []})
            s["wall_s"].append(r["wall_s"])
            s["t_shoot_cells"].append(r["t_shoot_cells"])
    return {name: {"median_wall_s": bl.median(v["wall_s"]), "median_t_shoot_cells": bl.median(v["t_shoot_cells"]),
                   "n": len(v["wall_s"])} for name, v in per.items()}


def time_configs(build, rung, cfgs, repeats, ref_root, scratch, sample_gpu=False, sample_host=False):
    records = []
    for k in range(1, repeats + 1):
        for cfg in cf.ordered(cfgs):
            outdir = Path(scratch) / cfg.name
            rec = bl.run_ipm(cf.exe_for(build, cfg.build), cf.full_args(cfg, ref_root), outdir, sample_gpu=sample_gpu,
                             sample_host=sample_host, timeout=3600)
            rec.update({"config": cfg.name, "repeat": k, "rung": rung})
            records.append(rec)
            shoot = f"{rec['t_shoot_cells']:.3f}" if rec["ok"] else "FAIL " + str(rec["error"])
            mem = f" peak {rec['peak_gpu_mem_mib']} MiB" if rec["peak_gpu_mem_mib"] is not None else ""
            print(f"{rung:7s} {cfg.name:12s} run{k} wall {rec['wall_s']:.2f} s t_shoot {shoot}{mem}", flush=True)
            bl.clear_outputs(outdir)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--set", default="ladder", choices=["ladder", "full", "memory", "fallback"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--results", default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    problem = bl.check_gpu_allocation()
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    selectable = cf.ladder_configs(a.rung) if a.set == "ladder" else cf.configs_in(a.set)
    names = {c.name for c in selectable}
    unknown = [n for n in (a.only or []) if n not in names]
    cfgs = selectable if a.only is None else [c for c in selectable if c.name in a.only]
    if unknown or not cfgs:
        print("unknown or empty configuration selection", file=sys.stderr)
        return 2
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    results_dir = Path(a.results or bl.bench_path("results", a.rung))
    out_path = results_dir / f"timing_{a.set}.json"
    if out_path.exists() and not a.force:
        print(f"{out_path} exists; use --force to re-time")
        return 0
    scratch = Path(a.scratch or bl.SCRATCH / a.rung / "time")
    if not a.no_warmup:
        warm_up(a.build, a.ref_root, scratch)
    records = time_configs(a.build, a.rung, cfgs, a.repeats, a.ref_root, scratch, sample_gpu=(a.set == "memory"),
                            sample_host=(a.set == "memory"))
    out = {"rung": a.rung, "set": a.set, "repeats": a.repeats, "env": bl.environment_info(),
           "build": bl.build_meta(a.build), "records": records, "summary": summarize(records)}
    bl.write_json(out_path, out)
    for name, s in out["summary"].items():
        print(f"{a.rung:7s} {name:12s} median wall {s['median_wall_s']:.2f} s, median t_shoot_cells "
              f"{s['median_t_shoot_cells']:.3f} s (n={s['n']})")
    return 0 if all(r["ok"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
