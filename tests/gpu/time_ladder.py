#!/usr/bin/env python3
"""Interleaved timing of all rung builds (spec section 7): for each repeat, for each rung, for each
ladder configuration, so that clock drift affects every rung alike. Resumable."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import configs as cf  # noqa: E402
from time_rung import summarize, warm_up  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=bl.RUNGS)
    ap.add_argument("--builds-root", help="default: $IPM_BENCH_ROOT/builds")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ref-root", help="default: $IPM_BENCH_ROOT/ref")
    ap.add_argument("--out", help="default: $IPM_BENCH_ROOT/results/ladder_interleaved.json")
    ap.add_argument("--scratch", default=str(bl.SCRATCH / "ladder"))
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    a.builds_root = a.builds_root or str(bl.bench_path("builds"))
    a.ref_root = a.ref_root or str(bl.bench_path("ref"))
    a.out = a.out or str(bl.bench_path("results", "ladder_interleaved.json"))
    builds = {r: Path(a.builds_root) / r for r in a.rungs}
    missing = [str(b) for b in builds.values() if not (b / "ipm_gpu").exists()]
    if missing:
        print("missing builds:", missing)
        return 2
    out_path = Path(a.out)
    if out_path.exists():
        out = bl.read_json(out_path)
    else:
        out = {"rungs": a.rungs, "repeats": a.repeats, "env": bl.environment_info(),
               "builds": {r: bl.build_meta(b) for r, b in builds.items()}, "records": []}
    # position of each (rung, config, repeat) record; only successful records count as done, and a
    # configuration measured again replaces its failed record in place
    index = {(r["rung"], r["config"], r["repeat"]): i for i, r in enumerate(out["records"])}
    done = {key for key, i in index.items() if out["records"][i]["ok"]}
    if not a.no_warmup:
        warm_up(builds[a.rungs[0]], a.ref_root, a.scratch)
    for k in range(1, a.repeats + 1):
        for rung in a.rungs:
            for cfg in cf.ladder_configs(rung):
                key = (rung, cfg.name, k)
                if key in done:
                    continue
                outdir = Path(a.scratch) / rung / cfg.name
                rec = bl.run_ipm(cf.exe_for(builds[rung], cfg.build), cf.full_args(cfg, a.ref_root), outdir,
                                 timeout=3600)
                rec.update({"config": cfg.name, "repeat": k, "rung": rung})
                if key in index:
                    out["records"][index[key]] = rec
                else:
                    index[key] = len(out["records"])
                    out["records"].append(rec)
                bl.write_json(out_path, out)
                shoot = f"{rec['t_shoot_cells']:.3f}" if rec["ok"] else "FAIL " + str(rec["error"])
                print(f"rep{k} {rung:7s} {cfg.name:12s} wall {rec['wall_s']:.2f} s t_shoot {shoot}", flush=True)
                bl.clear_outputs(outdir)
    out["summary"] = {rung: summarize([r for r in out["records"] if r["rung"] == rung]) for rung in a.rungs}
    bl.write_json(out_path, out)
    return 0 if all(r["ok"] for r in out["records"]) else 1


if __name__ == "__main__":
    sys.exit(main())
