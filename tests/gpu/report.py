#!/usr/bin/env python3
"""Render the result tables (spec section 10) from the JSON results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import configs as cf  # noqa: E402
from time_rung import summarize  # noqa: E402


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def speedup(base, value):
    return None if base is None or value is None or value == 0 else base / value


def ladder_table(summary, rungs, names, key):
    head = "| rung | " + " | ".join(f"{n} [s] | vs R0 | vs prev" for n in names) + " |"
    sep = "|---|" + "---|---|---|" * len(names)
    rows = [head, sep]
    prev = {n: None for n in names}
    for rung in rungs:
        cells = []
        for n in names:
            v = summary.get(rung, {}).get(n, {}).get(key)
            base = summary.get("R0", {}).get(n, {}).get(key)
            cells.append(f"{fmt(v)} | {fmt(speedup(base, v), 2)} | {fmt(speedup(prev[n], v), 2)}")
            if v is not None:
                prev[n] = v
        rows.append(f"| {bl.rung_label(rung)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def matrix_table(r0, r5, key):
    names = sorted(set(r0) | set(r5), key=lambda n: list(cf.CONFIGS).index(n) if n in cf.CONFIGS else 99)
    rows = ["| configuration | R0 [s] | R5 [s] | speedup |", "|---|---|---|---|"]
    for n in names:
        a, b = r0.get(n, {}).get(key), r5.get(n, {}).get(key)
        rows.append(f"| {n} | {fmt(a)} | {fmt(b)} | {fmt(speedup(a, b), 2)} |")
    return "\n".join(rows)


def python_table(root):
    rows = ["| configuration | rung | run() median [s] | t_shoot_cells median [s] | check |", "|---|---|---|---|---|"]
    for rung in ("R0", "R5"):
        p = Path(root) / rung / "python.json"
        if not p.exists():
            continue
        for name, e in bl.read_json(p)["configs"].items():
            rows.append(f"| {name} | {rung} | {fmt(e['median_run_wall_s'])} | {fmt(e['median_t_shoot_cells'])} | "
                        f"{'pass' if e['verdict']['pass'] else 'FAIL'} |")
    return "\n".join(rows)


def memory_table(root):
    rows = ["| configuration | R0 peak [MiB] | R5 peak [MiB] |", "|---|---|---|"]
    peaks = {}
    for rung in ("R0", "R5"):
        p = Path(root) / rung / "timing_memory.json"
        if p.exists():
            for r in bl.read_json(p)["records"]:
                if r["ok"] and r["peak_gpu_mem_mib"] is not None:
                    peaks.setdefault(r["config"], {})[rung] = max(peaks.get(r["config"], {}).get(rung, 0), r["peak_gpu_mem_mib"])
    for name, v in peaks.items():
        rows.append(f"| {name} | {v.get('R0', '–')} | {v.get('R5', '–')} |")
    return "\n".join(rows)


def regression_table(root, rungs):
    rows = ["| rung | set | configuration | result | statistic |", "|---|---|---|---|---|"]
    for rung in rungs:
        for set_name in ("ladder", "full"):
            p = Path(root) / rung / f"regression_{set_name}.json"
            if not p.exists():
                continue
            for c in bl.read_json(p)["configs"]:
                m = c.get("maps", {}).get("pixels", {})
                s = m.get("vs_truth") or m.get("vs_r0")
                stat = f"p99.9 {s['p999']:.2e}, max {s['max']:.2e}" if s else "; ".join(c["failures"])
                rows.append(f"| {bl.rung_label(rung)} | {set_name} | {c['config']} | {'pass' if c['pass'] else 'FAIL'} | {stat} |")
    return "\n".join(rows)


def caption(env):
    return (f"GPU {env.get('gpu', '?')} (nvidia-smi: name, driver, memory MiB, SM clock, max SM clock; "
            f"CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices')}), host {env.get('hostname')}, {env.get('nvcc')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="default: $IPM_BENCH_ROOT/results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.results or bl.bench_path("results"))
    inter = bl.read_json(root / "ladder_interleaved.json")
    summary, rungs = inter["summary"], inter["rungs"]
    parts = ["# IPM performance ladder: result tables", "", f"All tables: {caption(inter['env'])}.", ""]
    parts += ["## Ladder: median shooting time t_shoot_cells", "", ladder_table(summary, rungs, cf.LADDER, "median_t_shoot_cells"), ""]
    parts += ["## Ladder: median end-to-end wall time", "", ladder_table(summary, rungs, cf.LADDER, "median_wall_s"), ""]
    r0 = bl.read_json(root / "R0" / "timing_full.json")
    r5_path = root / "R5" / "timing_full.json"
    s0 = r0.get("summary") or summarize(r0["records"])
    s5 = (bl.read_json(r5_path).get("summary") or summarize(bl.read_json(r5_path)["records"])) if r5_path.exists() else {}
    parts += ["## Full matrix R0 vs R5: median shooting time", "", matrix_table(s0, s5, "median_t_shoot_cells"), ""]
    parts += ["## Full matrix R0 vs R5: median end-to-end wall time", "", matrix_table(s0, s5, "median_wall_s"), ""]
    parts += ["## Python wrapper: run() including the copy into numpy", "", python_table(root), ""]
    parts += ["## Peak device memory", "", memory_table(root), ""]
    parts += ["## Regression results", "", regression_table(root, rungs), ""]
    Path(a.out).write_text("\n".join(parts))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
