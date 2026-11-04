"""Login-node tests (no GPU) of the helpers of bench_batch.py and gpu_busy.py:
python -m unittest tests.batch.test_bench_batch -v"""
import io
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu"))

import bench_batch  # noqa: E402
import gpu_busy  # noqa: E402


def record(key, kind, config="small_f", mode="memory", gpus=1, maps_per_s=2.0, foreign=None, ok=True,
           save_root=None, maps=8, **extra):
    rec = {"key": key, "kind": kind, "config": config, "mode": mode, "save_root": save_root, "gpus": gpus,
           "maps": maps, "wall_s": 4.0, "maps_per_s": maps_per_s, "peak_rss_mib": 1000, "ok": ok,
           "chosen_gpus": [1], "foreign_before": foreign or {}, "foreign_after": {}}
    rec.update(extra)
    return rec


class TestBenchBatch(unittest.TestCase):
    def test_median(self):
        self.assertEqual(bench_batch.median([3, 1, 2]), 2)
        self.assertEqual(bench_batch.median([1, 2]), 1.5)

    def test_speedup(self):
        self.assertEqual(bench_batch.speedup({"maps_per_s": 2.0}, {"maps_per_s": 5.0}), 2.5)

    def test_render_table_has_one_row_per_result(self):
        results = {}
        for cfg, mode in (("small_f", "memory"), ("small_f", "save"), ("nb_f", "memory")):
            for rep in (0, 1):
                k = f"{cfg}|{mode}|1|serial|{rep}"
                results[k] = record(k, "serial", config=cfg, mode=mode, maps_per_s=2.0)
                for g in (1, 2):
                    k = f"{cfg}|{mode}|{g}|batch|{rep}"
                    results[k] = record(k, "batch", config=cfg, mode=mode, gpus=g, maps_per_s=5.0 * g)
        results["nb_f|memory|2|batch|1"]["foreign_before"] = {"0": [{"pid": 7, "used_mib": 100}]}
        lines = bench_batch.render_table(results).strip().splitlines()
        rows = [ln for ln in lines if ln.startswith("|")][2:]
        self.assertEqual(len(rows), 6)    # (config, mode, gpus) groups, repeats are medians
        self.assertEqual(sum("foreign" in ln for ln in rows), 1)
        self.assertIn("2.50", [ln for ln in rows if "small_f" in ln and "memory" in ln][0])
        self.assertIn("n=2/2", rows[0])

    def test_render_table_separates_save_roots_and_maps(self):
        results = {}
        for root, n in (("/tmp/a", 8), ("/nfs/b", 8), ("/tmp/a", 64)):
            for kind in ("serial", "batch"):
                k = f"small_f|save|{root}|1|{kind}|{n}|0"
                results[k] = record(k, kind, mode="save", save_root=root, maps=n)
        rows = [ln for ln in bench_batch.render_table(results).splitlines() if ln.startswith("|")][2:]
        self.assertEqual(len(rows), 3)

    def test_render_table_counts_ok_repeats_and_unknown_foreign(self):
        results = {}
        for rep, ok in enumerate((True, False)):
            k = f"small_f|memory|-|1|batch|8|{rep}"
            results[k] = record(k, "batch", ok=ok, foreign=None)
            results[k]["foreign_before"] = None      # nvidia-smi failed
        row = [ln for ln in bench_batch.render_table(results).splitlines() if ln.startswith("|")][2]
        self.assertIn("n=0/1", row)
        self.assertIn("unknown", row)
        self.assertNotIn("foreign", row)

    def test_render_table_without_serial_or_failed(self):
        k = "small_f|memory|1|batch|0"
        text = bench_batch.render_table({k: record(k, "batch", ok=False, maps_per_s=None)})
        self.assertIn("small_f", text)

    def test_choose_gpus_prefers_clean(self):
        self.assertEqual(bench_batch.choose_gpus([0, 1], 1, {0: [{"pid": 1}]}), [1])
        self.assertEqual(bench_batch.choose_gpus([0, 1], 2, {0: [{"pid": 1}]}), [0, 1])
        self.assertEqual(bench_batch.choose_gpus([0, 1, 2], 2, {}), [0, 1])
        with self.assertRaises(ValueError):
            bench_batch.choose_gpus([0], 2, {})

    def test_parse_foreign(self):
        apps = "00000000:01:00.0, 24223, 1474\n00000000:25:00.0, 5, 10\n"
        buses = "0, 00000000:01:00.0\n1, 00000000:25:00.0\n"
        got = bench_batch.parse_foreign(apps, buses, own_pids={5})
        self.assertEqual(got, {0: [{"pid": 24223, "used_mib": 1474}]})

    def test_plan_keys_and_maps(self):
        specs = bench_batch.plan(["small_f", "px40k_f"], [1, 2], ["memory"], 2, maps=4, save_root=None)
        keys = [s["key"] for s in specs]
        self.assertEqual(len(keys), len(set(keys)))
        px = [s for s in specs if s["config"] == "px40k_f" and s["kind"] == "batch" and s["gpus"] == 2]
        self.assertEqual(px[0]["maps"], 16)
        self.assertEqual([s["maps"] for s in specs if s["config"] == "small_f"], [4] * 6)
        self.assertTrue(all(s["gpus"] == 1 for s in specs if s["kind"] == "serial"))

    def test_keys_differ_with_save_root_and_maps(self):
        def keys(root, maps):
            return {s["key"] for s in bench_batch.plan(["nb_f"], [1, 2], ["memory", "save"], 2, maps=maps, save_root=root)}
        a, b, c = keys("/tmp/x", 64), keys("/nfs/x", 64), keys("/tmp/x", 512)
        self.assertEqual(a & c, set())                    # maps are in every key
        shared = a & b                                    # only memory keys, which have no save root, may coincide
        self.assertTrue(shared and all("|memory|" in k for k in shared))
        self.assertEqual({k for k in a if "|save|" in k} & b, set())
        multi = bench_batch.plan(["nb_f"], [1], ["save"], 1, maps=8, save_root=["/tmp/x", "/nfs/x"])
        self.assertEqual(len({s["key"] for s in multi}), len(multi))
        self.assertEqual({s["save_root"] for s in multi}, {"/tmp/x", "/nfs/x"})

    def test_resume_retries_failed_records(self):
        specs = bench_batch.plan(["small_f"], [1], ["memory"], 1, maps=4, save_root=None)
        bad = specs[0]["key"]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            bench_batch.write_json(out, {bad: record(bad, "serial", ok=False, error="boom")})
            results = bench_batch.run_plan(specs, out, lambda sp: record(sp["key"], sp["kind"]))
        self.assertTrue(results[bad]["ok"])
        self.assertEqual(results[bad]["previous_errors"], ["boom"])

    def test_timeout_removes_save_directory(self):
        with tempfile.TemporaryDirectory() as d:
            spec = bench_batch.plan(["small_f"], [1], ["save"], 1, maps=4, save_root=d)[0]
            seen = []

            def hang(cmd, **kw):
                save_dir = Path(cmd[cmd.index("--save-dir") + 1])
                self.assertTrue(save_dir.is_dir())
                (save_dir / "partial.bin").write_bytes(b"x")
                seen.append(save_dir)
                raise subprocess.TimeoutExpired(cmd, 1, stderr=b"slow")

            with mock.patch.object(bench_batch.subprocess, "run", hang), \
                    mock.patch.object(bench_batch, "allocated_gpus", lambda: [0]), \
                    mock.patch.object(bench_batch, "foreign_processes", lambda own=(): {}):
                rec = bench_batch.launch_child(spec, 1)
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].exists())
            self.assertEqual(list(Path(d).iterdir()), [])
            self.assertFalse(rec["ok"])
            self.assertIn("timeout", rec["error"])
            self.assertEqual(rec["stderr_tail"], "slow")

    def test_launch_child_with_failing_nvidia_smi_is_unknown(self):
        spec = bench_batch.plan(["small_f"], [1], ["memory"], 1, maps=4, save_root=None)[1]
        done = subprocess.CompletedProcess([], 0, stdout='{"ok": true, "maps_per_s": 3.0, "maps": 4, "peak_rss_mib": 5}\n',
                                           stderr="")
        with mock.patch.object(bench_batch.subprocess, "run", lambda *a, **k: done), \
                mock.patch.object(bench_batch, "allocated_gpus", lambda: [0, 1]), \
                mock.patch.object(bench_batch, "foreign_processes", lambda own=(): None):
            rec = bench_batch.launch_child(spec, 1)
        self.assertEqual(rec["chosen_gpus"], [0])
        self.assertIsNone(rec["foreign_before"])
        self.assertIsNone(rec["foreign_after"])
        row = [ln for ln in bench_batch.render_table({rec["key"]: rec}).splitlines() if ln.startswith("|")][2]
        self.assertIn("unknown", row)

    def test_foreign_processes_none_when_smi_fails(self):
        with mock.patch.object(bench_batch, "_smi", lambda *a: None):
            self.assertIsNone(bench_batch.foreign_processes())

    WARMUP = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25, half_length_y2=25,
                  num_pixels_y1=1000, num_pixels_y2=1000)

    def test_warm_up_is_the_same_for_every_configuration(self):
        # the warm-up only creates the CUDA contexts and loads the kernels: a 1000^2 map of the nb lens whatever the
        # configuration (a 200^2 map of the stars1e6_f lens fails with "Expansion order needs to be >= 3")
        for cfg in bench_batch.CONFIGS:
            self.assertEqual(bench_batch.warmup_params(cfg), self.WARMUP, cfg)

    def test_measure_warms_up_with_the_nb_lens(self):
        calls = []

        class Fake:
            last_run_stats = {}

            def __init__(self, **kw):
                calls.append((type(self).__name__, kw))

            def run(self, save=False):
                return iter(())

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

            def close(self):
                pass

        class FakeIPM(Fake):
            pass

        class FakeBatch(Fake):
            pass

        with mock.patch("microlensing.IPM.ipm.IPM", FakeIPM), \
                mock.patch("microlensing.IPM.batch.BatchedIPM", FakeBatch):
            for cfg in bench_batch.CONFIGS:
                calls.clear()
                bench_batch.measure(cfg, "batch", "memory", 2, 4, None)
                (_, warm_ipm), (_, warm_batch), (_, timed) = calls
                self.assertEqual({k: v for k, v in warm_ipm.items() if k != "random_seed"}, self.WARMUP, cfg)
                self.assertEqual({k: v for k, v in warm_batch.items()
                                  if k not in ("num_maps", "random_seed", "devices")}, self.WARMUP, cfg)
                self.assertEqual((warm_batch["num_maps"], warm_batch["devices"]), (2, [0, 1]), cfg)
                self.assertEqual({k: timed[k] for k in bench_batch.CONFIGS[cfg]}, bench_batch.CONFIGS[cfg], cfg)

    def test_resume_skips_existing_keys(self):
        specs = bench_batch.plan(["small_f"], [1], ["memory"], 1, maps=4, save_root=None)
        done = specs[0]["key"]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            bench_batch.write_json(out, {done: record(done, "serial")})
            launched = []

            def launch(spec):
                launched.append(spec["key"])
                return record(spec["key"], spec["kind"])

            results = bench_batch.run_plan(specs, out, launch)
            self.assertNotIn(done, launched)
            self.assertEqual(len(launched), len(specs) - 1)
            self.assertEqual(set(bench_batch.read_json(out)), {s["key"] for s in specs})
            self.assertEqual(set(results), {s["key"] for s in specs})


class TestGpuBusy(unittest.TestCase):
    def test_gpu_busy_union_and_overlap(self):
        s = gpu_busy.busy_stats([(0, 1), (0.5, 2), (3, 4)])
        self.assertAlmostEqual(s["wall_s"], 4)
        self.assertAlmostEqual(s["busy_fraction"], 3 / 4)
        self.assertAlmostEqual(s["overlap_fraction"], 0.5 / 3)

    def test_read_trace_keeps_kernels_only(self):
        text = ("Processing events...\n"
                '"Start (ns)","Duration (ns)","GrdX","Device","Name"\n'
                '"0","1000000000","8","NVIDIA A100 (0)","k1"\n'
                '"500000000","1500000000","8","NVIDIA A100 (1)","k2"\n'
                '"0","5","","NVIDIA A100 (0)","[CUDA memcpy HtoD]"\n')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text(text)
            iv = gpu_busy.read_trace(p)
            devs = gpu_busy.read_trace_devices(p)
        self.assertEqual(iv, [(0.0, 1.0), (0.5, 2.0)])
        self.assertEqual(devs, {"NVIDIA A100 (0)": [(0.0, 1.0)], "NVIDIA A100 (1)": [(0.5, 2.0)]})

    def test_read_trace_without_grid_column_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text('"Start (ns)","Duration (ns)","Name"\n"0","5","[CUDA memcpy HtoD]"\n')
            with self.assertRaisesRegex(ValueError, "kernel-identifying"):
                gpu_busy.read_trace(p)

    def test_after_gap(self):
        ivs = [(0, 1), (0.5, 1.2), (2, 3), (3.1, 4)]
        self.assertEqual(gpu_busy.after_gap(ivs, 0.5), [(2, 3), (3.1, 4)])
        with self.assertRaises(ValueError):
            gpu_busy.after_gap(ivs, 5)

    def test_main_after_gap_and_devices(self):
        import contextlib
        text = ('"Start (ns)","Duration (ns)","GrdX","Device","Name"\n'
                '"0","1000000000","8","d0","warm"\n'
                '"2000000000","1000000000","8","d0","k"\n'
                '"2000000000","1000000000","8","d1","k"\n')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text(text)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gpu_busy.main([str(p), "--after-gap", "0.5"])
        out = buf.getvalue()
        self.assertIn("wall_s 1.000000", out)
        self.assertIn("overlap_fraction 1.000000", out)
        self.assertIn("device d1: busy_fraction 1.000000", out)


if __name__ == "__main__":
    unittest.main()
