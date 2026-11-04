"""Unit tests for the IPM GPU benchmark harness. No GPU needed:
python -m unittest discover -s tests/gpu -v"""
import contextlib
import io
import json
import os
import re
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_lib  # noqa: E402


def write_map(path, arr):
    arr = np.asarray(arr)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", arr.shape[0], arr.shape[1]))
        f.write(arr.tobytes())


class ReadMapTests(unittest.TestCase):
    def test_float32_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.bin"
            arr = np.arange(12, dtype=np.float32).reshape(3, 4)
            write_map(p, arr)
            m = bench_lib.read_map(p)
            self.assertEqual(m.dtype, np.float64)
            self.assertEqual(m.shape, (3, 4))
            np.testing.assert_array_equal(m, arr.astype(np.float64))

    def test_float64_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.bin"
            arr = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
            write_map(p, arr)
            np.testing.assert_array_equal(bench_lib.read_map(p), arr)

    def test_size_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.bin"
            with open(p, "wb") as f:
                f.write(struct.pack("ii", 3, 4))
                f.write(b"\0" * 10)
            with self.assertRaises(ValueError):
                bench_lib.read_map(p)


class ParamInfoTests(unittest.TestCase):
    def test_parses_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ipm_parameter_info.txt"
            p.write_text("kappa_tot 0.4\nmass_function equal\nt_shoot_cells 8.123\n")
            info = bench_lib.read_param_info(p)
            self.assertEqual(info["kappa_tot"], "0.4")
            self.assertEqual(info["mass_function"], "equal")
            self.assertEqual(bench_lib.t_shoot_cells(info), 8.123)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError) as ctx:
            bench_lib.t_shoot_cells({"kappa_tot": "0.4"})
        self.assertIn("t_shoot_cells", str(ctx.exception))


class RungNameTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(bench_lib.rung_label("R0p"), "R0′")
        self.assertEqual(bench_lib.rung_label("R3"), "R3")
        self.assertEqual(bench_lib.RUNGS[0], "R0")
        self.assertEqual(bench_lib.RUNGS[-1], "R5")


import compare_maps  # noqa: E402


class MetricsTests(unittest.TestCase):
    def base(self):
        rng = np.random.default_rng(1)
        return 5.0 + rng.random((50, 40))

    def test_identical_maps_give_zero(self):
        b = self.base()
        m = compare_maps.map_metrics(b.copy(), b)
        for key in ("mean", "p999", "p9999", "max", "offmask_max", "sum_ratio"):
            self.assertEqual(m[key], 0.0, key)
        self.assertTrue(m["finite"] and m["nonneg"])
        self.assertEqual(m["n_mask"], b.size)

    def test_planted_difference(self):
        b = self.base()
        a = b.copy()
        a[3, 4] *= 1.001          # +1e-3 relative on one pixel
        m = compare_maps.map_metrics(a, b)
        self.assertAlmostEqual(m["max"], 1e-3, places=9)
        self.assertLess(m["p999"], 1e-3)       # one pixel of 2000 is below the 99.9th percentile
        self.assertGreater(m["p9999"], 0.0)
        self.assertAlmostEqual(m["mean"], 1e-3 / b.size, places=12)
        self.assertAlmostEqual(m["sum_ratio"], 1e-3 * b[3, 4] / b.sum(), places=12)

    def test_floor_excludes_low_pixels(self):
        b = self.base()
        b[0, 0] = 1e-6            # far below 1e-3 x mean
        a = b.copy()
        a[0, 0] = 2e-6            # 100 % relative, but off the mask
        m = compare_maps.map_metrics(a, b)
        self.assertEqual(m["n_mask"], b.size - 1)
        self.assertEqual(m["max"], 0.0)
        self.assertAlmostEqual(m["offmask_max"], 1e-6 / b.mean(), places=12)

    def test_nan_and_negative_fail(self):
        b = self.base()
        a = b.copy()
        a[1, 1] = np.nan
        m = compare_maps.map_metrics(a, b)
        self.assertFalse(m["finite"])
        self.assertFalse(compare_maps.evaluate_double(m, compare_maps.default_tolerances())["pass"])
        a = b.copy()
        a[1, 1] = -1.0
        m = compare_maps.map_metrics(a, b)
        self.assertTrue(m["finite"])
        self.assertFalse(m["nonneg"])
        self.assertFalse(compare_maps.evaluate_double(m, compare_maps.default_tolerances())["pass"])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            compare_maps.map_metrics(np.ones((3, 4)), np.ones((4, 3)))
        self.assertIn("(3, 4)", str(ctx.exception))
        self.assertIn("(4, 3)", str(ctx.exception))

    def test_parity_consistency(self):
        total = self.base()
        minima = 0.3 * total
        saddles = 0.7 * total
        self.assertLess(compare_maps.parity_consistency(total, minima, saddles), 1e-12)
        saddles[2, 2] += 1.0
        self.assertGreater(compare_maps.parity_consistency(total, minima, saddles), 0.1)


class EvaluateTests(unittest.TestCase):
    def metrics(self, **over):
        m = {"mean": 1e-9, "p999": 1e-7, "p9999": 1e-7, "max": 1e-6, "offmask_max": 1e-8,
             "sum_ratio": 1e-8, "n_mask": 10, "n_total": 10, "finite": True, "nonneg": True}
        m.update(over)
        return m

    def test_double_pass_and_fail(self):
        tol = compare_maps.default_tolerances()
        self.assertTrue(compare_maps.evaluate_double(self.metrics(), tol)["pass"])
        v = compare_maps.evaluate_double(self.metrics(p999=2e-5), tol)
        self.assertFalse(v["pass"])
        self.assertTrue(any("p999" in f for f in v["failures"]))
        v = compare_maps.evaluate_double(self.metrics(sum_ratio=-2e-6), tol)
        self.assertFalse(v["pass"])

    def test_float_rule_uses_ratio_to_reference_error(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2, "sum_ratio": 0.0}
        ok = compare_maps.evaluate_float({"mean": 1.2e-5, "p999": 1.4e-3, "p9999": 4e-3, "max": 1.5e-2, "sum_ratio": 0.0},
                                         ref, self.metrics(), tol)
        self.assertTrue(ok["pass"])
        self.assertEqual(ok["flags"], [])
        bad = compare_maps.evaluate_float({"mean": 1.2e-5, "p999": 1.6e-3, "p9999": 4e-3, "max": 5e-2, "sum_ratio": 0.0},
                                          ref, self.metrics(), tol)
        self.assertFalse(bad["pass"])
        self.assertTrue(any("p999" in f for f in bad["failures"]))
        self.assertTrue(any("max" in f for f in bad["flags"]))
        bad_sum = compare_maps.evaluate_float({**ref, "sum_ratio": 2e-5}, ref, self.metrics(), tol)
        self.assertFalse(bad_sum["pass"])


class ToleranceTests(unittest.TestCase):
    def test_load_merges_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tolerances.json"
            p.write_text(json.dumps({"double": {"p999": 2e-5}, "float_truth_error": {"nb_f": {"pixels": {"mean": 1}}}}))
            tol = compare_maps.load_tolerances(p)
            self.assertEqual(tol["double"]["p999"], 2e-5)
            self.assertEqual(tol["double"]["mean"], compare_maps.default_tolerances()["double"]["mean"])
            self.assertEqual(tol["float_truth_error"]["nb_f"]["pixels"]["mean"], 1)
            self.assertEqual(tol["floor_frac"], 1e-3)

    def test_missing_file_gives_defaults(self):
        self.assertEqual(compare_maps.load_tolerances("/nonexistent/t.json"), compare_maps.default_tolerances())


class EvaluateRuleTests(unittest.TestCase):
    def test_double_actual_pass_fail_and_flag(self):
        tol = compare_maps.default_tolerances()
        tol["double_actual"]["default"].update(mean=2e-6, p999=3e-5, p9999=4e-4, offmask_max=1e-5, sum_ratio=1e-6, flag_max=1e-2)
        m = {"mean": 1e-6, "p999": 1e-5, "p9999": 1e-4, "max": 5e-3, "offmask_max": 0.0, "sum_ratio": -1e-7,
             "finite": True, "nonneg": True}
        v = compare_maps.evaluate_double_actual(m, tol, "nb_d")   # no entry of its own: the default applies
        self.assertEqual((v["pass"], v["failures"], v["flags"]), (True, [], []))
        for key, bad in (("mean", 3e-6), ("p999", 4e-5), ("p9999", 5e-4), ("offmask_max", 2e-5), ("sum_ratio", -2e-6)):
            v = compare_maps.evaluate_double_actual({**m, key: bad}, tol, "nb_d")
            self.assertFalse(v["pass"], key)
            self.assertEqual(len(v["failures"]), 1, key)
            self.assertTrue(v["failures"][0].endswith("[default]"), v["failures"])
        v = compare_maps.evaluate_double_actual({**m, "max": 2e-2}, tol, "nb_d")   # a large max alone is a flag, not a failure
        self.assertTrue(v["pass"])
        self.assertEqual(len(v["flags"]), 1)
        self.assertFalse(compare_maps.evaluate_double_actual({**m, "nonneg": False}, tol, "nb_d")["pass"])

    def test_double_actual_uses_the_configurations_own_entry_and_names_it(self):
        tol = compare_maps.default_tolerances()
        tol["double_actual"]["nb_d"] = {**tol["double_actual"]["default"], "mean": 1e-6, "flag_max": 1e-2}
        m = {"mean": 5e-7, "p999": 0.0, "p9999": 0.0, "max": 5e-3, "offmask_max": 0.0, "sum_ratio": 0.0,
             "finite": True, "nonneg": True}
        self.assertTrue(compare_maps.evaluate_double_actual(m, tol, "nb_d")["pass"])      # own entry: mean limit 1e-6
        v = compare_maps.evaluate_double_actual(m, tol, "rect_d")                         # default: mean limit 1e-7
        self.assertFalse(v["pass"])
        self.assertIn("[default]", v["failures"][0])
        self.assertEqual(len(v["flags"]), 1)                                              # max 5e-3 > default flag_max 1e-3
        v = compare_maps.evaluate_double_actual({**m, "mean": 2e-6}, tol, "nb_d")
        self.assertIn("[nb_d]", v["failures"][0])

    def test_default_double_actual_matches_double_and_load_merges(self):
        d = compare_maps.default_tolerances()
        self.assertEqual(list(d["double_actual"]), ["default"])
        self.assertEqual({k: v for k, v in d["double_actual"]["default"].items() if k != "flag_max"}, d["double"])
        self.assertEqual(d["double_actual"]["default"]["flag_max"], 1e-3)
        self.assertEqual(d["double_shift_sensitivity"], {})
        with tempfile.TemporaryDirectory() as t:
            Path(t, "tol.json").write_text(json.dumps({"double_actual": {"nb_d": {"mean": 5e-6}}}))
            tol = compare_maps.load_tolerances(Path(t, "tol.json"))
        self.assertEqual(tol["double_actual"]["nb_d"], {"mean": 5e-6})   # an entry replaces nothing but itself
        self.assertEqual(tol["double_actual"]["default"], d["double_actual"]["default"])

    def metrics(self, **over):
        m = {"mean": 1e-9, "p999": 1e-7, "p9999": 1e-7, "max": 1e-6, "offmask_max": 1e-8,
             "sum_ratio": 1e-8, "n_mask": 10, "n_total": 10, "finite": True, "nonneg": True}
        m.update(over)
        return m

    def test_double_threshold_for_each_key(self):
        tol = compare_maps.default_tolerances()
        for key in ("mean", "p999", "p9999", "offmask_max"):
            thr = tol["double"][key]
            with self.subTest(key=key):
                above = compare_maps.evaluate_double(self.metrics(**{key: 1.01 * thr}), tol)
                self.assertFalse(above["pass"])
                self.assertTrue(any(f.startswith(key + "=") for f in above["failures"]), above["failures"])
                below = compare_maps.evaluate_double(self.metrics(**{key: 0.99 * thr}), tol)
                self.assertTrue(below["pass"], below["failures"])

    def test_float_common_failures_gate_the_rule(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2, "sum_ratio": 0.0}
        v = compare_maps.evaluate_float(ref, ref, self.metrics(finite=False), tol)
        self.assertFalse(v["pass"])
        self.assertIn("non-finite pixels", v["failures"])
        v = compare_maps.evaluate_float(ref, ref, self.metrics(nonneg=False), tol)
        self.assertFalse(v["pass"])
        self.assertIn("negative pixels", v["failures"])

    def test_float_max_excess_is_a_flag_not_a_failure(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2, "sum_ratio": 0.0}
        m_truth = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 4e-2, "sum_ratio": 0.0}  # max above 3.0x R0's, rest within 1.5x
        v = compare_maps.evaluate_float(m_truth, ref, self.metrics(), tol)
        self.assertIs(v["pass"], True)
        self.assertEqual(v["failures"], [])
        self.assertTrue(v["flags"])
        self.assertTrue(any(f.startswith("max ") for f in v["flags"]), v["flags"])

    def test_float_mean_limit_alone_fails(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2, "sum_ratio": 0.0}
        m_truth = {"mean": 1.6e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2, "sum_ratio": 0.0}  # mean above 1.5x, rest within
        v = compare_maps.evaluate_float(m_truth, ref, self.metrics(), tol)
        self.assertFalse(v["pass"])
        self.assertEqual(len(v["failures"]), 1)
        self.assertTrue(v["failures"][0].startswith("mean "), v["failures"])

    def test_float_p9999_limit_alone_fails(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2, "sum_ratio": 0.0}
        m_truth = {"mean": 1e-5, "p999": 1e-3, "p9999": 5e-3, "max": 1e-2, "sum_ratio": 0.0}  # p9999 above 1.5x, rest within
        v = compare_maps.evaluate_float(m_truth, ref, self.metrics(), tol)
        self.assertFalse(v["pass"])
        self.assertEqual(len(v["failures"]), 1)
        self.assertTrue(v["failures"][0].startswith("p9999 "), v["failures"])

    def test_float_sum_ratio_is_judged_against_the_truth_with_an_absolute_floor(self):
        tol = compare_maps.default_tolerances()
        stats = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        # the limit is max(1e-5, 1.5 x |R0's sum error|), applied to |sum_ratio| (either sign)
        for ref_sum, limit in ((0.0, 1e-5), (-9e-5, 1.35e-4)):
            ref = {**stats, "sum_ratio": ref_sum}
            for sign in (1, -1):
                ok = compare_maps.evaluate_float({**stats, "sum_ratio": sign * 0.99 * limit}, ref, self.metrics(), tol)
                self.assertTrue(ok["pass"], (ref_sum, sign, ok["failures"]))
                bad = compare_maps.evaluate_float({**stats, "sum_ratio": sign * 1.01 * limit}, ref, self.metrics(), tol)
                self.assertFalse(bad["pass"], (ref_sum, sign))
                self.assertTrue(any(f.startswith("|sum_ratio| vs truth") and "x R0's" in f for f in bad["failures"]), bad["failures"])
        # the direct sum against R0-float no longer decides
        ref = {**stats, "sum_ratio": 0.0}
        self.assertTrue(compare_maps.evaluate_float({**stats, "sum_ratio": 0.0}, ref, self.metrics(sum_ratio=1e-3), tol)["pass"])

    def test_double_negative_off_mask_pixel_fails(self):
        rng = np.random.default_rng(1)
        b = 5.0 + rng.random((50, 40))
        b[0, 0] = 1e-9                  # far below 1e-3 x mean, so off the mask
        a = b.copy()
        a[0, 0] = -1e-9
        m = compare_maps.map_metrics(a, b)
        self.assertTrue(m["finite"])
        self.assertFalse(m["nonneg"])
        self.assertEqual(m["mean"], 0.0)
        self.assertEqual(m["p999"], 0.0)
        v = compare_maps.evaluate_double(m, compare_maps.default_tolerances())
        self.assertFalse(v["pass"])
        self.assertIn("negative pixels", v["failures"])

    def test_parity_consistency_hand_value_and_scale(self):
        total = np.array([[2.0, 4.0], [6.0, 8.0]])          # mean(total) = 5
        minima = np.array([[1.0, 2.0], [3.0, 4.0]])
        saddles = np.array([[1.0, 2.8], [3.0, 3.5]])
        # total - (minima + saddles) = [[0, -0.8], [0, 0.5]], so max |.| / mean(total) = 0.8 / 5
        value = compare_maps.parity_consistency(total, minima, saddles)
        self.assertAlmostEqual(value, 0.8 / 5.0, places=12)
        scaled = compare_maps.parity_consistency(10 * total, 10 * minima, 10 * saddles)
        self.assertAlmostEqual(scaled, value, places=12)


import configs  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_every_float_config_has_a_double_truth_with_identical_args(self):
        for cfg in configs.CONFIGS.values():
            self.assertIn(cfg.build, ("float", "double"), cfg.name)
            if cfg.build == "float":
                truth = configs.CONFIGS[cfg.truth]
                self.assertEqual(truth.build, "double", cfg.name)
                self.assertEqual(truth.args, cfg.args, cfg.name)
                self.assertEqual(truth.starfile_from, cfg.starfile_from, cfg.name)

    def test_ladder_and_sets(self):
        self.assertEqual(configs.LADDER, ["nb_f", "nb_d", "rect_f", "hm100_f"])
        names = [c.name for c in configs.ladder_configs()]
        self.assertEqual(names, configs.LADDER)
        self.assertEqual([c.name for c in configs.ladder_configs("R2p")], ["nb_d"])
        full = {c.name for c in configs.configs_in("full")}
        self.assertEqual(len(full), 15)
        self.assertEqual({c.name for c in configs.regression_configs("R2p", "ladder")},
                         {"nb_d", "rect_d", "rect_a0_d", "hm100_d"})
        self.assertEqual({c.name for c in configs.regression_configs("R3", "full")}, full)
        self.assertEqual({c.name for c in configs.configs_in("memory")}, {"big_f", "hm1000_f", "stars1e6_f"})

    def test_full_args_and_starfile(self):
        cfg = configs.CONFIGS["sf_f"]
        args = configs.full_args(cfg, "/ref")
        self.assertIn("-sf", args)
        self.assertEqual(args[args.index("-sf") + 1], "/ref/nb_f/run1/ipm_stars.bin")
        self.assertEqual(args[:2], ["-rs", configs.SEED])
        self.assertNotIn("-v", args)
        self.assertNotIn("-o", args)
        self.assertEqual(len(args) % 2, 0)

    def test_ordered_puts_starfile_users_last(self):
        # reversed declaration order puts every star-file user ahead of its source
        order = [c.name for c in configs.ordered(list(reversed(list(configs.CONFIGS.values()))))]
        self.assertEqual(sorted(order), sorted(configs.CONFIGS))
        users = [i for i, n in enumerate(order) if configs.CONFIGS[n].starfile_from is not None]
        others = [i for i, n in enumerate(order) if configs.CONFIGS[n].starfile_from is None]
        self.assertGreater(min(users), max(others))
        self.assertGreater(order.index("sf_f"), order.index("nb_f"))
        self.assertGreater(order.index("sf_d"), order.index("nb_f"))

    def test_exact_membership_of_each_set(self):
        self.assertEqual({c.name for c in configs.configs_in("full")}, {
            "nb_f", "nb_d", "rect_f", "hm100_f", "circ_a0_f", "rect_a0_f", "rect_d", "rect_a0_d",
            "par_f", "nry10_f", "big_f", "hm1000_f", "hm100_d", "sf_f", "stars1e6_f"})
        self.assertEqual({c.name for c in configs.configs_in("truth")}, {
            "circ_a0_d", "par_d", "nry10_d", "big_d", "hm1000_d", "sf_d", "stars1e6_d"})
        self.assertEqual({c.name for c in configs.configs_in("ladder")}, {"nb_f", "nb_d", "rect_f", "hm100_f"})

    def test_binding_constraints_hold_for_every_configuration(self):
        for cfg in configs.CONFIGS.values():
            with self.subTest(config=cfg.name):
                self.assertIn(cfg.build, ("float", "double"))
                self.assertEqual(cfg.parities, "-wp" in cfg.args)
                args = configs.full_args(cfg, "/ref")
                self.assertEqual(args[:len(configs.COMMON)], list(configs.COMMON))
                self.assertNotIn("-v", args)
                self.assertNotIn("-o", args)
                self.assertEqual(len(args) % 2, 0)

    def test_hm1000_differs_from_hm100_only_in_kappa_and_shear(self):
        hm100 = list(configs.CONFIGS["hm100_f"].args)
        hm1000 = list(configs.CONFIGS["hm1000_f"].args)
        k = hm100.index("-k") + 1
        self.assertEqual(hm100[k], "0.5")
        y = hm100.index("-y") + 1
        self.assertEqual(hm100[y], "0.49")
        a = hm100.index("-a") + 1
        self.assertEqual(hm100[a], "1")
        expected = list(hm100)
        expected[k] = "0.7252"
        expected[y] = "0.2729"
        self.assertEqual(hm1000, expected)
        self.assertEqual(hm1000[a], "1")

    def test_shifted_args_and_double_regression_configs(self):
        cfg = configs.CONFIGS["nb_d"]
        self.assertEqual(configs.shifted_args(cfg, "/ref"), configs.full_args(cfg, "/ref") + ["-cy1", "1e-7", "-cy2", "1e-7"])
        self.assertEqual([c.name for c in configs.double_regression_configs()], ["nb_d", "rect_d", "rect_a0_d", "hm100_d"])
        for c in configs.CONFIGS.values():   # the shift must not collide with a centre already in a configuration
            self.assertFalse({"-cy1", "-cy2"} & set(c.args), c.name)

    def test_exe_for(self):
        self.assertEqual(configs.exe_for("/b", "float").name, "ipm_gpu")
        self.assertEqual(configs.exe_for("/b", "double").name, "ipm_gpu_double")


FAKE = str(HERE / "fake_ipm.py")


class RunIpmTests(unittest.TestCase):
    def run_fake(self, env_over=None, **kw):
        env = dict(os.environ)
        env.update(env_over or {})
        with tempfile.TemporaryDirectory() as d:
            rec = bench_lib.run_ipm(sys.executable, [FAKE, "-npy1", "4", "-npy2", "3"], Path(d) / "out", env=env, **kw)
            files = sorted(p.name for p in (Path(d) / "out").iterdir())
            return rec, files

    def test_success_record(self):
        rec, files = self.run_fake()
        self.assertTrue(rec["ok"], rec)
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["returncode"], 0)
        self.assertEqual(rec["t_shoot_cells"], 0.123)
        self.assertGreater(rec["wall_s"], 0.0)
        self.assertEqual(rec["param_info"]["num_stars"], "137")
        self.assertIn("ipm_magnifications.bin", files)
        self.assertIn("fake run done", rec["stdout_tail"])
        self.assertEqual(rec["args"][-2:], ["-npy2", "3"])   # -v and -o are not part of args

    def test_failure_exit_code(self):
        rec, _ = self.run_fake({"FAKE_IPM_FAIL": "1"})
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["returncode"], 3)
        self.assertIn("exit code 3", rec["error"])
        self.assertIn("fake failure", rec["stderr_tail"])

    def test_missing_param_file_not_ok(self):
        rec, _ = self.run_fake({"FAKE_IPM_NO_PARAM": "1"})
        self.assertFalse(rec["ok"])
        self.assertIn("ipm_parameter_info.txt", rec["error"])
        self.assertIsNone(rec["t_shoot_cells"])

    def test_missing_map_not_ok(self):
        rec, _ = self.run_fake({"FAKE_IPM_NO_MAP": "1"})
        self.assertFalse(rec["ok"])
        self.assertIn("ipm_magnifications.bin", rec["error"])
        rec, _ = self.run_fake({"FAKE_IPM_NO_MAP": "1"}, expect_map=False)
        self.assertTrue(rec["ok"])

    def test_old_outputs_are_removed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            out.mkdir()
            (out / "ipm_stale.bin").write_bytes(b"x")
            bench_lib.run_ipm(sys.executable, [FAKE], out)
            self.assertFalse((out / "ipm_stale.bin").exists())

    def test_timeout_gives_failed_record(self):
        rec, _ = self.run_fake({"FAKE_IPM_SLEEP": "3"}, timeout=0.5)
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["returncode"], -1)
        self.assertIn("timeout", rec["error"])
        self.assertIsInstance(rec["stdout_tail"], str)
        with tempfile.TemporaryDirectory() as d:
            bench_lib.write_json(Path(d) / "rec.json", rec)
            self.assertEqual(bench_lib.read_json(Path(d) / "rec.json")["error"], rec["error"])

    def test_timeout_keeps_partial_output_as_text(self):
        # The fake prints nothing before its sleep, so this child prints first to exercise the bytes path.
        code = "import time; print('partial', flush=True); time.sleep(30)"
        with tempfile.TemporaryDirectory() as d:
            rec = bench_lib.run_ipm(sys.executable, ["-c", code], Path(d) / "out", timeout=2)
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["stdout_tail"], "partial\n")
            bench_lib.write_json(Path(d) / "rec.json", rec)

    def test_missing_executable_gives_failed_record(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), tempfile.TemporaryDirectory() as d:
            before = threading.active_count()
            rec = bench_lib.run_ipm("/nonexistent/ipm_gpu", [], Path(d) / "out", sample_gpu=True)
            after = threading.active_count()
        self.assertFalse(rec["ok"])
        self.assertIn("cannot start", rec["error"])
        self.assertIn("/nonexistent/ipm_gpu", rec["error"])
        self.assertIsNone(rec["returncode"])
        self.assertEqual((rec["stdout_tail"], rec["stderr_tail"]), ("", ""))
        self.assertEqual(before, after)


class SamplerTests(unittest.TestCase):
    def test_no_index_gives_none(self):
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-b5449ec5-cd58-e6c6-e691-901514c1de2c"
            self.assertIsNone(bench_lib.gpu_index())
            self.assertIsNone(bench_lib.GpuMemorySampler().start().stop())
            os.environ.pop("CUDA_VISIBLE_DEVICES")
            self.assertIsNone(bench_lib.gpu_index())
            os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
            self.assertEqual(bench_lib.gpu_index(), "4")
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved


class HelperTests(unittest.TestCase):
    def test_median(self):
        self.assertEqual(bench_lib.median([3, 1, 2]), 2)
        self.assertEqual(bench_lib.median([4, 1, 3, 2]), 2.5)
        self.assertEqual(bench_lib.median([None, 5]), 5)
        self.assertIsNone(bench_lib.median([]))

    def test_json_roundtrip_and_build_meta(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "x.json"
            bench_lib.write_json(p, {"a": [1, 2]})
            self.assertEqual(bench_lib.read_json(p), {"a": [1, 2]})
            (Path(d) / "git_rev.txt").write_text("abc123 clean\n")
            meta = bench_lib.build_meta(d)
            self.assertEqual(meta["git_rev"], "abc123 clean")
            self.assertIsNone(meta["nvcc_version"])

    def test_environment_info_has_keys(self):
        info = bench_lib.environment_info()
        for key in ("hostname", "slurm_job_id", "cuda_visible_devices", "gpu", "nvcc", "timestamp"):
            self.assertIn(key, info)


import ptxas_stats  # noqa: E402


class PtxasTests(unittest.TestCase):
    LOG = """=== ipm_gpu ===
ptxas info    : Compiling entry function '_Z16histogram_kernelIfEvPT_7ComplexIiEiPii' for 'sm_80'
ptxas info    : Function properties for _Z16histogram_kernelIfEvPT_7ComplexIiEiPii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z18shoot_cells_kernelIfEvT_S0_S0_P4starIS0_ES0_P8TreeNodeIS0_Ei7ComplexIS0_EiiS7_S6_IiES7_S7_S7_S7_PS0_S9_S9_S8_Pyi' for 'sm_80'
ptxas info    : Function properties for _Z18shoot_cells_kernelIfEvT_S0_S0_P4starIS0_ES0_P8TreeNodeIS0_Ei7ComplexIS0_EiiS7_S6_IiES7_S7_S7_S7_PS0_S9_S9_S8_Pyi
    376 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 74 registers, 544 bytes cmem[0]
=== ipm_gpu_double ===
ptxas info    : Compiling entry function '_ZN8ipm_fast24shoot_cells_tiled_kernelIdfEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E' for 'sm_80'
ptxas info    : Function properties for _ZN8ipm_fast24shoot_cells_tiled_kernelIdfEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E
    48 bytes stack frame, 56 bytes spill stores, 56 bytes spill loads
ptxas info    : Used 80 registers, 6168 bytes smem, 760 bytes cmem[0]
=== lib_ipm.so ===
ptxas info    : Compiling entry function '_ZN8ipm_fast24shoot_cells_tiled_kernelIffEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E' for 'sm_80'
ptxas info    : Function properties for _ZN8ipm_fast24shoot_cells_tiled_kernelIffEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E
    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 74 registers, 16 bytes cumulative stack size, 6168 bytes smem, 760 bytes cmem[0]
"""

    def test_parse_shoot_kernels(self):
        rows = ptxas_stats.parse(self.LOG)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["target"], "ipm_gpu")
        self.assertEqual(rows[0]["registers"], 74)
        self.assertEqual(rows[0]["stack_bytes"], 376)
        self.assertEqual(rows[0]["spill_stores"], 0)
        self.assertEqual(rows[0]["smem_bytes"], 0)
        self.assertEqual(rows[1]["target"], "ipm_gpu_double")
        self.assertEqual(rows[1]["registers"], 80)
        self.assertEqual(rows[1]["spill_loads"], 56)
        self.assertEqual(rows[1]["smem_bytes"], 6168)
        self.assertIn("tiled", rows[1]["function"])
        self.assertEqual(rows[2]["target"], "lib_ipm.so")
        self.assertEqual(rows[2]["registers"], 74)
        self.assertEqual(rows[2]["smem_bytes"], 6168)


import shutil  # noqa: E402
import make_reference  # noqa: E402


def fake_build_dir(build_dir):
    """A build directory whose executables are the fake."""
    build = Path(build_dir)
    build.mkdir(parents=True, exist_ok=True)
    for name in ("ipm_gpu", "ipm_gpu_double"):
        shutil.copy(FAKE, build / name)
        os.chmod(build / name, 0o755)
    return build


class MakeReferenceTests(unittest.TestCase):
    def test_creates_runs_then_skips_existing_runs(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            argv = ["--only", "nb_f", "--build", str(build), "--ref-root", f"{d}/ref",
                    "--scratch", f"{d}/scratch", "--results", f"{d}/results"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv), 0)
            for k in (1, 2, 3):
                self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nb_f" / f"run{k}"))
            timing = bench_lib.read_json(Path(d) / "results" / "timing_full.json")
            self.assertEqual(len(timing["records"]), 3)
            self.assertEqual(timing["records"][0]["config"], "nb_f")
            self.assertEqual(timing["rung"], "R0")
            # a truth-only configuration gets one run and no timing record
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv[:1] + ["nry10_d"] + argv[2:]), 0)
            self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nry10_d" / "run1"))
            self.assertFalse((Path(d) / "ref" / "nry10_d" / "run2").exists())
            self.assertEqual(len(bench_lib.read_json(Path(d) / "results" / "timing_full.json")["records"]), 3)
            # resume: remove run2, rerun, only run2 is regenerated and its record is replaced, not duplicated
            shutil.rmtree(Path(d) / "ref" / "nb_f" / "run2")
            stamp = (Path(d) / "ref" / "nb_f" / "run1" / "ipm_parameter_info.txt").stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv), 0)
            self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nb_f" / "run2"))
            self.assertEqual((Path(d) / "ref" / "nb_f" / "run1" / "ipm_parameter_info.txt").stat().st_mtime_ns, stamp)
            self.assertFalse((Path(d) / "scratch" / "nb_f" / "run2").exists())
            self.assertEqual(len(bench_lib.read_json(Path(d) / "results" / "timing_full.json")["records"]), 3)

    def test_shift_creates_shift1_and_records_and_honours_force_and_only(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            argv = ["--shift", "--build", str(build), "--ref-root", f"{d}/ref",
                    "--scratch", f"{d}/scratch", "--results", f"{d}/results"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv + ["--only", "nb_d", "hm100_d"]), 0)
            ref = Path(d) / "ref"
            self.assertTrue(make_reference.is_complete(ref / "nb_d" / "shift1"))
            self.assertTrue(make_reference.is_complete(ref / "hm100_d" / "shift1"))
            self.assertFalse((ref / "rect_d").exists())
            self.assertFalse((ref / "nb_d" / "run1").exists())      # --shift does not make the matrix
            self.assertFalse((Path(d) / "results" / "timing_full.json").exists())
            runs = bench_lib.read_json(Path(d) / "results" / "shift_runs.json")
            self.assertEqual([r["config"] for r in runs["records"]], ["nb_d", "hm100_d"])
            self.assertEqual(runs["shift"], list(configs.SHIFT))
            stamp = (ref / "nb_d" / "shift1" / "ipm_parameter_info.txt").stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(make_reference.main(argv), 0)      # all four; two exist and are skipped
            self.assertEqual(out.getvalue().count("skip "), 2)
            self.assertEqual((ref / "nb_d" / "shift1" / "ipm_parameter_info.txt").stat().st_mtime_ns, stamp)
            runs = bench_lib.read_json(Path(d) / "results" / "shift_runs.json")
            self.assertEqual(sorted(r["config"] for r in runs["records"]), ["hm100_d", "nb_d", "rect_a0_d", "rect_d"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv + ["--only", "nb_d", "--force"]), 0)
            self.assertGreater((ref / "nb_d" / "shift1" / "ipm_parameter_info.txt").stat().st_mtime_ns, stamp)
            self.assertEqual(len(bench_lib.read_json(Path(d) / "results" / "shift_runs.json")["records"]), 4)   # replaced
            self.assertFalse((Path(d) / "scratch" / "nb_d" / "shift1").exists())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(make_reference.main(argv + ["--only", "nb_f"]), 2)   # float configurations have no shift run

    def test_failed_run_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            os.environ["FAKE_IPM_FAIL"] = "1"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = make_reference.main(["--only", "nb_f", "--build", str(build), "--ref-root", f"{d}/ref",
                                              "--scratch", f"{d}/scratch", "--results", f"{d}/results"])
            finally:
                os.environ.pop("FAKE_IPM_FAIL")
            self.assertEqual(rc, 1)
            self.assertFalse((Path(d) / "ref" / "nb_f" / "run1").exists())

    def test_force_replaces_reference_and_record(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            argv = ["--only", "nb_f", "--build", str(build), "--ref-root", f"{d}/ref",
                    "--scratch", f"{d}/scratch", "--results", f"{d}/results"]
            timing_path = Path(d) / "results" / "timing_full.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv), 0)
            timing = bench_lib.read_json(timing_path)
            for r in timing["records"]:
                r["wall_s"] = -1.0                      # marks the first pass so a replacement is visible
            bench_lib.write_json(timing_path, timing)
            os.environ["FAKE_IPM_VALUE"] = "7.0"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(make_reference.main(argv + ["--force"]), 0)
            finally:
                os.environ.pop("FAKE_IPM_VALUE")
            np.testing.assert_array_equal(
                bench_lib.read_map(Path(d) / "ref" / "nb_f" / "run1" / "ipm_magnifications.bin"), np.full((3, 4), 7.0))
            records = bench_lib.read_json(timing_path)["records"]
            self.assertEqual(sorted((r["config"], r["repeat"]) for r in records), [("nb_f", 1), ("nb_f", 2), ("nb_f", 3)])
            self.assertTrue(all(r["wall_s"] > 0 for r in records))

    def test_failure_of_one_config_does_not_stop_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            (build / "ipm_gpu").write_text("#!/bin/sh\nexit 3\n")   # the float build fails
            os.chmod(build / "ipm_gpu", 0o755)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = make_reference.main(["--only", "nb_f", "nb_d", "--build", str(build), "--ref-root", f"{d}/ref",
                                          "--scratch", f"{d}/scratch", "--results", f"{d}/results"])
            self.assertEqual(rc, 1)
            self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nb_d" / "run1"))
            self.assertFalse((Path(d) / "ref" / "nb_f" / "run1").exists())
            self.assertEqual(list((Path(d) / "ref").rglob("*.partial")), [])


import regress  # noqa: E402


def synthetic_reference(ref_root, name, value, shape=(30, 20), parities=False, k=1):
    """k is a run number or a directory name such as "shift1"."""
    d = Path(ref_root) / name / (f"run{k}" if isinstance(k, int) else k)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base = value + rng.random(shape)
    write_map(d / "ipm_magnifications.bin", base.astype(np.float64))
    if parities:
        write_map(d / "ipm_magnifications_minima.bin", (0.4 * base).astype(np.float64))
        write_map(d / "ipm_magnifications_saddles.bin", (0.6 * base).astype(np.float64))
    (d / "ipm_parameter_info.txt").write_text("t_shoot_cells 1.0\n")
    return base


class RegressTests(unittest.TestCase):
    def tolerances(self):
        tol = compare_maps.default_tolerances()
        err = {"mean": 1e-6, "p999": 1e-5, "p9999": 2e-5, "max": 5e-5}
        tol["float_truth_error"] = {"nb_f": {"pixels": err}, "par_f": {"pixels": err, "minima": err, "saddles": err}}
        return tol

    def test_double_config_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            synthetic_reference(f"{d}/ref", "nb_d", 5.0, k="shift1")
            out, out_shift = Path(d) / "out", Path(d) / "out_shift"
            out.mkdir()
            out_shift.mkdir()
            write_map(out / "ipm_magnifications.bin", base * (1 + 1e-9))
            write_map(out_shift / "ipm_magnifications.bin", base * (1 + 1e-9))
            check = lambda: regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R3", out_shift)
            res = check()
            self.assertTrue(res["pass"], res["failures"])
            entry = res["maps"]["pixels"]
            self.assertIn("vs_r0_shift", entry)
            self.assertTrue(entry["verdict"]["pass"] and entry["verdict_shift"]["pass"])
            # either comparison alone fails the configuration, and the failure names the comparison
            write_map(out / "ipm_magnifications.bin", base * (1 + 1e-4))
            res = check()
            self.assertFalse(res["pass"])
            self.assertTrue(all(f.startswith("pixels vs_r0: ") for f in res["failures"]), res["failures"])
            self.assertTrue(res["maps"]["pixels"]["verdict_shift"]["pass"])
            write_map(out / "ipm_magnifications.bin", base * (1 + 1e-9))
            write_map(out_shift / "ipm_magnifications.bin", base * (1 + 1e-4))
            res = check()
            self.assertFalse(res["pass"])
            self.assertTrue(all(f.startswith("pixels vs_r0_shift: ") for f in res["failures"]), res["failures"])
            self.assertTrue(res["maps"]["pixels"]["verdict"]["pass"])

    def test_double_config_uses_the_calibrated_actual_limits_and_flags_a_large_max(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            synthetic_reference(f"{d}/ref", "nb_d", 5.0, k="shift1")
            out, out_shift = Path(d) / "out", Path(d) / "out_shift"
            out.mkdir()
            out_shift.mkdir()
            write_map(out_shift / "ipm_magnifications.bin", base)
            bumped = base.copy()
            bumped[0, 0] *= 1.002      # one pixel: mean ~3e-6, max 2e-3
            write_map(out / "ipm_magnifications.bin", bumped)
            tol = self.tolerances()
            tol["double_actual"]["nb_d"] = {**tol["double_actual"]["default"],
                                            "mean": 1e-5, "p999": 1e-3, "p9999": 1e-2, "sum_ratio": 1e-5}
            res = regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", tol, "R3", out_shift)
            self.assertTrue(res["pass"], res["failures"])
            self.assertTrue(any(f.startswith("pixels vs_r0: max=") for f in res["flags"]), res["flags"])

    def test_double_config_requires_the_shifted_output_and_reference(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base)
            with self.assertRaises(ValueError):
                regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R3")
            with self.assertRaises(FileNotFoundError) as ctx:
                regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R3", out)
            self.assertIn("nb_d/shift1", str(ctx.exception))

    def test_float_config_uses_truth(self):
        with tempfile.TemporaryDirectory() as d:
            truth = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            synthetic_reference(f"{d}/ref", "nb_f", 5.0)      # same values: R0-float == truth here
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", (truth * (1 + 1e-6)).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertTrue(res["pass"], res["failures"])
            self.assertIn("vs_truth", res["maps"]["pixels"])
            write_map(out / "ipm_magnifications.bin", (truth * (1 + 1e-4)).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertFalse(res["pass"])

    def test_parity_maps_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "par_d", 5.0, parities=True)
            synthetic_reference(f"{d}/ref", "par_f", 5.0, parities=True)
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base.astype(np.float32))
            write_map(out / "ipm_magnifications_minima.bin", (0.4 * base).astype(np.float32))
            write_map(out / "ipm_magnifications_saddles.bin", (0.6 * base).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["par_f"], out, f"{d}/ref", self.tolerances(), "R4")
            self.assertTrue(res["pass"], res["failures"])
            self.assertEqual(set(res["maps"]), {"pixels", "minima", "saddles"})
            write_map(out / "ipm_magnifications_saddles.bin", (0.7 * base).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["par_f"], out, f"{d}/ref", self.tolerances(), "R4")
            self.assertFalse(res["pass"])

    def test_missing_reference_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", np.ones((3, 3)))
            with self.assertRaises(FileNotFoundError) as ctx:
                regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R1a", out)
            self.assertIn("nb_d/run1", str(ctx.exception))

    def write_fake_regression(self, d):
        """Fake build, 3x4 ladder references and a tolerances file under d; returns the build directory."""
        build = fake_build_dir(Path(d) / "build")
        # fake writes float32 maps of constant 2.0; references of the same value pass
        for name in ("nb_d", "nb_f", "rect_d", "rect_f", "hm100_d", "hm100_f"):
            ref = Path(d) / "ref" / name / "run1"
            ref.mkdir(parents=True)
            write_map(ref / "ipm_magnifications.bin", np.full((3, 4), 2.0, dtype=np.float32))
            (ref / "ipm_parameter_info.txt").write_text("t_shoot_cells 1.0\n")
            if name.endswith("_d"):
                shutil.copytree(ref, ref.with_name("shift1"))
        tol = self.tolerances()
        for name in ("nb_f", "rect_f", "hm100_f"):
            tol["float_truth_error"][name] = {"pixels": {"mean": 1e-6, "p999": 1e-5, "p9999": 2e-5, "max": 5e-5}}
        bench_lib.write_json(Path(d) / "tolerances.json", tol)
        return build

    def run_fake_regression(self, d, build, env):
        """regress.main over the fixture under d with the given environment switches; returns the exit code."""
        argv = ["--build", str(build), "--rung", "R1a", "--ref-root", f"{d}/ref",
                "--tolerances", f"{d}/tolerances.json", "--results", f"{d}/results", "--scratch", f"{d}/scratch"]
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            return regress.main(argv)

    def test_main_with_fake_executable(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            self.assertEqual(self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"}), 0)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            self.assertTrue(out["pass"])
            self.assertTrue(out["complete"])
            self.assertEqual(out["n_configs"], 4)
            self.assertEqual([c["config"] for c in out["configs"]], ["nb_f", "nb_d", "rect_f", "hm100_f"])
            self.assertEqual([("run_shift" in c) for c in out["configs"]], [False, True, False, False])
            nb_d = out["configs"][1]
            self.assertEqual(set(nb_d["run_shift"]), set(nb_d["run"]))
            self.assertIn("vs_r0_shift", nb_d["maps"]["pixels"])
            self.assertIn("double_actual", out["tolerances"])
            self.assertFalse((Path(d) / "scratch" / "nb_d_shift").exists())
            self.assertFalse((Path(d) / "scratch" / "nb_f").exists())

    def test_failed_shift_run_fails_the_double_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            real = bench_lib.run_ipm

            def fail_shift(exe, args, outdir, **kw):
                if "-cy1" in args:
                    exe = "/nonexistent/ipm"
                return real(exe, args, outdir, **kw)
            with mock.patch.object(bench_lib, "run_ipm", side_effect=fail_shift):
                rc = self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"})
            self.assertEqual(rc, 1)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            by_name = {c["config"]: c for c in out["configs"]}
            self.assertTrue(by_name["nb_f"]["pass"])
            self.assertFalse(by_name["nb_d"]["pass"])
            self.assertTrue(any(f.startswith("shift run failed") for f in by_name["nb_d"]["failures"]), by_name["nb_d"]["failures"])
            self.assertEqual(list((Path(d) / "scratch").glob("*")), [])

    def test_check_failure_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            # 5x5 maps against the 3x4 references: every check raises a shape mismatch, and every configuration still runs
            rc = self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0", "FAKE_IPM_SHAPE": "5x5"})
            self.assertEqual(rc, 1)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            self.assertTrue(out["complete"])
            self.assertFalse(out["pass"])
            self.assertEqual(len(out["configs"]), 4)
            for c in out["configs"]:
                self.assertFalse(c["pass"])
                self.assertTrue(any(f.startswith("check failed: ValueError: shape mismatch") for f in c["failures"]),
                                c["failures"])

    def test_empty_selection_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = regress.main(["--build", f"{d}/build", "--rung", "R1a", "--only", "does_not_exist",
                                   "--ref-root", f"{d}/ref", "--tolerances", f"{d}/tolerances.json",
                                   "--results", f"{d}/results", "--scratch", f"{d}/scratch"])
            self.assertEqual(rc, 2)
            self.assertIn("no configurations selected", err.getvalue())
            self.assertFalse((Path(d) / "results").exists())

    def test_earlier_result_file_does_not_survive_an_abort(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            result = Path(d) / "results" / "regression_ladder.json"
            bench_lib.write_json(result, {"complete": True, "pass": True, "configs": []})
            # KeyboardInterrupt stands in for a killed job: it is not an Exception, so the per-configuration handler does not see it
            with mock.patch.object(bench_lib, "run_ipm", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"})
            out = bench_lib.read_json(result)
            self.assertFalse(out["complete"])
            self.assertEqual(out["configs"], [])
            # a later complete run replaces the file
            self.assertEqual(self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"}), 0)
            out = bench_lib.read_json(result)
            self.assertTrue(out["complete"])
            self.assertEqual(out["n_configs"], 4)

    def test_runs_have_a_timeout_and_failed_runs_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            with mock.patch.object(bench_lib, "run_ipm", wraps=bench_lib.run_ipm) as runs:
                rc = self.run_fake_regression(d, build, {"FAKE_IPM_FAIL": "1"})
            self.assertEqual(rc, 1)
            self.assertEqual(runs.call_count, 5)   # the double configuration runs twice
            for call in runs.call_args_list:
                self.assertEqual(call.kwargs.get("timeout"), 3600)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            self.assertTrue(out["complete"])
            self.assertEqual(len(out["configs"]), 4)
            for c in out["configs"]:
                self.assertFalse(c["pass"])
                self.assertTrue(any("run failed: exit code 3" in f for f in c["failures"]), c["failures"])
            self.assertEqual(list((Path(d) / "scratch").glob("*")), [])

    def test_float_rule_is_judged_against_the_truth(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            # R0-float sits 2e-6 from the truth. The candidate is R0-float itself, so only the truth comparison can fail it.
            ref_f = Path(d) / "ref" / "nb_f" / "run1"
            ref_f.mkdir(parents=True)
            write_map(ref_f / "ipm_magnifications.bin", base * (1 + 2e-6))
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base * (1 + 2e-6))
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertFalse(res["pass"])
            self.assertTrue(res["failures"] and all("vs truth" in f for f in res["failures"]), res["failures"])

    def test_sum_ratio_is_judged_against_the_truth(self):
        with tempfile.TemporaryDirectory() as d:
            truth = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            ref_f = Path(d) / "ref" / "nb_f" / "run1"
            ref_f.mkdir(parents=True)
            write_map(ref_f / "ipm_magnifications.bin", truth * (1 - 1e-4))   # R0-float deposits 1e-4 too little
            out = Path(d) / "out"
            out.mkdir()
            tol = self.tolerances()
            tol["float_truth_error"]["nb_f"]["pixels"]["sum_ratio"] = -1e-4
            check = lambda: regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", tol, "R2")
            write_map(out / "ipm_magnifications.bin", truth)   # conserves the area: far from R0-float, equal to the truth
            res = check()
            self.assertTrue(res["pass"], res["failures"])
            self.assertAlmostEqual(res["maps"]["pixels"]["vs_r0"]["sum_ratio"], 1e-4, places=6)   # information only
            write_map(out / "ipm_magnifications.bin", truth * (1 - 1.6e-4))   # truth error above 1.5 x 1e-4
            res = check()
            self.assertFalse(res["pass"])
            self.assertTrue(any("|sum_ratio| vs truth" in f for f in res["failures"]), res["failures"])

    def test_parity_consistency_fails_alone(self):
        with tempfile.TemporaryDirectory() as d:
            base = 5.0 + np.random.default_rng(3).random((30, 20))
            # self-inconsistent maps: minima + saddles = 0.9 x total, so only the parity check can fail
            maps = {"ipm_magnifications.bin": base,
                    "ipm_magnifications_minima.bin": 0.4 * base,
                    "ipm_magnifications_saddles.bin": 0.5 * base}
            for name in ("par_d", "par_f"):
                ref = Path(d) / "ref" / name / "run1"
                ref.mkdir(parents=True)
                for fname, arr in maps.items():
                    write_map(ref / fname, arr)
            out = Path(d) / "out"
            out.mkdir()
            for fname, arr in maps.items():
                write_map(out / fname, arr)
            res = regress.check_config(configs.CONFIGS["par_f"], out, f"{d}/ref", self.tolerances(), "R4")
            self.assertFalse(res["pass"])
            self.assertTrue(res["failures"] and all(f.startswith("pixels != minima + saddles") for f in res["failures"]),
                            res["failures"])
            self.assertTrue(all(entry["verdict"]["pass"] for entry in res["maps"].values()))

    def test_star_file_source_is_information_only(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "sf_d", 5.0)
            synthetic_reference(f"{d}/ref", "sf_f", 5.0)
            ref_nb = Path(d) / "ref" / "nb_f" / "run1"
            ref_nb.mkdir(parents=True)
            write_map(ref_nb / "ipm_magnifications.bin", 1.5 * base)   # the source field is 1.5x the candidate
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base)
            tol = self.tolerances()
            tol["float_truth_error"]["sf_f"] = {"pixels": {"mean": 1e-6, "p999": 1e-5, "p9999": 2e-5, "max": 5e-5}}
            res = regress.check_config(configs.CONFIGS["sf_f"], out, f"{d}/ref", tol, "R2")
            self.assertTrue(res["pass"], res["failures"])
            self.assertAlmostEqual(res["vs_source_field"]["mean"], 1 / 3, places=6)

    def test_unknown_configuration_name_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = regress.main(["--build", f"{d}/build", "--rung", "R1a", "--only", "nb_f", "nb_ff",
                                   "--ref-root", f"{d}/ref", "--tolerances", f"{d}/tolerances.json",
                                   "--results", f"{d}/results", "--scratch", f"{d}/scratch"])
            self.assertEqual(rc, 2)
            self.assertIn("unknown configuration(s): nb_ff", err.getvalue())
            self.assertFalse((Path(d) / "results").exists())


import calibrate  # noqa: E402

# Synthetic references for calibrate.py. Every relative difference is amplitude * SPREAD. The outlier gives the spread a
# heavy tail, so the mean, p99.9, p99.99 and max of a comparison differ in their first three digits (what the summary
# lines print) and are known in closed form.
SPREAD = np.append(np.linspace(0.5, 1.5, 599), 20.0).reshape(30, 20)
FLOAT_AMPLITUDE = {"nb_f": 4e-6, "rect_f": 5e-6, "hm100_f": 6e-6, "circ_a0_f": 7e-6, "rect_a0_f": 8e-6,
                   "nry10_f": 9e-6, "big_f": 1.0e-5, "hm1000_f": 1.1e-5, "sf_f": 1.2e-5, "stars1e6_f": 1.3e-5}
PARITY_AMPLITUDE = {"minima": 2e-6, "saddles": 3e-6}   # par_f; its pixels are minima + saddles


def spread_stats(amplitude):
    return {"mean": amplitude * SPREAD.mean(), "p999": amplitude * np.percentile(SPREAD, 99.9),
            "p9999": amplitude * np.percentile(SPREAD, 99.99), "max": amplitude * SPREAD.max()}


def map_path(ref, name, k, which):
    return Path(ref) / name / (f"run{k}" if isinstance(k, int) else k) / regress.MAP_FILES[which]


SHIFT_AMPLITUDE = 4e-10   # shift1 = run1 times (1 + SHIFT_AMPLITUDE * SPREAD) for the double regression configurations


def calibration_tree(ref, shift=True):
    """Write synthetic references for all 22 configurations; returns {(float configuration, map): amplitude}.

    A double configuration has a map level of its own. A float configuration is its double twin times
    (1 + amplitude * SPREAD), with a different amplitude for every configuration and map, and the pixels of a parity
    configuration are minima + saddles. The configurations with three runs get run2 and run3 as run1 times
    (1 + 2e-9 * SPREAD) and (1 + 3e-9 * SPREAD). With shift, the four double regression configurations also get a
    shift1 directory (see SHIFT_AMPLITUDE).
    """
    texture = np.random.default_rng(7).random(SPREAD.shape)
    maps = {}
    for i, cfg in enumerate(c for c in configs.CONFIGS.values() if c.build == "double"):
        pixels = 5.0 + 0.5 * i + texture
        maps[cfg.name] = {"pixels": pixels}
        if cfg.parities:
            maps[cfg.name].update(minima=0.4 * pixels, saddles=0.6 * pixels)
    amplitude = {}
    for cfg in configs.CONFIGS.values():
        if cfg.build == "float":
            twin = maps[cfg.truth]
            if cfg.parities:
                run = {which: twin[which] * (1 + a * SPREAD) for which, a in PARITY_AMPLITUDE.items()}
                run["pixels"] = run["minima"] + run["saddles"]
                amplitude.update({(cfg.name, which): a for which, a in PARITY_AMPLITUDE.items()})
                amplitude[(cfg.name, "pixels")] = 0.4 * PARITY_AMPLITUDE["minima"] + 0.6 * PARITY_AMPLITUDE["saddles"]
            else:
                run = {"pixels": twin["pixels"] * (1 + FLOAT_AMPLITUDE[cfg.name] * SPREAD)}
                amplitude[(cfg.name, "pixels")] = FLOAT_AMPLITUDE[cfg.name]
            maps[cfg.name] = run
    for name, run1 in maps.items():
        runs = [run1]
        if "full" in configs.CONFIGS[name].sets:
            runs += [{which: m * (1 + eps * SPREAD) for which, m in run1.items()} for eps in (2e-9, 3e-9)]
        for k, run in enumerate(runs, start=1):
            for which, arr in run.items():
                map_path(ref, name, k, which).parent.mkdir(parents=True, exist_ok=True)
                write_map(map_path(ref, name, k, which), arr)
    if shift:
        for cfg in configs.double_regression_configs():
            path = map_path(ref, cfg.name, "shift1", "pixels")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_map(path, maps[cfg.name]["pixels"] * (1 + SHIFT_AMPLITUDE * SPREAD))
    return amplitude


def perturb(ref, name, k, which, factor):
    path = map_path(ref, name, k, which)
    write_map(path, bench_lib.read_map(path) * factor)


def run_quietly(func, *args):
    """Call func with its printed output captured; returns (result, printed text)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        result = func(*args)
    return result, printed.getvalue()


def numbers_in(line):
    return [float(x) for x in re.findall(r"\d\.\d\de[+-]\d\d", line)]


class CalibrateTests(unittest.TestCase):
    def test_floors_and_truth_errors(self):
        with tempfile.TemporaryDirectory() as d:
            ref = f"{d}/ref"
            for name in ("nb_f", "nb_d", "rect_f", "rect_d", "hm100_f", "hm100_d", "circ_a0_f", "circ_a0_d",
                         "rect_a0_f", "rect_a0_d", "par_f", "par_d", "nry10_f", "nry10_d", "big_f", "big_d",
                         "hm1000_f", "hm1000_d", "sf_f", "sf_d", "stars1e6_f", "stars1e6_d"):
                parities = name.startswith("par_")
                base = synthetic_reference(ref, name, 5.0, parities=parities, k=1)
                if "full" in configs.CONFIGS[name].sets:
                    for k in (2, 3):
                        dd = Path(ref) / name / f"run{k}"
                        dd.mkdir(parents=True)
                        noisy = base * (1 + 1e-9 * k)
                        write_map(dd / "ipm_magnifications.bin", noisy)
                        if parities:
                            write_map(dd / "ipm_magnifications_minima.bin", 0.4 * noisy)
                            write_map(dd / "ipm_magnifications_saddles.bin", 0.6 * noisy)
            with contextlib.redirect_stdout(io.StringIO()):   # calibrate() prints one progress line per map
                tol, lines, warnings = calibrate.calibrate(ref, compare_maps.default_tolerances())
            self.assertAlmostEqual(tol["floors"]["nb_d"]["pixels"][1]["max"], 3e-9, places=12)
            self.assertIn("nb_f", tol["float_truth_error"])
            self.assertEqual(set(tol["float_truth_error"]["par_f"]), {"pixels", "minima", "saddles"})
            self.assertEqual(len(warnings), 1)   # no shift1 references in this tree
            self.assertIn("no shift1 references", warnings[0])
            self.assertTrue(any(line.startswith("nb_f") for line in lines))

    def test_shift_sensitivity_and_derived_double_actual(self):
        with tempfile.TemporaryDirectory() as d:
            calibration_tree(f"{d}/ref")
            # a different shift effect per configuration; the largest drives double_actual
            perturb(f"{d}/ref", "hm100_d", "shift1", "pixels", 1 + 3e-5)
            (tol, lines, warnings), printed = run_quietly(calibrate.calibrate, f"{d}/ref", compare_maps.default_tolerances())
        self.assertEqual(list(tol["double_shift_sensitivity"]), [c.name for c in configs.double_regression_configs()])
        sens = tol["double_shift_sensitivity"]
        self.assertAlmostEqual(sens["nb_d"]["max"], SHIFT_AMPLITUDE * 20.0, places=12)
        self.assertAlmostEqual(sens["hm100_d"]["mean"], 3e-5, delta=1e-7)
        act, dbl = tol["double_actual"], tol["double"]
        self.assertEqual(list(act), [c.name for c in configs.double_regression_configs()] + ["default"])
        # nb_d's sensitivity is far below the signed-off thresholds: they are the floor; only flag_max has its own floor
        self.assertEqual(act["nb_d"], {**dbl, "flag_max": 1e-3})
        # hm100_d: 2 x ~3.0e-5 rounded up to one significant figure
        self.assertEqual(act["hm100_d"]["mean"], 7e-5)
        for key in ("mean", "p999", "p9999"):
            self.assertEqual(act["hm100_d"][key], calibrate.round_up_1sf(max(2 * sens["hm100_d"][key], dbl[key])), key)
        self.assertEqual(act["hm100_d"]["flag_max"], calibrate.round_up_1sf(max(3 * sens["hm100_d"]["max"], 1e-3)))
        self.assertEqual((act["hm100_d"]["offmask_max"], act["hm100_d"]["sum_ratio"]), (dbl["offmask_max"], dbl["sum_ratio"]))
        self.assertEqual(act["default"], act["hm100_d"])      # hm100_d has the largest sensitivity everywhere
        self.assertLess(act["nb_d"]["mean"], act["default"]["mean"])
        self.assertEqual(tol["double_actual_note"], "calibrated from R0 shift sensitivity; pending user sign-off")
        self.assertEqual(tol["double"], compare_maps.default_tolerances()["double"])   # the signed-off thresholds are untouched
        self.assertEqual(sum("shift sensitivity" in line for line in lines), 4)
        self.assertEqual(sum(line.startswith("double_actual") for line in lines), 5)   # four configurations and the default
        # the signed-off mean (1e-7) is below the sensitivity of hm100_d (3e-5): a warning for each of the three keys
        self.assertEqual(sum("is below R0's own shift sensitivity" in w for w in warnings), 3, warnings)
        self.assertTrue(all(("WARNING " + w) in printed.splitlines() for w in warnings))

    def test_warns_per_missing_shift_reference_when_only_some_exist(self):
        with tempfile.TemporaryDirectory() as d:
            calibration_tree(f"{d}/ref")
            shutil.rmtree(Path(d) / "ref" / "rect_d" / "shift1")
            (tol, _, warnings), _ = run_quietly(calibrate.calibrate, f"{d}/ref", compare_maps.default_tolerances())
        self.assertEqual(sorted(tol["double_shift_sensitivity"]), ["hm100_d", "nb_d", "rect_a0_d"])
        missing = [w for w in warnings if "no shift1 reference" in w]
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].startswith("rect_d:"))

    def test_round_up_to_one_significant_figure(self):
        for x, want in ((1.76e-6, 2e-6), (2e-6, 2e-6), (2.0000001e-6, 3e-6), (0.91, 1.0), (1.1e-3, 2e-3), (0.0, 0.0)):
            self.assertEqual(calibrate.round_up_1sf(x), want, x)

    def test_without_shift_references_double_actual_keeps_its_defaults_and_warns(self):
        with tempfile.TemporaryDirectory() as d:
            calibration_tree(f"{d}/ref", shift=False)
            (tol, lines, warnings), printed = run_quietly(calibrate.calibrate, f"{d}/ref", compare_maps.default_tolerances())
        self.assertEqual(tol["double_actual"], compare_maps.default_tolerances()["double_actual"])
        self.assertEqual(tol["double_shift_sensitivity"], {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("WARNING " + warnings[0], printed.splitlines())
        self.assertFalse(any("shift sensitivity" in line for line in lines))

    def test_warns_when_double_floor_is_close_to_threshold(self):
        tol = compare_maps.default_tolerances()
        tol["floors"] = {"nb_d": {"pixels": [{"mean": 0, "p999": 2e-6, "p9999": 0, "offmask_max": 0}]}}
        self.assertEqual(len(calibrate.sanity_warnings(tol)), 1)
        self.assertIn("nb_d", calibrate.sanity_warnings(tol)[0])

    def test_stored_floors_and_truth_errors_follow_the_references(self):
        with tempfile.TemporaryDirectory() as d:
            amplitude = calibration_tree(f"{d}/ref")
            (tol, _, warnings), _ = run_quietly(calibrate.calibrate, f"{d}/ref", compare_maps.default_tolerances())
        full = {cfg.name: cfg for cfg in configs.configs_in("full")}
        self.assertEqual(set(tol["floors"]), set(full))
        for name, cfg in full.items():
            maps = ["pixels", "minima", "saddles"] if cfg.parities else ["pixels"]
            self.assertEqual(set(tol["floors"][name]), set(maps))
            for which in maps:
                run2, run3 = tol["floors"][name][which]   # exactly two entries: run2 vs run1, then run3 vs run1
                for stored, eps in ((run2, 2e-9), (run3, 3e-9)):
                    for key, want in spread_stats(eps).items():
                        self.assertAlmostEqual(stored[key], want, places=12, msg=f"{name} {which} {key}")
        floats = {name for name, cfg in full.items() if cfg.build == "float"}
        self.assertEqual(set(tol["float_truth_error"]), floats)   # every float configuration and no double one
        self.assertEqual({name for name, _ in amplitude}, floats)
        for (name, which), amp in amplitude.items():
            self.assertEqual(set(tol["float_truth_error"][name]), {"pixels", "minima", "saddles"} if name == "par_f" else {"pixels"})
            for key, want in spread_stats(amp).items():   # against its own double twin, in this order of arguments
                self.assertAlmostEqual(tol["float_truth_error"][name][which][key], want, places=12, msg=f"{name} {which} {key}")
        self.assertEqual(warnings, [])

    def test_summary_lines_show_the_stored_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            calibration_tree(f"{d}/ref")
            (tol, lines, _), printed = run_quietly(calibrate.calibrate, f"{d}/ref", compare_maps.default_tolerances())
        self.assertEqual(printed.splitlines(), lines)
        extra = [line for line in lines if "shift sensitivity" in line or line.startswith("double_actual")]
        for line in extra[:4]:   # one line per double regression configuration, then one per double_actual entry
            name = line.split()[0]
            sens = tol["double_shift_sensitivity"][name]
            self.assertEqual(numbers_in(line), [float(f"{sens[k]:.2e}") for k in ("mean", "p999", "p9999", "max")], line)
        self.assertEqual(len(extra), 9)
        for line, (name, actual) in zip(extra[4:], tol["double_actual"].items()):
            self.assertEqual(line.split()[:2], ["double_actual", name])
            self.assertEqual(numbers_in(line), [float(f"{v:.2e}") for v in actual.values()], line)
        lines = [line for line in lines if line not in extra]
        shown = [tuple(line.split()[:2]) for line in lines]
        stored = [(name, which) for name, per_map in tol["floors"].items() for which in per_map] + [("par_f", "parity")]
        self.assertEqual(sorted(shown), sorted(stored))   # every map once, plus one parity line
        for line in lines:
            name, which = line.split()[:2]
            if which == "parity":
                want = tol["parity_floor"][name]
            else:
                floors = tol["floors"][name][which]
                want = [max(f[key] for f in floors) for key in ("mean", "p999", "max")]
                if name in tol["float_truth_error"]:
                    want += [tol["float_truth_error"][name][which][key] for key in ("mean", "p999", "p9999", "max")]
            self.assertEqual(numbers_in(line), [float(f"{v:.2e}") for v in want], line)

    def test_floor_fraction_comes_from_the_tolerances(self):
        with tempfile.TemporaryDirectory() as d:
            calibration_tree(f"{d}/ref")
            tol = compare_maps.default_tolerances()
            tol["floor_frac"] = 1.0   # only the pixels at or above the mean are compared
            (tol, _, _), _ = run_quietly(calibrate.calibrate, f"{d}/ref", tol)
        stored = [m for per_map in tol["floors"].values() for floors in per_map.values() for m in floors]
        stored += [m for per_map in tol["float_truth_error"].values() for m in per_map.values()]
        self.assertEqual(len(stored), 2 * 17 + 13)   # two floors for each of 17 maps, a truth error for each of 13 float maps
        for m in stored:
            self.assertTrue(0 < m["n_mask"] < m["n_total"], m)

    def test_double_floor_rule_per_key(self):
        def floor(key=None, value=0.0):
            return {"mean": 0.0, "p999": 0.0, "p9999": 0.0, "offmask_max": 0.0, **({key: value} if key else {})}

        for key in ("mean", "p999", "p9999", "offmask_max"):
            with self.subTest(key=key):
                tol = compare_maps.default_tolerances()
                tenth = tol["double"][key] / 10
                for floors in ([floor(), floor(key, 2 * tenth)], [floor(key, 2 * tenth), floor()]):   # the worse run decides
                    tol["floors"] = {"nb_d": {"pixels": floors}}
                    found = calibrate.sanity_warnings(tol)
                    self.assertEqual(len(found), 1, found)
                    self.assertIn("nb_d pixels", found[0])
                    self.assertIn(f" {key}={2 * tenth:.2e} ", found[0])   # the floor ...
                    self.assertIn(f"threshold {tol['double'][key]:.1e}", found[0])   # ... and the threshold it is compared with
                for value, count in ((0.99 * tenth, 0), (1.01 * tenth, 1)):   # just below and above threshold / 10
                    tol["floors"] = {"nb_d": {"pixels": [floor(key, value), floor()]}}
                    self.assertEqual(len(calibrate.sanity_warnings(tol)), count, value)
                tol["double"][key] = 1.0
                tol["floors"] = {"nb_d": {"pixels": [floor(key, 0.1), floor()]}}   # exactly 10x is still allowed
                self.assertEqual(calibrate.sanity_warnings(tol), [])

    def test_float_floors_are_not_checked_and_every_double_map_is(self):
        big = {"mean": 1.0, "p999": 1.0, "p9999": 1.0, "offmask_max": 1.0}
        tol = compare_maps.default_tolerances()
        tol["floors"] = {"nb_f": {"pixels": [big, big]}}
        self.assertEqual(calibrate.sanity_warnings(tol), [])   # float noise is far above the double thresholds
        for order in (("nb_f", "nb_d"), ("nb_d", "nb_f")):   # a float configuration does not hide a double one
            tol["floors"] = {name: {"pixels": [big, big]} for name in order}
            found = calibrate.sanity_warnings(tol)
            self.assertEqual(len(found), 4, found)   # one per key
            self.assertTrue(all(w.startswith("nb_d pixels") for w in found), found)
        ok = {"mean": 0.0, "p999": 0.0, "p9999": 0.0, "offmask_max": 0.0}
        tol["floors"] = {"par_d": {"pixels": [ok, ok], "minima": [{**ok, "mean": 1.0}, ok], "saddles": [ok, {**ok, "mean": 1.0}]}}
        found = calibrate.sanity_warnings(tol)
        self.assertEqual(sorted(w.split(":")[0] for w in found), ["par_d minima", "par_d saddles"])

    def test_main_round_trip_with_a_regress_check(self):
        with tempfile.TemporaryDirectory() as d:
            ref, out = f"{d}/ref", f"{d}/results/tolerances.json"
            calibration_tree(ref)
            rc, printed = run_quietly(calibrate.main, ["--ref-root", ref, "--out", out])
            self.assertEqual(rc, 0)
            (measured, lines, _), _ = run_quietly(calibrate.calibrate, ref, compare_maps.default_tolerances())
            tol = compare_maps.load_tolerances(out)
            self.assertEqual(tol, json.loads(json.dumps(measured)))   # the file holds what calibrate() measures
            self.assertEqual(tol["double"], compare_maps.default_tolerances()["double"])   # no earlier file: the defaults
            self.assertEqual(tol["warnings"], [])
            self.assertEqual(set(tol["parity_floor"]), {"par_f"})
            summary = Path(out).with_name("tolerances_summary.md")
            self.assertEqual(summary.read_text(), "```\n" + "\n".join(lines) + "\n```\n")
            self.assertEqual(printed.splitlines(), lines + [f"wrote {out} and {summary}"])
            cand = Path(d) / "cand"   # a candidate equal to the references passes with that file, one that is off does not
            cand.mkdir()
            for fname in regress.MAP_FILES.values():
                shutil.copy(Path(ref) / "par_f" / "run1" / fname, cand / fname)
            res = regress.check_config(configs.CONFIGS["par_f"], cand, ref, tol, "R3")
            self.assertTrue(res["pass"], res["failures"])
            saddles = cand / regress.MAP_FILES["saddles"]
            write_map(saddles, bench_lib.read_map(saddles) * 1.001)
            self.assertFalse(regress.check_config(configs.CONFIGS["par_f"], cand, ref, tol, "R3")["pass"])

    def test_main_defaults_to_the_bench_root(self):
        with tempfile.TemporaryDirectory() as d:
            calibration_tree(f"{d}/ref")
            with mock.patch.object(bench_lib, "BENCH_ROOT", Path(d)):
                rc, _ = run_quietly(calibrate.main, [])
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(p.name for p in (Path(d) / "results").iterdir()), ["tolerances.json", "tolerances_summary.md"])

    def test_main_without_a_bench_root_asks_for_one(self):
        with mock.patch.object(bench_lib, "BENCH_ROOT", None):
            with self.assertRaises(SystemExit) as ctx:
                run_quietly(calibrate.main, [])
        self.assertIn("IPM_BENCH_ROOT is not set", str(ctx.exception))

    def test_parity_floor_has_a_slot_for_each_run_and_the_double_twin(self):
        with tempfile.TemporaryDirectory() as d:
            ref = f"{d}/ref"
            calibration_tree(ref)
            (tol, _, warnings), _ = run_quietly(calibrate.calibrate, ref, compare_maps.default_tolerances())
            self.assertEqual(set(tol["parity_floor"]), {"par_f"})
            self.assertEqual(len(tol["parity_floor"]["par_f"]), 4)
            self.assertLess(max(tol["parity_floor"]["par_f"]), 1e-12)   # consistent references
            self.assertEqual(warnings, [])
            # a different inconsistency in run1, run2 and run3 of par_f and in run1 of par_d lands in its own slot
            want = []
            for i, (name, k) in enumerate([("par_f", 1), ("par_f", 2), ("par_f", 3), ("par_d", 1)], start=1):
                eps = i * 1e-4
                pixels, saddles = (bench_lib.read_map(map_path(ref, name, k, which)) for which in ("pixels", "saddles"))
                want.append(float((saddles * eps).max() / pixels.mean()))
                perturb(ref, name, k, "saddles", 1 + eps)
            (tol, lines, warnings), _ = run_quietly(calibrate.calibrate, ref, compare_maps.default_tolerances())
            got = tol["parity_floor"]["par_f"]
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, delta=1e-9)
            (line,) = [line for line in lines if line.startswith("par_f") and "parity" in line]
            self.assertEqual(numbers_in(line), [float(f"{v:.2e}") for v in got])
            self.assertIn("par_d", line)
            self.assertEqual(len(warnings), 1, warnings)

    def test_one_inconsistent_run_is_enough_for_a_parity_warning(self):
        for name, k in (("par_f", 1), ("par_f", 2), ("par_f", 3), ("par_d", 1)):
            with self.subTest(run=f"{name} run{k}"), tempfile.TemporaryDirectory() as d:
                calibration_tree(f"{d}/ref")
                perturb(f"{d}/ref", name, k, "saddles", 1 + 1e-3)
                (tol, _, warnings), printed = run_quietly(calibrate.calibrate, f"{d}/ref", compare_maps.default_tolerances())
                self.assertEqual(len(warnings), 1, warnings)
                self.assertTrue(warnings[0].startswith("par_f parity"), warnings)
                self.assertEqual(tol["warnings"], warnings)
                self.assertIn("WARNING " + warnings[0], printed.splitlines())

    def test_parity_floor_warning_rule(self):
        for limit in (1e-5, 1e-3):   # the default threshold and an edited one
            tol = compare_maps.default_tolerances()
            tol["parity_consistency"] = limit
            below, above = 0.99 * limit / 10, 1.01 * limit / 10
            for values, count in (([below] * 4, 0), ([above, 0, 0, 0], 1), ([0, above, 0, 0], 1),
                                  ([0, 0, above, 0], 1), ([0, 0, 0, above], 1)):
                tol["parity_floor"] = {"par_f": values}
                found = calibrate.sanity_warnings(tol)
                self.assertEqual(len(found), count, (limit, values))
                if count:
                    self.assertTrue(found[0].startswith("par_f parity"), found)
                    self.assertIn(f"floor {above:.2e} ", found[0])   # the worst value ...
                    self.assertIn(f"threshold {limit:.1e}", found[0])   # ... against the threshold
        tol["parity_consistency"] = 1.0
        tol["parity_floor"] = {"par_f": [0.1]}   # exactly 10x is still allowed
        self.assertEqual(calibrate.sanity_warnings(tol), [])

    def test_rerun_keeps_edited_thresholds_and_replaces_the_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            ref, out = f"{d}/ref", Path(d) / "results" / "tolerances.json"
            calibration_tree(ref)
            bench_lib.write_json(out, {"double": {"p999": 2e-5, "mean": 1e-9}, "float": {"ratio": 1.2},
                                       "floors": {"stale": {}}, "float_truth_error": {"stale": {}},
                                       "parity_floor": {"stale": [1.0]}, "warnings": ["stale"]})
            rc, printed = run_quietly(calibrate.main, ["--ref-root", ref, "--out", str(out)])
            self.assertEqual(rc, 0)
            tol = json.loads(out.read_text())
            default = compare_maps.default_tolerances()
            self.assertEqual(tol["double"], {**default["double"], "p999": 2e-5, "mean": 1e-9})
            self.assertEqual(tol["float"], {**default["float"], "ratio": 1.2})
            for key in ("floors", "float_truth_error", "parity_floor"):
                self.assertNotIn("stale", tol[key])
            self.assertEqual(len(tol["floors"]), 15)
            # the edited threshold is the one the sanity check uses: the four double configurations are now too close
            self.assertEqual(sorted(w.split(" pixels")[0] for w in tol["warnings"]), ["hm100_d", "nb_d", "rect_a0_d", "rect_d"])
            self.assertTrue(all(" mean=" in w for w in tol["warnings"]), tol["warnings"])
            body = out.with_name("tolerances_summary.md").read_text().splitlines()[1:-1]   # inside the fences
            n = len(tol["warnings"])
            self.assertEqual(body[-n:], tol["warnings"])   # the warnings follow the measurements
            self.assertFalse(set(body[:-n]) & set(tol["warnings"]))
            for w in tol["warnings"]:
                self.assertIn("WARNING " + w, printed.splitlines())


import time_rung  # noqa: E402
import time_ladder  # noqa: E402


class TimingTests(unittest.TestCase):
    def test_summarize_medians(self):
        recs = [{"config": "nb_f", "ok": True, "wall_s": 3.0, "t_shoot_cells": 2.0},
                {"config": "nb_f", "ok": True, "wall_s": 1.0, "t_shoot_cells": 0.5},
                {"config": "nb_f", "ok": True, "wall_s": 2.0, "t_shoot_cells": 1.0},
                {"config": "nb_d", "ok": False, "wall_s": 9.0, "t_shoot_cells": None}]
        s = time_rung.summarize(recs)
        self.assertEqual(s["nb_f"], {"median_wall_s": 2.0, "median_t_shoot_cells": 1.0, "n": 3})
        self.assertNotIn("nb_d", s)

    def test_time_rung_main_with_fake(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = time_rung.main(["--build", str(build), "--rung", "R1a", "--repeats", "2", "--ref-root", f"{d}/ref",
                                     "--results", f"{d}/results", "--scratch", f"{d}/scratch", "--no-warmup"])
            self.assertEqual(rc, 0)
            out = bench_lib.read_json(Path(d) / "results" / "timing_ladder.json")
            self.assertEqual(out["repeats"], 2)
            self.assertEqual(len(out["records"]), 8)
            self.assertEqual(out["summary"]["hm100_f"]["n"], 2)
            self.assertEqual(out["summary"]["nb_d"]["median_t_shoot_cells"], 0.123)
            self.assertFalse((Path(d) / "scratch" / "nb_f").exists())
            # existing results are not overwritten without --force
            with contextlib.redirect_stdout(io.StringIO()):
                rc = time_rung.main(["--build", str(build), "--rung", "R1a", "--ref-root", f"{d}/ref",
                                     "--results", f"{d}/results", "--scratch", f"{d}/scratch", "--no-warmup"])
            self.assertEqual(rc, 0)
            self.assertEqual(bench_lib.read_json(Path(d) / "results" / "timing_ladder.json")["repeats"], 2)

    def test_time_ladder_interleaves_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            builds = Path(d) / "builds"
            for rung in ("R0", "R2p", "R5"):
                fake_build_dir(builds / rung)
            out_path = Path(d) / "ladder.json"
            argv = ["--rungs", "R0", "R2p", "R5", "--builds-root", str(builds), "--repeats", "2",
                    "--ref-root", f"{d}/ref", "--out", str(out_path), "--scratch", f"{d}/scratch", "--no-warmup"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(time_ladder.main(argv), 0)
            out = bench_lib.read_json(out_path)
            order = [(r["repeat"], r["rung"], r["config"]) for r in out["records"]]
            self.assertEqual(order[:5], [(1, "R0", "nb_f"), (1, "R0", "nb_d"), (1, "R0", "rect_f"), (1, "R0", "hm100_f"), (1, "R2p", "nb_d")])
            self.assertEqual(len(out["records"]), 2 * (4 + 1 + 4))
            self.assertEqual(out["summary"]["R2p"]["nb_d"]["n"], 2)
            self.assertNotIn("nb_f", out["summary"]["R2p"])
            # resume: drop the last record and rerun; only that record is re-measured (one run_ipm call)
            out["records"].pop()
            bench_lib.write_json(out_path, out)
            with mock.patch.object(bench_lib, "run_ipm", wraps=bench_lib.run_ipm) as spy, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(time_ladder.main(argv), 0)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(len(bench_lib.read_json(out_path)["records"]), 18)

    def test_time_ladder_replaces_a_failed_record(self):
        with tempfile.TemporaryDirectory() as d:
            builds = Path(d) / "builds"
            for rung in ("R0", "R2p"):
                fake_build_dir(builds / rung)
            out_path = Path(d) / "ladder.json"
            failed = {"rung": "R0", "config": "nb_f", "repeat": 1, "ok": False, "error": "exit code 3",
                      "wall_s": 0.1, "t_shoot_cells": None}
            bench_lib.write_json(out_path, {"rungs": ["R0", "R2p"], "repeats": 1, "env": {}, "builds": {},
                                            "records": [failed]})
            argv = ["--rungs", "R0", "R2p", "--builds-root", str(builds), "--repeats", "1",
                    "--ref-root", f"{d}/ref", "--out", str(out_path), "--scratch", f"{d}/scratch", "--no-warmup"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(time_ladder.main(argv), 0)
            records = bench_lib.read_json(out_path)["records"]
            keys = [(r["rung"], r["config"], r["repeat"]) for r in records]
            self.assertEqual(len(keys), 5)   # R0: nb_f, nb_d, rect_f, hm100_f; R2p: nb_d
            self.assertEqual(len(set(keys)), len(keys), keys)   # one record per configuration and repeat
            nb_f = [r for r in records if (r["rung"], r["config"]) == ("R0", "nb_f")]
            self.assertEqual(len(nb_f), 1)
            self.assertTrue(nb_f[0]["ok"], nb_f[0]["error"])

    def test_warm_up_passes_the_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            with mock.patch.object(bench_lib, "run_ipm", wraps=bench_lib.run_ipm) as spy:
                rec = time_rung.warm_up(build, f"{d}/ref", f"{d}/scratch")
            self.assertTrue(rec["ok"], rec["error"])
            self.assertEqual(spy.call_args.kwargs.get("timeout"), 3600)

    def test_time_rung_rejects_an_unknown_or_empty_selection(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            out_path = Path(d) / "results" / "timing_ladder.json"
            for only in (["nb_ff"], ["sf_f"], [], ["nb_f", "nb_ff"]):   # misspelt, outside the ladder, none, one bad of two
                with self.subTest(only=only):
                    err = io.StringIO()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                        rc = time_rung.main(["--build", str(build), "--rung", "R1a", "--only", *only,
                                             "--ref-root", f"{d}/ref", "--results", f"{d}/results",
                                             "--scratch", f"{d}/scratch", "--no-warmup"])
                    self.assertEqual(rc, 2)
                    self.assertIn("unknown or empty configuration selection", err.getvalue())
                    self.assertFalse(out_path.exists())
            # an existing timing file is left alone, and a bad selection still fails rather than reporting "exists"
            bench_lib.write_json(out_path, {"sentinel": True})
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = time_rung.main(["--build", str(build), "--rung", "R1a", "--only", "nb_ff",
                                     "--ref-root", f"{d}/ref", "--results", f"{d}/results",
                                     "--scratch", f"{d}/scratch", "--no-warmup"])
            self.assertEqual(rc, 2)
            self.assertEqual(bench_lib.read_json(out_path), {"sentinel": True})

    def test_time_rung_main_bounds_every_run_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            with mock.patch.object(bench_lib, "run_ipm", wraps=bench_lib.run_ipm) as spy, \
                    contextlib.redirect_stdout(io.StringIO()):
                rc = time_rung.main(["--build", str(build), "--rung", "R1a", "--repeats", "1", "--ref-root", f"{d}/ref",
                                     "--results", f"{d}/results", "--scratch", f"{d}/scratch"])
            self.assertEqual(rc, 0)
            self.assertEqual(spy.call_count, 5)   # warm-up and the four ladder configurations
            self.assertEqual({c.kwargs.get("timeout") for c in spy.call_args_list}, {3600})
            self.assertEqual(list((Path(d) / "scratch").glob("*")), [])

    def test_time_ladder_bounds_every_run_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            builds = Path(d) / "builds"
            for rung in ("R0", "R2p"):
                fake_build_dir(builds / rung)
            out_path = Path(d) / "ladder.json"
            argv = ["--rungs", "R0", "R2p", "--builds-root", str(builds), "--repeats", "1",
                    "--ref-root", f"{d}/ref", "--out", str(out_path), "--scratch", f"{d}/scratch"]
            with mock.patch.object(bench_lib, "run_ipm", wraps=bench_lib.run_ipm) as spy, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(time_ladder.main(argv), 0)
            self.assertEqual(spy.call_count, 6)   # warm-up and the five ladder records
            self.assertEqual({c.kwargs.get("timeout") for c in spy.call_args_list}, {3600})
            scratch = Path(d) / "scratch"
            self.assertEqual([p for p in scratch.rglob("*") if p.is_file()], [])
            self.assertEqual([p for p in scratch.rglob("*") if p.is_dir() and p.parent != scratch], [])


import python_check  # noqa: E402


class PythonCheckTests(unittest.TestCase):
    def test_keyword_table_matches_configs(self):
        for name, kw in python_check.KW.items():
            cfg = configs.CONFIGS[name]
            args = dict(zip(cfg.args[::2], cfg.args[1::2]))
            self.assertEqual(kw["kappa_tot"], float(args["-k"]), name)
            self.assertEqual(kw["shear"], float(args["-y"]), name)
            self.assertEqual(kw["kappa_star"], float(args["-ks"]), name)
            self.assertEqual(kw["num_pixels_y1"], int(args["-npy1"]), name)
            self.assertEqual(kw["half_length_y2"], float(args["-hly2"]), name)
            self.assertEqual(kw.get("is_double", False), cfg.build == "double", name)
            self.assertEqual(kw.get("rectangular", False), args.get("-r", "0") == "1", name)


class PythonCheckDoubleTests(unittest.TestCase):
    def test_shift_center_comes_from_shift(self):
        self.assertEqual(configs.shift_center(), (1e-7, 1e-7))

    def test_double_verdict_combines_both_parts(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            synthetic_reference(f"{d}/ref", "nb_d", 5.0, k="shift1")
            tol = compare_maps.default_tolerances()
            good, bad = base * (1 + 1e-9), base * (1 + 1e-4)
            m_r0, m_shift, v = python_check.judge_double(good, good, f"{d}/ref", "nb_d", tol)
            self.assertTrue(v["pass"], v["failures"])
            self.assertLess(m_r0["mean"], 1e-8)
            v = python_check.judge_double(bad, good, f"{d}/ref", "nb_d", tol)[2]
            self.assertFalse(v["pass"])
            self.assertTrue(v["failures"] and all(f.startswith("vs_r0: ") for f in v["failures"]), v["failures"])
            v = python_check.judge_double(good, bad, f"{d}/ref", "nb_d", tol)[2]
            self.assertFalse(v["pass"])
            self.assertTrue(v["failures"] and all(f.startswith("vs_r0_shift: ") for f in v["failures"]), v["failures"])
            tol["double_actual"]["nb_d"] = {**tol["double_actual"]["default"], "flag_max": 1e-9}   # only a flag
            v = python_check.judge_double(base * (1 + 1e-8), base, f"{d}/ref", "nb_d", tol)[2]
            self.assertTrue(v["pass"], v["failures"])
            self.assertTrue(any(f.startswith("vs_r0: max=") for f in v["flags"]), v["flags"])
            with self.assertRaises(FileNotFoundError):
                python_check.judge_double(good, good, f"{d}/ref", "rect_d", tol)


import report  # noqa: E402


class ReportTests(unittest.TestCase):
    def test_ladder_table_speedups(self):
        summary = {"R0": {"nb_f": {"median_t_shoot_cells": 10.0, "median_wall_s": 14.0}},
                   "R0p": {"nb_f": {"median_t_shoot_cells": 10.0, "median_wall_s": 14.0}},
                   "R1a": {"nb_f": {"median_t_shoot_cells": 5.0, "median_wall_s": 9.0}},
                   "R2p": {},
                   "R3": {"nb_f": {"median_t_shoot_cells": 2.5, "median_wall_s": 6.0}}}
        table = report.ladder_table(summary, ["R0", "R0p", "R1a", "R2p", "R3"], ["nb_f"], "median_t_shoot_cells")
        lines = table.splitlines()
        self.assertTrue(lines[0].startswith("| rung |"))
        self.assertIn("| R0′ | 10.000 | 1.00 | 1.00 |", table)
        self.assertIn("| R1a | 5.000 | 2.00 | 2.00 |", table)
        self.assertIn("| R2′ | – | – | – |", table)
        self.assertIn("| R3 | 2.500 | 4.00 | 2.00 |", table)   # vs prev skips the empty R2′ row

    def test_matrix_table(self):
        r0 = {"nb_f": {"median_t_shoot_cells": 8.0}, "big_f": {"median_t_shoot_cells": 40.0}}
        r5 = {"nb_f": {"median_t_shoot_cells": 2.0}}
        table = report.matrix_table(r0, r5, "median_t_shoot_cells")
        self.assertIn("| nb_f | 8.000 | 2.000 | 4.00 |", table)
        self.assertIn("| big_f | 40.000 | – | – |", table)

    def test_main_renders_from_results_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rec = lambda rung, cfg, t, w: {"rung": rung, "config": cfg, "repeat": 1, "ok": True, "t_shoot_cells": t, "wall_s": w}
            bench_lib.write_json(root / "ladder_interleaved.json", {
                "rungs": ["R0", "R5"], "env": {"gpu": "A100", "hostname": "h", "nvcc": "x", "cuda_visible_devices": "4"},
                "records": [rec("R0", "nb_f", 10, 12), rec("R5", "nb_f", 2, 4)],
                "summary": {"R0": time_rung.summarize([rec("R0", "nb_f", 10, 12)]),
                            "R5": time_rung.summarize([rec("R5", "nb_f", 2, 4)])}})
            bench_lib.write_json(root / "R0" / "timing_full.json", {"records": [rec("R0", "nb_f", 10, 12)]})
            bench_lib.write_json(root / "R5" / "timing_full.json", {"records": [rec("R5", "nb_f", 2, 4)],
                                                                   "summary": time_rung.summarize([rec("R5", "nb_f", 2, 4)])})
            out = root / "tables.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = report.main(["--results", str(root), "--out", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text()
            self.assertIn("| R5 | 2.000 | 5.00 | 5.00 |", text)
            self.assertIn("A100", text)


if __name__ == "__main__":
    unittest.main()
