"""Login-node tests of the rule that judges batched maps against serial maps:
python -m unittest tests.batch.test_batch_rules -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu"))

import batch_rules  # noqa: E402


def _map(seed=0, shape=(300, 300)):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.5, sigma=0.8, size=shape)


def _perturbed(a, rel, seed):
    rng = np.random.default_rng(seed)
    return a * (1 + rel * rng.standard_normal(a.shape))


class JudgeTests(unittest.TestCase):
    def test_identical_maps_pass(self):
        a = _map()
        self.assertEqual(batch_rules.judge(a, a, batch_rules.noise(a, a), False), [])
        self.assertEqual(batch_rules.judge(a, a, batch_rules.noise(a, a), True), [])

    def test_noise_level_difference_passes(self):
        a = _map()
        b = _perturbed(a, 1e-6, 1)
        self.assertEqual(batch_rules.judge(b, a, batch_rules.noise(a, b), False), [])

    def test_ten_times_noise_fails(self):
        a = _map()
        b = _perturbed(a, 1e-6, 1)
        c = _perturbed(a, 1e-5, 2)
        fails = batch_rules.judge(c, a, batch_rules.noise(a, b), False)
        self.assertNotEqual(fails, [])
        self.assertTrue(any(f.startswith("mean") for f in fails), fails)

    def test_floor_depends_on_precision(self):
        a = _map()
        c = _perturbed(a, 1e-9, 3)   # a mean relative difference of about 8e-10: under the float floor, over the double one
        zero = batch_rules.noise(a, a)
        self.assertEqual(batch_rules.judge(c, a, zero, False), [])
        self.assertNotEqual(batch_rules.judge(c, a, zero, True), [])

    def test_sum_ratio_beyond_noise_fails(self):
        a = _map()
        b = _perturbed(a, 1e-6, 1)
        c = b * (1 + 1e-5)   # the same scatter as the noise, with the total off by 1e-5
        fails = batch_rules.judge(c, a, batch_rules.noise(a, b), False)
        self.assertTrue(any(f.startswith("|sum_ratio|") for f in fails), fails)

    def test_max_is_never_a_failure(self):
        a = _map()
        b = _perturbed(a, 1e-6, 1)
        c = b.copy()
        c[10, 10] *= 1.001   # one pixel far off: max jumps, the percentiles and the mean barely move
        self.assertGreater(batch_rules.noise(a, c)["max"], 100 * batch_rules.noise(a, b)["max"])
        self.assertEqual(batch_rules.judge(c, a, batch_rules.noise(a, b), False), [])

    def test_noise_takes_the_largest_of_several_runs(self):
        a = _map()
        b = _perturbed(a, 1e-6, 1)
        c = _perturbed(a, 1e-5, 2)
        both = batch_rules.noise(a, b, c)
        for key in ("mean", "p999", "p9999", "max"):
            self.assertEqual(both[key], max(batch_rules.noise(a, b)[key], batch_rules.noise(a, c)[key]), key)
        self.assertEqual(abs(both["sum_ratio"]), max(abs(batch_rules.noise(a, b)["sum_ratio"]),
                                                     abs(batch_rules.noise(a, c)["sum_ratio"])))
        d = _perturbed(a, 1e-5, 3)   # ten times the noise of b, as much as c
        self.assertNotEqual(batch_rules.judge(d, a, batch_rules.noise(a, b), False), [])
        self.assertEqual(batch_rules.judge(d, a, both, False), [])

    def test_parity_maps_allow_a_larger_sum_difference(self):
        a = _map()
        b = _perturbed(a, 1e-6, 1)
        c = b.copy()
        k = np.unravel_index(np.argmax(a), a.shape)
        c[k] += 5e-6 * a.sum()   # the sum moves by 5e-6, as when triangles change parity; the mean barely moves
        noise = batch_rules.noise(a, b)
        self.assertTrue(any(f.startswith("|sum_ratio|") for f in batch_rules.judge(c, a, noise, False)))
        self.assertEqual(batch_rules.judge(c, a, noise, False, parity=True), [])

    def test_non_finite_candidate_fails(self):
        a = _map()
        c = a.copy()
        c[0, 0] = np.nan
        self.assertIn("non-finite pixels", batch_rules.judge(c, a, batch_rules.noise(a, a), False))


if __name__ == "__main__":
    unittest.main()
