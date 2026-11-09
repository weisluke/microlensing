"""Tests of the IPM wrapper's shared input handling and host-only results:
python -m unittest tests.batch.test_ipm_wrapper -v"""
import filecmp
import inspect
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM import lib_ipm  # noqa: E402
from microlensing.IPM.ipm import IPM, _INPUT_NAMES  # noqa: E402

HAS_GPU = lib_ipm.lib.get_device_count() > 0 if hasattr(lib_ipm.lib, "get_device_count") else False

CASES = [
    {},
    dict(kappa_tot=0.5, shear=0.2, smooth_fraction=0.7),
    dict(mass_function="Salpeter", m_solar=0.5, m_lower=0.08, m_upper=1.5, rectangular=True,
         approx=False, num_pixels_y1=700, random_seed=5, outfile_prefix="x/"),
]


class InputTests(unittest.TestCase):
    def test_input_names_match_signature(self):
        self.assertEqual(set(_INPUT_NAMES),
                         set(inspect.signature(IPM.__init__).parameters) - {"self", "verbose", "is_double"})
        self.assertEqual(len(_INPUT_NAMES), len(set(_INPUT_NAMES)))

    def test_apply_inputs_matches_init(self):
        for kw in CASES:
            with self.subTest(kw=kw):
                a = IPM(**kw)
                b = IPM()
                b._apply_inputs(**kw)
                for name in _INPUT_NAMES:
                    if name == "starfile":
                        continue
                    self.assertEqual(getattr(a, name), getattr(b, name), name)


def _make_worker(**extra):
    kw = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=10, half_length_y2=10,
              num_pixels_y1=600, num_pixels_y2=600, random_seed=3, write_parities=True)
    kw.update(extra)
    w = IPM(**kw)
    assert w.lib.set_worker(w.obj, 0)
    return w


def _sorted_stars(arr):
    arr = np.asarray(arr)
    return arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]


@unittest.skipUnless(HAS_GPU, "no GPU visible")
class WorkerResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_from_worker_matches_worker(self):
        w = _make_worker()
        self.assertTrue(w.lib.run(w.obj, 0))
        r = IPM._from_worker(w)
        self.assertTrue(r._host_only)
        shape = (w.num_pixels_y2, w.num_pixels_y1)
        for name, get in (("magnifications", w.lib.copy_pixels),
                          ("magnifications_minima", w.lib.copy_pixels_minima),
                          ("magnifications_saddles", w.lib.copy_pixels_saddles)):
            ref = np.empty(shape, dtype=w._map_dtype)
            self.assertTrue(get(w.obj, ref.ctypes.data_as(
                __import__("ctypes").POINTER(w._map_ctype))))
            self.assertTrue(np.array_equal(getattr(r, name), ref), name)
        ws = np.ctypeslib.as_array(w.lib.get_stars(w.obj), shape=(w.num_stars, 3))
        self.assertTrue(np.array_equal(_sorted_stars(r.stars.stars), _sorted_stars(ws)))
        self.assertEqual(r.num_stars, w.num_stars)
        self.assertEqual(r.corner, w.corner)
        self.assertEqual(r.t_shoot_cells, w.t_shoot_cells)
        self.assertEqual(r.random_seed, w.random_seed)

    def test_from_worker_without_run_raises(self):
        w = _make_worker()
        with self.assertRaises(Exception):
            IPM._from_worker(w)

    def test_result_save_matches_worker_save(self):
        flags = dict(write_stars=True, write_maps=True, write_parities=True, write_histograms=True)
        w = _make_worker(**flags)
        self.assertTrue(w.lib.run(w.obj, 0))
        r = IPM._from_worker(w)
        a, b = Path(self.tmp, "A"), Path(self.tmp, "B")
        a.mkdir(), b.mkdir()
        r.outfile_prefix = str(a) + "/"
        r.save()
        w.outfile_prefix = str(b) + "/"
        self.assertTrue(w.lib.save(w.obj, 0))
        names = sorted(p.name for p in a.iterdir())
        self.assertEqual(names, sorted(p.name for p in b.iterdir()))
        self.assertGreater(len(names), 4)
        for n in names:
            self.assertTrue(filecmp.cmp(a / n, b / n, shallow=False), n)

    def test_save_rejects_bad_array(self):
        w = _make_worker(write_maps=True)
        self.assertTrue(w.lib.run(w.obj, 0))
        r = IPM._from_worker(w)
        r.magnifications = r.magnifications[:, ::2]
        with self.assertRaises(ValueError):
            r.save()

    def test_rerun_of_result_is_plain_ipm(self):
        w = _make_worker(write_maps=True)
        self.assertTrue(w.lib.run(w.obj, 0))
        r = IPM._from_worker(w)
        r.outfile_prefix = self.tmp + "/"
        r.run()
        self.assertFalse(r._host_only)
        r.save()
        self.assertTrue(any(Path(self.tmp).iterdir()))


if __name__ == "__main__":
    unittest.main()
