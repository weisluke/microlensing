"""GPU tests of the worker, estimate and snapshot exports (skipped without a GPU):
python -m unittest tests.batch.test_exports_gpu -v"""
import ctypes
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM import lib_ipm, lib_ipm_double  # noqa: E402
import measure_tree_bytes as mtb  # noqa: E402
import test_exports as te  # noqa: E402

LIBS = (("float", lib_ipm.lib), ("double", lib_ipm_double.lib))
HAS_GPU = lib_ipm.lib.get_device_count() > 0 if hasattr(lib_ipm.lib, "get_device_count") else False


@unittest.skipUnless(HAS_GPU, "no GPU visible")
class ExportGpuTests(unittest.TestCase):
    def test_estimate_bounds_allocation(self):
        names = ("pixel", "star", "tree")
        for label, lib in LIBS:
            for name, params in mtb.CONFIGS.items():
                for parities in (0, 1):
                    for seed in (3, 4, 5):
                        with self.subTest(precision=label, config=name, parities=parities, seed=seed):
                            obj = lib.IPM_init()
                            try:
                                self.assertTrue(lib.set_worker(obj, 0))
                                mtb.apply(lib, obj, params, parities)
                                lib.set_random_seed(obj, seed)
                                est = (ctypes.c_ulonglong * 4)()
                                alloc = (ctypes.c_ulonglong * 3)()
                                self.assertTrue(lib.estimate_memory(obj, est))
                                self.assertTrue(lib.run(obj, 0))
                                lib.get_allocated_bytes(obj, alloc)
                                for k in range(3):
                                    self.assertGreaterEqual(
                                        est[k], alloc[k],
                                        f"{names[k]}: est={est[k]} alloc={alloc[k]} ratio={est[k] / alloc[k]:.4f}")
                                self.assertLessEqual(
                                    sum(est[:3]), 1.5 * sum(alloc) + (64 << 20),
                                    f"total: est={sum(est[:3])} alloc={sum(alloc)} ratio={sum(est[:3]) / sum(alloc):.4f}")
                                print(f"ratios {label} {name} p{parities} s{seed}: "
                                      + " ".join(f"{names[k]}={est[k] / alloc[k]:.3f}" for k in range(3)))
                            finally:
                                lib.IPM_delete(obj)

    def test_snapshot_copies_inputs(self):
        for label, lib in LIBS:
            obj = lib.IPM_init()
            try:
                self.assertTrue(lib.set_worker(obj, 0))
                te.set_valid_map(lib, obj)
                lib.set_half_length_y1(obj, 5)
                lib.set_half_length_y2(obj, 5)
                lib.set_num_pixels_y1(obj, 100)
                lib.set_num_pixels_y2(obj, 80)
                lib.set_outfile_prefix(obj, b"snap_")
                lib.set_write_parities(obj, 1)
                lib.set_random_seed(obj, 99)
                self.assertTrue(lib.run(obj, 0))
                snap = lib.IPM_snapshot(obj)
                self.assertTrue(snap)
                try:
                    self.assertEqual(te.read_all(lib, snap), te.read_all(lib, obj))
                    self.assertEqual(lib.get_num_stars(snap), lib.get_num_stars(obj))
                finally:
                    lib.IPM_delete(snap)
            finally:
                lib.IPM_delete(obj)

    def test_take_last_cuda_error_reports_oom(self):
        lib = lib_ipm.lib
        lib.take_last_cuda_error()
        obj = lib.IPM_init()
        try:
            self.assertTrue(lib.set_worker(obj, 0))
            # three 100000 x 100000 float maps need 111 GiB, more than any GPU holds (a single map of 37 GiB fits on an A100)
            te.set_valid_map(lib, obj, write_parities=1)
            lib.set_num_pixels_y1(obj, 100000)
            lib.set_num_pixels_y2(obj, 100000)
            self.assertFalse(lib.run(obj, 0))
            self.assertEqual(lib.take_last_cuda_error(), 2)
            self.assertEqual(lib.take_last_cuda_error(), 0)
        finally:
            lib.IPM_delete(obj)

    def test_set_worker_rejects_invalid_device(self):
        for label, lib in LIBS:
            obj = lib.IPM_init()
            try:
                self.assertFalse(lib.set_worker(obj, lib.get_device_count()))
                self.assertFalse(lib.set_worker(obj, -1))
            finally:
                lib.IPM_delete(obj)

    def test_check_device_healthy(self):
        self.assertEqual(lib_ipm.lib.check_device(0), 0)

    def test_get_device_info(self):
        out = (ctypes.c_ulonglong * 2)()
        cm = ctypes.c_int(-1)
        self.assertTrue(lib_ipm.lib.get_device_info(0, out, ctypes.byref(cm)))
        self.assertGreater(out[0], 0)
        self.assertLessEqual(out[0], out[1])
        self.assertEqual(cm.value, 1)
        self.assertFalse(lib_ipm.lib.get_device_info(lib_ipm.lib.get_device_count(), out, ctypes.byref(cm)))


if __name__ == "__main__":
    unittest.main()
