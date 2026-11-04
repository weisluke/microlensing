"""Login-node tests (no GPU) of the ctypes backend's failure handling, with the real library and a slot whose
set_worker and run succeed without a GPU: python -m unittest tests.batch.test_batch_backend -v"""
import ctypes
import gc
import sys
import unittest
import weakref
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM.batch import BatchedIPM  # noqa: E402
from microlensing.IPM.batch_backend import CtypesBackend  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError, DeviceInfo, MapFailed, MemoryEstimate  # noqa: E402

GiB = 1 << 30
MAP = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=5, half_length_y2=5, num_pixels_y1=100,
           num_pixels_y2=100)


class _HostLib:
    """The real library, except that set_worker and run succeed without touching a GPU. A worker that "ran"
    this way has no stars, so IPM._from_worker raises (its snapshot is null) with no CUDA error."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def set_worker(self, obj, device):
        return True

    def run(self, obj, verbose):
        return True


class HostOnlyBackend(CtypesBackend):
    """A CtypesBackend with one fake GPU that keeps every slot it opens."""

    def __init__(self):
        super().__init__(False, 0)
        self.lib = _HostLib(self.lib)
        self.slots = []

    def device_count(self):
        return 1

    def device_info(self, device):
        return DeviceInfo(free=40 * GiB, total=40 * GiB, concurrent_managed=True)

    def check_device(self, device):
        return 0

    def open_slot(self, device):
        slot = super().open_slot(device)
        self.slots.append(slot)
        return slot


class FailedMapReleaseTests(unittest.TestCase):
    def run_failing_map(self, backend):
        """Run one map that fails in its slot; return (the BatchMapError, a weakref to the slot's worker) after
        close(), with the error still held."""
        with mock.patch("microlensing.IPM.batch._make_backend", return_value=backend):
            b = BatchedIPM(**MAP, num_maps=1)
        out = dict(b.run())
        self.assertEqual(len(backend.slots), 1)
        worker = weakref.ref(backend.slots[0].ipm)
        b.close()
        gc.collect()
        return out[0], worker

    def test_failed_copy_frees_the_worker_on_close(self):
        error, worker = self.run_failing_map(HostOnlyBackend())
        self.assertIsInstance(error, BatchMapError)
        self.assertIn("Error copying the IPM state", str(error))
        self.assertEqual(error.cuda_error, 0)
        self.assertIsInstance(error.__cause__, MapFailed)
        self.assertIsNotNone(error.__cause__.__cause__)   # the original exception stays the cause
        self.assertIsNone(worker(), "the error handed to the user keeps the worker (and its GPU memory) alive")

    def test_rejected_inputs_free_the_worker_on_close(self):
        backend = HostOnlyBackend()
        # the estimate passes, so that the slot's own _apply_inputs raises the ValueError
        backend.estimate = lambda params: MemoryEstimate(pixel=1 << 20, result=1 << 20)
        with mock.patch.dict(MAP, m_lower=2.0, m_upper=1.0):
            error, worker = self.run_failing_map(backend)
        self.assertIsInstance(error, BatchMapError)
        self.assertIn("m_lower must be <= m_upper", str(error))
        self.assertIsNone(worker(), "the error handed to the user keeps the worker (and its GPU memory) alive")

    def test_failed_estimate_keeps_no_frames_of_the_probe(self):
        backend = HostOnlyBackend()
        with self.assertRaises(MapFailed) as cm:
            backend.estimate(dict(MAP, m_lower=2.0, m_upper=1.0))
        tb = cm.exception.__cause__.__traceback__
        while tb is not None:   # the frames below estimate() (the probe's _apply_inputs and setters) are cleared
            if tb.tb_frame.f_code.co_name != "estimate":
                self.assertEqual(tb.tb_frame.f_locals, {}, tb.tb_frame.f_code.co_name)
            tb = tb.tb_next


if __name__ == "__main__":
    unittest.main()
