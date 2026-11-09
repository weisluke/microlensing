"""The ctypes backend of the batch scheduler (batch_scheduler.Backend and Slot).

Each slot owns one long-lived C++ IPM worker, bound to its GPU and its own stream by set_worker, which keeps
its device buffers from one map to the next. Before each map the worker's inputs go back to their defaults
(reset_inputs) and the map's are applied, so nothing of the previous map's inputs carries over (run() writes
derived values such as the scaled mass limits and the seed back into the object). A finished map is returned
as an IPM object that holds no GPU memory (IPM._from_worker). The estimates use one probe object that never
runs."""
import ctypes
import traceback

import numpy as np

from . import lib_ipm, lib_ipm_double
from .batch_scheduler import CUDA_ERROR_MEMORY_ALLOCATION, DeviceInfo, MapFailed, MemoryEstimate, SlotOutcome
from .ipm import IPM


def _library(is_double: bool):
    return (lib_ipm_double if is_double else lib_ipm).lib


def _map_failed(message: str, cause: BaseException, cuda_error: int = 0) -> MapFailed:
    """A MapFailed for an exception that IPM's Python code raised, to be raised `from cause`. The local variables
    of the cause's finished frames are cleared: they hold the worker (self in _apply_inputs, worker in
    _from_worker), and the error reaches the user, who may hold it after close() should have freed the worker and
    its GPU memory. The traceback's file and line information stays."""
    traceback.clear_frames(cause.__traceback__)
    return MapFailed(message, cuda_error)


def _open_failure(lib, device: int, code: int) -> str:
    """Why set_worker failed on device, for the error message."""
    count = max(0, lib.get_device_count())
    if not 0 <= device < count:
        return f"there is no GPU {device} among the {count} visible"
    if code == CUDA_ERROR_MEMORY_ALLOCATION:
        return (f"CUDA error {code} (out of memory) while creating its stream; another process may hold the "
                f"GPU's memory")
    if code != 0:
        return f"CUDA error {code} while making the GPU current or creating its stream"
    return "set_worker failed"


class CtypesBackend:
    """The Backend of the batch scheduler for the float (is_double=False) or double IPM library."""

    def __init__(self, is_double: bool, verbose: int):
        self.is_double = bool(is_double)
        self.verbose = verbose
        self.lib = _library(self.is_double)
        if not hasattr(self.lib, "set_worker"):
            raise RuntimeError("this IPM library has no batch exports")
        self._probe = IPM(is_double=self.is_double)   # only estimates: it never runs, so it holds no GPU memory

    def device_count(self) -> int:
        return max(0, self.lib.get_device_count())

    def device_info(self, device: int) -> DeviceInfo:
        out = (ctypes.c_ulonglong * 2)()
        concurrent_managed = ctypes.c_int(0)
        if not self.lib.get_device_info(device, out, ctypes.byref(concurrent_managed)):
            raise RuntimeError(f"cannot read the free memory of GPU {device}")
        return DeviceInfo(free=out[0], total=out[1], concurrent_managed=bool(concurrent_managed.value))

    def estimate(self, params: dict) -> MemoryEstimate:
        """Called on the scheduler's calling thread only, never while maps run (a star file is read here)."""
        probe = self._probe
        self.lib.reset_inputs(probe.obj)
        try:
            probe._apply_inputs(**params)
        except ValueError as exc:
            raise _map_failed(str(exc), exc) from exc
        out = (ctypes.c_ulonglong * 4)()
        if not self.lib.estimate_memory(probe.obj, out):
            raise MapFailed("the input parameters were rejected (see the messages above)")
        return MemoryEstimate(pixel=out[0], star=out[1], tree=out[2], result=out[3])

    def open_slot(self, device: int) -> "CtypesSlot":
        return CtypesSlot(self, device)

    def check_device(self, device: int) -> int:
        return self.lib.check_device(device)


class CtypesSlot:
    """A worker on one GPU. run() is called from the slot's own thread only, and release() and close() only
    while no map runs in it (the scheduler's contract)."""

    def __init__(self, backend: CtypesBackend, device: int):
        self.device = device
        self.verbose = backend.verbose
        self.lib = backend.lib
        self.ipm = IPM(is_double=backend.is_double, verbose=backend.verbose)
        self._itemsize = np.dtype(self.ipm._map_dtype).itemsize
        self.lib.take_last_cuda_error()
        if not self.lib.set_worker(self.ipm.obj, device):
            code = self.lib.take_last_cuda_error()
            self.ipm = None
            raise RuntimeError(f"cannot open a worker on GPU {device}: {_open_failure(self.lib, device, code)}")

    def run(self, params: dict) -> SlotOutcome:
        lib, obj = self.lib, self.ipm.obj
        lib.reset_inputs(obj)
        try:
            self.ipm._apply_inputs(**params)
        except ValueError as exc:
            raise _map_failed(str(exc), exc) from exc
        lib.take_last_cuda_error()   # the code is per thread, and a successful earlier call can leave one behind
        if not lib.run(obj, self.verbose):
            raise MapFailed("IPM run failed (see the messages above)", lib.take_last_cuda_error())
        lib.take_last_cuda_error()
        try:
            result = IPM._from_worker(self.ipm)
        except Exception as exc:   # the copy of the maps or the snapshot of the stars failed
            detail = str(exc) or type(exc).__name__
            raise _map_failed(f"copying the finished map from GPU {self.device} failed: {detail}", exc,
                              lib.take_last_cuda_error()) from exc
        alloc = (ctypes.c_ulonglong * 3)()
        lib.get_allocated_bytes(obj, alloc)
        held = MemoryEstimate(pixel=alloc[0], star=alloc[1], tree=alloc[2], result=0)
        maps = (result.magnifications, result.magnifications_minima, result.magnifications_saddles)
        result_bytes = sum(m.nbytes for m in maps if m is not None) + result.num_stars * 3 * self._itemsize
        return SlotOutcome(result, result_bytes, held, direct_deposit=not lib.get_perimeter_deposit_used(obj))

    def release(self) -> None:
        """Free the worker's device buffers; it can run again afterwards."""
        self.lib.take_last_cuda_error()
        if not self.lib.release_memory(self.ipm.obj):
            raise MapFailed(f"freeing the memory of a worker on GPU {self.device} failed",
                            self.lib.take_last_cuda_error())

    def close(self) -> None:
        """Delete the worker, which frees its device memory and its stream."""
        self.ipm = None
