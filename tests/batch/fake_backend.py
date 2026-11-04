"""Stand-in for the batch scheduler's ctypes backend in login-node tests: slots sleep instead of running
a map, and the backend records what happened.

Each map's estimate comes from its params keys pixel, star, tree and result (defaults 1 MiB, 0, 0 and
1 MiB), and it runs for params.get("time", run_time) seconds. While a map runs, its slot holds the
componentwise maximum of what it held and the estimate. Afterwards it holds exactly the estimate or, with
grow_only=True, keeps that maximum (as the real grow-only buffers do) until release(). release() takes
release_time seconds before the memory counts as free. With params["direct_deposit"] true, the outcome
reports the direct deposit. A result records the device that ran it (FakeResult.device).

Failure injection:
- fail_estimate_when / fail_run_when: callables params -> exception instance or None; the exception is raised.
- save_fails_when: callable params -> bool; FakeResult.save() raises OSError when it is true.
- oom_above={device: k}: a run raises MapFailed(..., 2) when it would make more than k maps run on that device.
- dead_after={device: n}: after n maps started on the device, runs raise MapFailed("illegal address", 700)
  (also a run in progress when it happens) and check_device returns 700.
- device_errors={device: code}: what check_device returns for a device the fake has not killed (default 0).
  A test may change it while a batch runs, for a context that broke without failing the runs in progress.
  check_device takes check_time seconds.
- open_fails_for: devices whose open_slot raises RuntimeError.
- fail_release_when / fail_close_when: callables slot -> exception instance or None; release() or close()
  raises it. A failed release() keeps the slot's memory allocated; a failed close() still closes the slot.

Records, all under one lock:
- log: (event, device, slot_id, params.get("index"), time.monotonic()) for the events "estimate", "open",
  "open_failed", "start", "end", "oom", "dead" (dead_after killed the device), "release", "release_failed",
  "check" and "checked" (a check_device call begins and returns) and "close". A run that is not rejected for
  memory logs "start" and "end".
- max_concurrency[d]: the most maps seen running at once on device d; peak_alloc[d]: the most memory seen held
  by d's open slots.
- per slot: release_calls and close_calls (failed calls included) and run_threads (the name of the thread of
  each run).
- violations: a slot run while busy, closed or being released, or released or closed while running."""
import threading
import time

from microlensing.IPM.batch_scheduler import DeviceInfo, MapFailed, MemoryEstimate, SlotOutcome

MiB = 1 << 20
GiB = 1 << 30
NOTHING = MemoryEstimate()


class FakeResult:
    def __init__(self, params, backend=None, device=None):
        self.params = params
        self.device = device
        self.save_calls = 0
        self._backend = backend

    def save(self):
        self.save_calls += 1
        backend = self._backend
        if backend is not None:
            with backend.lock:
                backend.saved.append(self.params)
            if backend.save_fails_when is not None and backend.save_fails_when(self.params):
                raise OSError(f"fake save failed for map {self.params.get('index')}")


class FakeSlot:
    def __init__(self, backend, device, slot_id):
        self.backend = backend
        self.device = device
        self.slot_id = slot_id
        self.alloc = NOTHING
        self.busy = False
        self.releasing = False
        self.closed = False
        self.release_calls = 0
        self.close_calls = 0
        self.run_threads = []

    def run(self, params):
        b = self.backend
        d = self.device
        e = b.estimate_of(params)
        with b.lock:
            if self.busy or self.closed or self.releasing:
                b.violations.append(f"run on slot {self.slot_id} while busy={self.busy} closed={self.closed} "
                                    f"releasing={self.releasing}")
            self.run_threads.append(threading.current_thread().name)
            if b.oom_above is not None and d in b.oom_above and b.running[d] + 1 > b.oom_above[d]:
                b.log_event("oom", d, self.slot_id, params)
                raise MapFailed(f"out of memory on fake GPU {d}", 2)
            self.busy = True
            b.running[d] += 1
            b.max_concurrency[d] = max(b.max_concurrency[d], b.running[d])
            b.device_runs[d] += 1
            if b.dead_after is not None and d in b.dead_after and b.device_runs[d] > b.dead_after[d] \
                    and not b.dead[d]:
                b.dead[d] = True
                b.log_event("dead", d, self.slot_id, params)
            dead = b.dead[d]
            self.alloc = self.alloc.cmax(e)
            b.note_alloc(d)
            b.log_event("start", d, self.slot_id, params)
        try:
            if dead:
                raise MapFailed("illegal address", 700)
            if b.fail_run_when is not None:
                exc = b.fail_run_when(params)
                if exc is not None:
                    raise exc
            time.sleep(params.get("time", b.run_time))
            with b.lock:
                if b.dead[d]:
                    raise MapFailed("illegal address", 700)
                if not b.grow_only:
                    self.alloc = e
                held = self.alloc
            return SlotOutcome(FakeResult(params, b, d), e.result, held, bool(params.get("direct_deposit", False)))
        finally:
            with b.lock:
                b.running[d] -= 1
                self.busy = False
                b.log_event("end", d, self.slot_id, params)

    def release(self):
        b = self.backend
        with b.lock:
            if self.busy or self.releasing:
                b.violations.append(f"release of slot {self.slot_id} while busy={self.busy} "
                                    f"releasing={self.releasing}")
            self.releasing = True
            self.release_calls += 1
        time.sleep(b.release_time)
        exc = b.fail_release_when(self) if b.fail_release_when is not None else None
        with b.lock:
            self.releasing = False
            if exc is None:
                self.alloc = NOTHING
                b.log_event("release", self.device, self.slot_id, {})
            else:
                b.log_event("release_failed", self.device, self.slot_id, {})
        if exc is not None:
            raise exc

    def close(self):
        b = self.backend
        with b.lock:
            if self.busy or self.releasing:
                b.violations.append(f"close of slot {self.slot_id} while busy={self.busy} "
                                    f"releasing={self.releasing}")
            self.close_calls += 1
            self.closed = True
            self.alloc = NOTHING
            b.log_event("close", self.device, self.slot_id, {})
        exc = b.fail_close_when(self) if b.fail_close_when is not None else None
        if exc is not None:
            raise exc


class FakeBackend:
    def __init__(self, devices=2, free=40 << 30, total=40 << 30, concurrent_managed=True, run_time=0.01,
                 fail_estimate_when=None, fail_run_when=None, save_fails_when=None, oom_above=None,
                 dead_after=None, open_fails_for=(), grow_only=False, release_time=0.0, device_errors=None,
                 fail_release_when=None, fail_close_when=None, check_time=0.0):
        self.devices = devices
        self.free = free
        self.total = total
        self.concurrent_managed = concurrent_managed
        self.run_time = run_time
        self.fail_estimate_when = fail_estimate_when
        self.fail_run_when = fail_run_when
        self.save_fails_when = save_fails_when
        self.oom_above = oom_above
        self.dead_after = dead_after
        self.open_fails_for = tuple(open_fails_for)
        self.grow_only = grow_only
        self.release_time = release_time
        self.device_errors = dict(device_errors or {})
        self.fail_release_when = fail_release_when
        self.fail_close_when = fail_close_when
        self.check_time = check_time
        self.lock = threading.Lock()
        self.log = []
        self.slots = []
        self.saved = []
        self.violations = []
        self.open_calls = 0
        self.running = {d: 0 for d in range(devices)}
        self.max_concurrency = {d: 0 for d in range(devices)}
        self.peak_alloc = {d: 0 for d in range(devices)}
        self.device_runs = {d: 0 for d in range(devices)}
        self.dead = {d: False for d in range(devices)}

    # helpers (the caller holds self.lock where noted)
    def log_event(self, event, device, slot_id, params):   # holds the lock
        self.log.append((event, device, slot_id, params.get("index"), time.monotonic()))

    def allocated(self, device):   # holds the lock
        return sum(s.alloc.device_bytes() for s in self.slots if s.device == device and not s.closed)

    def note_alloc(self, device):   # holds the lock
        self.peak_alloc[device] = max(self.peak_alloc[device], self.allocated(device))

    def log_snapshot(self):
        with self.lock:
            return list(self.log)

    @staticmethod
    def estimate_of(params):
        return MemoryEstimate(params.get("pixel", MiB), params.get("star", 0), params.get("tree", 0),
                              params.get("result", MiB))

    # the Backend protocol
    def device_count(self):
        return self.devices

    def device_info(self, device):
        with self.lock:
            return DeviceInfo(self.free - self.allocated(device), self.total, self.concurrent_managed)

    def estimate(self, params):
        with self.lock:
            self.log_event("estimate", None, None, params)
        if self.fail_estimate_when is not None:
            exc = self.fail_estimate_when(params)
            if exc is not None:
                raise exc
        return self.estimate_of(params)

    def open_slot(self, device):
        with self.lock:
            if device in self.open_fails_for:
                self.log_event("open_failed", device, None, {})
                raise RuntimeError(f"cannot open a slot on fake GPU {device}")
            self.open_calls += 1
            slot = FakeSlot(self, device, len(self.slots))
            self.slots.append(slot)
            self.log_event("open", device, slot.slot_id, {})
            return slot

    def check_device(self, device):
        with self.lock:
            self.log_event("check", device, None, {})
        time.sleep(self.check_time)
        with self.lock:
            self.log_event("checked", device, None, {})
            return 700 if self.dead[device] else self.device_errors.get(device, 0)
