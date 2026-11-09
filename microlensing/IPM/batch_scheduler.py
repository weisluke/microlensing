"""Scheduling of a batch of maps over worker slots on one or more GPUs.

Each slot runs one map at a time and keeps its GPU memory between maps. During results(), one thread per
slot drives it. A map is admitted to a GPU when the extra memory its slot needs, on top of what that GPU's
slots already hold, fits the GPU's budget. A map goes to a GPU that runs fewer maps when that GPU can start it
at once, so that a batch spreads over every GPU. A map that does not fit runs alone once nothing else runs on
the GPU. A slot that keeps more memory than its next map needs gives it back while another slot waits for memory,
and no map is admitted on a GPU while memory is being freed on it. Finished results wait in a queue until the
consumer releases them. The waiting results are bounded in bytes by max_result_bytes, with always room for one:
a slot starts a map only while the map's result estimate fits next to the waiting results, and a finished map
whose result does not fit keeps it in its slot thread until there is room. Maps in progress do not count against
the limit; the slots and the device memory bound them. A failing map becomes a BatchMapError result and the batch
carries on.

A map that runs out of device memory while other maps run on its GPU goes back to the front of the queue, to be
retried on that GPU only, and that GPU runs fewer maps at a time than were in progress when it ran out (at least
one map fewer than its limit). A map that runs out of memory alone has the
memory of the GPU's idle slots freed and is tried once more; it fails only when it runs out of memory alone on a
GPU whose idle slots hold nothing, and that GPU then gets back the slots its retries parked. A GPU whose CUDA
context broke (a sticky error), or where a slot cannot open, leaves the batch for the scheduler's lifetime: its
threads exit and the maps it was running move to the other GPUs, once each. When no GPU is left, the remaining
maps fail. No map is admitted on a GPU while its memory is being freed or the GPU is being checked.

The scheduler knows nothing of IPM. A backend estimates the maps, describes the GPUs and opens the slots,
and the parameter dicts pass through untouched."""
import itertools
import operator
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Protocol

CUDA_ERROR_MEMORY_ALLOCATION = 2
RESERVE_MIN = 1 << 30
RESERVE_FRACTION = 0.05
DEFAULT_SLOT_CAP = 8
DEFAULT_MAX_RESULT_BYTES = 8 << 30


@dataclass(frozen=True)
class MemoryEstimate:
    """Device memory by component (pixel, star, tree), and the host bytes of a map's result."""
    pixel: int = 0
    star: int = 0
    tree: int = 0
    result: int = 0

    def device_bytes(self) -> int:
        return self.pixel + self.star + self.tree

    def cmax(self, other: "MemoryEstimate") -> "MemoryEstimate":
        """The componentwise maximum: what a slot with grow-only buffers holds after running both."""
        return MemoryEstimate(max(self.pixel, other.pixel), max(self.star, other.star),
                              max(self.tree, other.tree), max(self.result, other.result))


_NOTHING = MemoryEstimate()


@dataclass(frozen=True)
class DeviceInfo:
    free: int
    total: int
    concurrent_managed: bool


class MapFailed(Exception):
    """Raised by a backend when a map cannot be estimated or run. cuda_error is 0 when no CUDA error is known."""

    def __init__(self, message: str, cuda_error: int = 0):
        super().__init__(message)
        self.cuda_error = cuda_error

    def __reduce__(self):
        return type(self), (str(self), self.cuda_error)


class BatchMapError(Exception):
    """The result of a map that failed. index is the map's position in the batch."""

    def __init__(self, index: int, message: str, cuda_error: int = 0):
        super().__init__(message)
        self.index = index
        self.cuda_error = cuda_error

    def __reduce__(self):
        return type(self), (self.index, str(self), self.cuda_error)


@dataclass
class SlotOutcome:
    """What a slot's run() returns: the result, its host bytes, the device memory the slot holds afterwards,
    and whether the map fell back to the direct deposit."""
    result: object
    result_bytes: int
    held: MemoryEstimate
    direct_deposit: bool = False


class Slot(Protocol):
    device: int

    def run(self, params: dict) -> SlotOutcome: ...   # raises MapFailed; always called from the slot's own thread

    def release(self) -> None: ...                    # frees the slot's GPU memory

    def close(self) -> None: ...


class Backend(Protocol):
    def device_count(self) -> int: ...

    def device_info(self, device: int) -> DeviceInfo: ...

    def estimate(self, params: dict) -> MemoryEstimate: ...   # raises MapFailed

    def open_slot(self, device: int) -> Slot: ...

    def check_device(self, device: int) -> int: ...


@dataclass
class Finished:
    """One map's result: an IPM-like object, or a BatchMapError. Its bytes count as waiting, against the
    scheduler's max_result_bytes, until release() is called."""
    index: int
    result: object
    _nbytes: int = field(default=0, repr=False, compare=False)
    _run: object = field(default=None, repr=False, compare=False)

    def release(self) -> None:
        """Stop counting the result's bytes as waiting. Calling it again does nothing."""
        run = self._run
        if run is not None:
            run.release(self)


def _describe(exc):
    return str(exc) or type(exc).__name__


def _cuda_error(exc):
    return exc.cuda_error if isinstance(exc, MapFailed) else 0


def _map_error(index, exc):
    error = BatchMapError(index, _describe(exc), _cuda_error(exc))
    error.__cause__ = exc
    return error


def _growth(held, e):
    """The device bytes a slot that holds `held` adds to run a map of estimate e (its buffers only grow)."""
    return held.cmax(e).device_bytes() - held.device_bytes()


def _empty_stats(devices, dead, max_result_bytes):
    return {"devices": {d: {"budget": 0, "maps": 0, "max_concurrent": 0, "peak_committed": 0, "oom_retries": 0,
                            "parked_slots": 0, "dead": d in dead} for d in devices},
            "direct_deposit_maps": 0, "peak_result_bytes": 0, "peak_in_flight_result_bytes": 0,
            "result_budget": max_result_bytes}


class _SlotState:
    """A slot position on a device. It lives as long as the scheduler; the slot opens on first use."""

    def __init__(self, device, number):
        self.device = device
        self.number = number
        self.slot = None
        self.held = _NOTHING      # device memory the slot holds, or will hold while it runs its admitted map
        self.broken = False       # its release() raised: it is not used again, and its memory stays counted
        self.shared = False       # another map was admitted on the device while this slot's latest map was


class _Job(NamedTuple):
    index: int
    estimate: MemoryEstimate
    frees: list                 # (slot state, slot, held before the admission): release() each before the map runs


class _Run:
    """The state of one results() call. Every field is read and written under the scheduler's condition."""

    def __init__(self, cond, params, devices, dead, max_result_bytes):
        self.cond = cond
        self.params = params
        self.estimates = [None] * len(params)
        self.pending = deque()    # indices waiting to run, in batch order
        self.queue = deque()      # Finished items not yet yielded
        self.stopping = False
        self.error = None         # an exception that escaped the scheduler's own code in a slot thread
        self.threads = []
        self.budget = {}
        self.n_slots = {}
        self.slot_limit = {}
        self.running = {}
        self.committed = {}
        self.releases = {}        # device -> release() calls in flight on it; admission on it waits for them
        self.mem_waiters = {}     # device -> slots whose latest admission attempt waited for memory
        self.checks = {}          # device -> check_device() calls in flight on it; admission on it waits for them
        self.active = {}          # device -> slots with an admitted map
        self.holding = set()      # slots whose thread keeps a finished result until there is room in the queue
        self.attempts = [0] * len(params)   # how often each map moved off a GPU that left the batch
        self.pinned = {}          # index -> the GPU that alone may run this out-of-memory retry
        self.parks = [0] * len(params)      # how much each map's retries lowered slot_limit on its pinned GPU
        self.total_running = 0
        self.waiting_bytes = 0    # host bytes of finished results queued, or yielded and not released
        # result bytes of the maps in progress: a running map's estimate, then its result's bytes while the result
        # waits in its slot thread for room in the queue
        self.in_progress_bytes = 0
        self.stats = _empty_stats(devices, dead, max_result_bytes)

    def release(self, item):
        with self.cond:
            if item._run is not self:
                return
            item._run = None
            self.waiting_bytes -= item._nbytes
            self.cond.notify_all()


class Scheduler:
    """Runs batches of maps on worker slots: maps_per_gpu slots per GPU (slot_cap when it is None), each
    driven by its own thread while results() runs.

    stats describes the latest results() call and is updated while it runs:
    {"devices": {d: {"budget", "maps", "max_concurrent", "peak_committed", "oom_retries", "parked_slots",
    "dead"}}, "direct_deposit_maps": int, "peak_result_bytes": int, "peak_in_flight_result_bytes": int,
    "result_budget": int}. "maps" counts the maps that succeeded on that GPU, "oom_retries" the maps sent back to
    the queue after running out of memory there, "parked_slots" how many of its slots no longer run maps in this
    call, and "dead" whether the GPU has left the batch. "peak_result_bytes" is the most bytes of finished results
    that waited at once (queued, or yielded and not released), and "peak_in_flight_result_bytes" the most bytes of
    those and of the results of the maps in progress together (their estimates while they run). "result_budget"
    is max_result_bytes, the limit on the waiting results."""

    def __init__(self, backend: Backend, devices=None, maps_per_gpu=None,
                 max_result_bytes=DEFAULT_MAX_RESULT_BYTES, slot_cap=DEFAULT_SLOT_CAP):
        count = backend.device_count()
        if devices is None:
            if count < 1:
                raise RuntimeError("no GPU is visible")
            devices = range(count)
        devices = [operator.index(d) for d in devices]
        if not devices:
            raise ValueError("devices is empty")
        if len(set(devices)) != len(devices):
            raise ValueError(f"devices has repeated entries: {devices}")
        missing = [d for d in devices if not 0 <= d < count]
        if missing:
            raise ValueError(f"no GPU {missing} among the {count} visible")
        if maps_per_gpu is not None and operator.index(maps_per_gpu) < 1:
            raise ValueError(f"maps_per_gpu must be at least 1, got {maps_per_gpu!r}")
        if operator.index(slot_cap) < 1:
            raise ValueError(f"slot_cap must be at least 1, got {slot_cap!r}")
        if max_result_bytes < 0:
            raise ValueError(f"max_result_bytes must not be negative, got {max_result_bytes!r}")
        self._backend = backend
        self._devices = devices
        self._maps_per_gpu = maps_per_gpu
        self._slot_cap = slot_cap
        self._max_result_bytes = max_result_bytes
        self._cond = threading.Condition()
        self._slots = {d: [] for d in devices}
        self._dead = {}           # device -> (cuda error, message); a dead device stays dead
        self._run = None          # the results() call whose threads may still be alive
        self._closed = False
        self.stats = _empty_stats(devices, self._dead, max_result_bytes)

    def results(self, params: list[dict]) -> Iterator[Finished]:
        """Run every map and yield one Finished per map, in completion order.

        Release each Finished when its result is no longer needed: until then it counts as waiting. At most
        max_result_bytes of finished results wait at a time (always at least one). While the waiting results
        leave no room for a map's result, no slot starts that map, and a finished map whose result does not fit
        waits in its slot until there is room; so a consumer that waits for the next result while it keeps
        unreleased results that leave no room for it waits forever. The maps in progress are not counted. Closing
        the generator (break, an exception in the consumer, Ctrl-C) lets the maps in progress finish, drops their
        results and joins every slot thread."""
        params = list(params)
        run = self._begin(params)
        try:
            self._estimate(run)
            self._start(run)
            for _ in range(len(params)):
                yield self._next_finished(run)
        finally:
            self._stop(run)

    def close(self) -> None:
        """Stop a results() call in progress, then close every slot. Calling it again does nothing. A slot of a
        GPU that left the batch, or one whose memory could not be freed, that fails to close gives a warning;
        any other slot that fails to close raises, after every slot was closed."""
        with self._cond:
            self._closed = True
            run = self._run
            if run is not None:
                run.stopping = True
            self._cond.notify_all()
        if run is not None and not self._finish_stop(run):
            return    # called from a slot thread, which cannot wait for itself: leave the slots open
        with self._cond:
            slots = [(st.device, st.slot, st.broken or st.device in self._dead)
                     for states in self._slots.values() for st in states if st.slot is not None]
            for states in self._slots.values():
                for st in states:
                    st.slot = None
                    st.held = _NOTHING
            self._cond.notify_all()
        first = None
        for d, slot, broken in slots:
            try:
                slot.close()
            except Exception as exc:
                if broken:    # a slot of a GPU that left the batch, or one whose memory could not be freed
                    warnings.warn(f"closing a slot on GPU {d} failed: {_describe(exc)}", RuntimeWarning,
                                  stacklevel=2)
                elif first is None:
                    first = exc
        if first is not None:
            raise first

    # ---- the consumer's side

    def _begin(self, params):
        with self._cond:
            if self._closed:
                raise RuntimeError("the scheduler is closed")
            prev = self._run
            if prev is not None and not prev.stopping:
                raise RuntimeError("another results() call is still in progress on this scheduler")
        if prev is not None:
            self._finish_stop(prev)   # a stop that could not wait for its threads: wait now
        with self._cond:
            if self._run is not None:
                raise RuntimeError("another results() call is still in progress on this scheduler")
            run = _Run(self._cond, params, self._devices, self._dead, self._max_result_bytes)
            self._run = run
            self.stats = run.stats
            self._cond.notify_all()
            return run

    def _estimate(self, run):
        """Estimate every map on the calling thread, before any slot thread starts: an estimate can read a
        star file and allocate, so it must not overlap running maps."""
        for i, p in enumerate(run.params):
            try:
                estimate = self._backend.estimate(p)
            except Exception as exc:
                with self._cond:
                    run.queue.append(Finished(i, _map_error(i, exc)))
                    self._cond.notify_all()
            else:
                with self._cond:
                    run.estimates[i] = estimate
                    run.pending.append(i)
                    self._cond.notify_all()

    def _start(self, run):
        with self._cond:
            devices = [d for d in self._devices if d not in self._dead]
        infos = {d: self._backend.device_info(d) for d in devices}
        unmanaged = []
        with self._cond:
            for d in devices:
                info = infos[d]
                held = self._held_on(d)
                run.budget[d] = info.free + held - max(RESERVE_MIN, int(RESERVE_FRACTION * info.total))
                n = self._maps_per_gpu or self._slot_cap
                if not info.concurrent_managed:
                    unmanaged.append(d)
                    n = 1
                states = self._slots[d]
                while len(states) < n:
                    states.append(_SlotState(d, len(states)))
                run.n_slots[d] = run.slot_limit[d] = n
                run.running[d] = 0
                run.committed[d] = held
                run.releases[d] = 0
                run.checks[d] = 0
                run.mem_waiters[d] = set()
                run.active[d] = set()
                dev = run.stats["devices"][d]
                dev["budget"] = run.budget[d]
                dev["peak_committed"] = held
            # slot 0 of every GPU first, then slot 1, and so on, so that the first maps find a thread on every GPU
            for k in range(max(run.n_slots.values(), default=0)):
                for d in devices:
                    st = self._slots[d][k] if k < run.n_slots[d] else None
                    if st is not None and not st.broken:
                        run.threads.append(threading.Thread(target=self._slot_thread, args=(run, st),
                                                            name=f"ipm-batch-{d}-{st.number}", daemon=True))
            self._cond.notify_all()
        for d in unmanaged:
            warnings.warn(f"GPU {d} has no concurrent managed memory access, so its maps run one at a time",
                          stacklevel=3)
        for t in run.threads:
            t.start()

    def _next_finished(self, run):
        with self._cond:
            while not run.queue:
                if run.error is not None:
                    raise RuntimeError("a batch scheduler thread failed") from run.error
                if run.stopping:
                    raise RuntimeError("the scheduler was closed while results() was running")
                if not any(t.is_alive() for t in run.threads):
                    self._fail_stranded(run)
                    if not run.queue:
                        raise RuntimeError("the batch scheduler lost track of a map")
                    break
                self._cond.wait(timeout=0.1)   # wakes up regularly so that Ctrl-C is delivered
            item = run.queue.popleft()
            self._cond.notify_all()
            return item

    def _fail_stranded(self, run):
        """Fail the maps still waiting when no GPU or slot thread is left to run them (lock held)."""
        while run.pending:
            self._fail(run, run.pending.popleft(), *self._no_gpu())
        self._cond.notify_all()

    def _stop(self, run):
        with self._cond:
            run.stopping = True
            self._cond.notify_all()
        self._finish_stop(run)

    def _finish_stop(self, run):
        """Wait for the stopping run's threads, then drop its unyielded results. Returns False, without
        waiting, when called from one of those threads."""
        if threading.current_thread() in run.threads:
            return False
        for t in run.threads:
            if t.ident is not None:
                t.join()
        with self._cond:
            leftovers = list(run.queue)
            run.queue.clear()
            if self._run is run:
                self._run = None
            self._cond.notify_all()
        for item in leftovers:
            item.release()
        return True

    # ---- the slot threads' side

    def _slot_thread(self, run, st):
        try:
            while self._one_map(run, st):
                pass
        except BaseException as exc:   # a fault in the scheduler itself: stop the batch and tell the consumer
            with self._cond:
                if run.error is None:
                    run.error = exc
                run.stopping = True
                self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def _one_map(self, run, st):
        """Admit, run and hand on one map in this slot; False when the thread should exit. The map's job, outcome,
        result and exception are locals of this call only, so that nothing of a finished map stays referenced
        while the thread waits in _admit for its next map."""
        job = self._admit(run, st)
        if job is None:
            return False
        stage, outcome, exc = self._execute(run, st, job)
        if stage == "done":
            self._complete(run, st, job, outcome)
        else:
            self._failed(run, st, job, stage, exc)
        return True

    def _admit(self, run, st):
        """Wait until the map at the head of pending can start in this slot. Returns None when the thread
        should exit: the run is stopping, the device is dead, the slot is broken, or nothing is left to run
        anywhere."""
        with self._cond:
            waiters = run.mem_waiters[st.device]
            while True:
                was_waiting = st in waiters
                waiters.discard(st)
                if run.stopping or st.device in self._dead or st.broken or \
                        (not run.pending and run.total_running == 0):
                    self._cond.notify_all()
                    return None
                job = self._try_admit(run, st) if run.pending else None
                if job is not None or was_waiting != (st in waiters):
                    self._cond.notify_all()
                if job is not None:
                    return job
                self._cond.wait()

    def _try_admit(self, run, st):
        """The admission rules (lock held) for the first map in pending that this slot's device may run (maps
        pinned to another GPU are skipped). Returns a _Job, or None to wait. A slot leaves the map to another GPU
        that runs fewer maps and would start it at once (_taken_elsewhere). A slot that waits because its map
        does not fit next to the maps running on its device is added to mem_waiters."""
        d = st.device
        if run.releases[d] > 0 or run.checks[d] > 0:
            return None    # memory of d is still being freed (committed[d] is not free yet), or d is being checked
        pos = next((k for k, j in enumerate(run.pending) if run.pinned.get(j, d) == d), None)
        if pos is None:
            return None
        i = run.pending[pos]
        e = run.estimates[i]
        if not self._room_for(run, e.result):
            return None
        if run.running[d] >= run.slot_limit[d]:
            return None
        if self._taken_elsewhere(run, d, pos):
            return None    # a GPU that runs fewer maps starts this one at once: leave it to that GPU
        need = st.held.cmax(e)
        victims = []
        if run.committed[d] + _growth(st.held, e) > run.budget[d]:
            if run.running[d] > 0:
                run.mem_waiters[d].add(st)
                return None
            # Nothing runs on d: free what its idle slots hold and run this map alone.
            victims = [x for x in self._slots[d] if x.held.device_bytes() > 0 and not x.broken]
        elif need.device_bytes() > e.device_bytes() and any(w is not st for w in run.mem_waiters[d]):
            # The map fits, but this slot keeps more than the map needs (grow-only buffers from a bigger map)
            # while another slot of d waits for memory: free this slot first, then hold just the map's estimate.
            victims = [st]
        frees = [(x, x.slot, x.held) for x in victims if x.slot is not None]
        for x in victims:
            x.held = _NOTHING
        need = st.held.cmax(e)
        del run.pending[pos]
        st.held = need
        st.shared = bool(run.active[d])
        for x in run.active[d]:
            x.shared = True
        run.active[d].add(st)
        run.committed[d] = self._held_on(d)
        run.running[d] += 1
        run.total_running += 1
        run.in_progress_bytes += e.result
        run.releases[d] += len(frees)
        self._update_peaks(run, d)
        return _Job(i, e, frees)

    def _execute(self, run, st, job):
        """Free the job's slots, open this slot if needed and run the map, all outside the lock. Never raises.
        Returns (stage, outcome, exception): stage is "done" when the map ran, "free" when it did not run because
        freeing memory took this slot or its device out of the batch, "open" when open_slot raised and "run"
        when the run raised."""
        stage = "free"
        try:
            error = self._free(run, st.device, job.frees)
            with self._cond:
                if st.broken or st.device in self._dead:
                    return stage, None, error
            stage = "open"
            slot = st.slot
            if slot is None:
                slot = self._backend.open_slot(st.device)
                with self._cond:
                    st.slot = slot
                    self._cond.notify_all()
            stage = "run"
            return "done", slot.run(run.params[job.index]), None
        except BaseException as exc:
            return stage, None, exc

    def _free(self, run, d, frees):
        """Call release() on each slot in frees, outside the lock. The caller counted each call in
        run.releases[d] and set the slot's held to what it holds once freed, so admission on d waits until the
        count is back down. A release() that raises leaves the slot's memory counted as before and the slot
        unusable, and makes the device checked: a device that reports an error leaves the batch. Returns the
        first exception raised, or None."""
        first = None
        for state, slot, held_before in frees:
            error = None
            failed, code = False, 0
            try:
                try:
                    slot.release()
                except BaseException as exc:
                    error = exc
                    failed, code = self._check(d, exc)
            finally:
                with self._cond:
                    if error is not None:
                        state.held = held_before    # the memory may still be allocated: keep counting it
                        state.broken = True
                        run.committed[d] = self._held_on(d)
                        if failed:
                            self._mark_dead(run, d, code, _describe(error))
                        elif not self._usable(run, d):
                            self._unpin(run, d)     # no slot of d can run its retries any more
                        self._update_peaks(run, d)
                    run.releases[d] -= 1
                    self._cond.notify_all()
            if error is not None:
                if first is None:
                    first = error
                warnings.warn(f"freeing the memory of a slot on GPU {d} failed: {_describe(error)}; "
                              + ("the GPU leaves the batch" if failed else "the slot is not used again"),
                              RuntimeWarning, stacklevel=2)
        return first

    def _check(self, d, exc):
        """Ask the backend whether device d still works, outside the lock. Returns (failed, CUDA error). A
        device that cannot be checked counts as failed, with the code of the error that made us check."""
        try:
            code = int(self._backend.check_device(d))
        except Exception:
            return True, _cuda_error(exc)
        return code != 0, code

    def _complete(self, run, st, job, outcome):
        """A map that ran. Its slot is idle on its GPU at once, and its result joins the queue once it fits next to
        the waiting results (_room_for). Until then this thread keeps the result, counted as in progress. A
        stopping run drops it."""
        d = st.device
        nbytes = outcome.result_bytes
        with self._cond:
            run.running[d] -= 1
            run.total_running -= 1
            run.active[d].discard(st)
            st.held = outcome.held
            run.committed[d] = self._held_on(d)
            run.in_progress_bytes += nbytes - job.estimate.result
            run.stats["devices"][d]["maps"] += 1
            if outcome.direct_deposit:
                run.stats["direct_deposit_maps"] += 1
            self._update_peaks(run, d)
            self._cond.notify_all()
            run.holding.add(st)
            while not run.stopping and not self._room_for(run, nbytes):
                self._cond.wait()
            run.holding.discard(st)
            run.in_progress_bytes -= nbytes
            if not run.stopping:
                run.waiting_bytes += nbytes
                run.queue.append(Finished(job.index, outcome.result, nbytes, run))
            self._update_peaks(run, d)
            self._cond.notify_all()

    def _failed(self, run, st, job, stage, exc):
        """A job whose map did not succeed. After an out-of-memory error the slot's memory is freed, and with it
        the memory of the device's idle slots if the map ran alone; after another CUDA error the device is
        checked. Both happen outside the lock and are counted so that nothing is admitted on the device
        meanwhile. Then _classify decides."""
        d = st.device
        code = _cuda_error(exc)
        device_error = None
        frees = []
        freed_idle = False
        checking = False
        if stage == "run":
            with self._cond:
                dead = d in self._dead
                if not dead and code == CUDA_ERROR_MEMORY_ALLOCATION and st.slot is not None:
                    frees = [(st, st.slot, st.held)]
                    st.held = _NOTHING
                    if not st.shared:
                        # it ran alone: the idle slots' memory may be what it lacked, so free that too
                        for x in self._slots[d]:
                            if x is not st and not x.broken and x not in run.active[d] and x.held.device_bytes() > 0:
                                if x.slot is not None:
                                    frees.append((x, x.slot, x.held))
                                x.held = _NOTHING
                                freed_idle = True
                    run.committed[d] = self._held_on(d)
                    run.releases[d] += len(frees)
                    self._cond.notify_all()
                elif not dead and code not in (0, CUDA_ERROR_MEMORY_ALLOCATION):
                    checking = True
                    run.checks[d] += 1
                    self._cond.notify_all()
            if frees:
                self._free(run, d, frees)
            elif checking:
                try:
                    failed, device_code = self._check(d, exc)
                except BaseException:
                    with self._cond:
                        run.checks[d] -= 1
                        self._cond.notify_all()
                    raise
                if failed:
                    device_error = device_code
        with self._cond:
            try:
                self._classify(run, st, job, stage, exc, device_error, freed_idle)
            finally:
                if checking:
                    run.checks[d] -= 1    # only now, so that d is not used before its check's verdict is in
                self._update_peaks(run, d)
                self._cond.notify_all()

    def _classify(self, run, st, job, stage, exc, device_error, freed_idle=False):
        """Decide what a failed job means (lock held). This is the one place that classifies failures:
        - a GPU where a slot cannot open, or whose check after a CUDA error reports an error, leaves the batch,
          and the map whose error made us check fails with the GPU's error;
        - a map that fails on a GPU that has left the batch moves to another GPU, once (_move);
        - a map that did not run because its slot could not free its memory goes back to the front of the queue;
        - a map that runs out of memory goes back to the front of the queue, pinned to its GPU, if another map
          ran on that GPU while it did, or if it ran alone while idle slots held memory, which _failed has freed.
          In the first case the GPU's slot_limit drops below the number of maps in progress when the failure is
          classified (this one included), and below its current value, so that the retry starts only once fewer
          maps run there. Otherwise the map fails, and its GPU gets back the slots its retries parked;
        - any other failure fails the map."""
        d = st.device
        i = job.index
        run.running[d] -= 1
        run.total_running -= 1
        run.active[d].discard(st)
        run.in_progress_bytes -= job.estimate.result
        if st.slot is None:
            st.held = _NOTHING    # the slot never opened, so it holds nothing
            run.committed[d] = self._held_on(d)
        if stage == "open":
            self._mark_dead(run, d, _cuda_error(exc), _describe(exc))
        elif device_error is not None and d not in self._dead:
            self._mark_dead(run, d, device_error, _describe(exc))
            self._fail(run, i, *self._dead[d], exc)
            return
        if d in self._dead:
            self._move(run, i, d, exc)
        elif stage == "free" and st.broken:
            run.pending.appendleft(i)
        elif stage == "run" and _cuda_error(exc) == CUDA_ERROR_MEMORY_ALLOCATION:
            dev = run.stats["devices"][d]
            if st.shared or freed_idle:
                run.pending.appendleft(i)
                if self._usable(run, d):
                    run.pinned[i] = d
                dev["oom_retries"] += 1
                if st.shared:
                    in_progress = run.running[d] + 1    # running[d] no longer counts this map
                    limit = max(1, min(run.slot_limit[d] - 1, in_progress - 1))
                    run.parks[i] += run.slot_limit[d] - limit
                    run.slot_limit[d] = limit
            else:
                run.slot_limit[d] = min(run.n_slots[d], run.slot_limit[d] + run.parks[i])
                run.parks[i] = 0
                self._fail(run, i, CUDA_ERROR_MEMORY_ALLOCATION,
                           f"out of device memory with no other map on GPU {d}", exc)
            dev["parked_slots"] = run.n_slots[d] - run.slot_limit[d]
        else:
            run.queue.append(Finished(i, _map_error(i, exc)))

    def _move(self, run, i, d, cause):
        """A map that failed on GPU d, which has left the batch, goes back to the front of the queue to run on
        another GPU, once (lock held). The second time it fails with d's error; with no GPU left, it fails."""
        if run.attempts[i] > 0:
            self._fail(run, i, *self._dead[d], cause)
        elif self._all_dead():
            self._fail(run, i, *self._no_gpu(), cause)
        else:
            run.attempts[i] += 1
            run.pending.appendleft(i)

    def _mark_dead(self, run, d, code, detail):
        """Take GPU d out of the batch for the scheduler's lifetime (lock held): its slot threads exit and it is
        not used again. When no GPU is left, the maps still waiting fail."""
        if d in self._dead:
            return
        self._dead[d] = (code, f"GPU {d} failed ({code}): {detail}")
        run.stats["devices"][d]["dead"] = True
        self._unpin(run, d)
        if self._all_dead():
            self._fail_stranded(run)
        self._cond.notify_all()

    # ---- helpers (lock held)

    def _room_for(self, run, nbytes):
        """Whether a result of nbytes fits next to the waiting results: within max_result_bytes, or nothing
        waits (always room for one). It decides both whether a map may start and whether a finished one may join
        the queue, so the waiting results never exceed max(max_result_bytes, one result)."""
        return run.waiting_bytes == 0 or run.waiting_bytes + nbytes <= self._max_result_bytes

    def _taken_elsewhere(self, run, d, pos):
        """Whether another GPU that runs fewer maps than d would start pending[pos], d's next map, at once: the map
        is not pinned to d, no map ahead of it is pinned to that GPU, the GPU is below its slot limit, no memory
        is being freed on it and it is not being checked, and one of its idle slots fits the map in its budget (a
        slot whose thread keeps a finished result until the queue has room is not idle).
        Each GPU on the way to the one that starts the map runs fewer maps than the one before, so it is found."""
        i = run.pending[pos]
        e = run.estimates[i]
        for d2, running in run.running.items():
            if d2 == d or d2 in self._dead or running >= run.running[d] or running >= run.slot_limit[d2] or \
                    run.releases[d2] > 0 or run.checks[d2] > 0 or run.pinned.get(i, d2) != d2:
                continue
            if any(run.pinned.get(j) == d2 for j in itertools.islice(run.pending, pos)):
                continue    # d2 would start a retry pinned to it first
            room = run.budget[d2] - run.committed[d2]
            if any(not x.broken and x not in run.active[d2] and x not in run.holding and _growth(x.held, e) <= room
                   for x in self._slots[d2][:run.n_slots[d2]]):
                return True
        return False

    def _usable(self, run, d):
        """Whether d can still run maps in this call: it is not dead and one of its slots is not broken."""
        return d not in self._dead and not all(x.broken for x in self._slots[d][:run.n_slots[d]])

    def _unpin(self, run, d):
        """Let any GPU run the retries pinned to d, which can no longer run them."""
        for i in [i for i, x in run.pinned.items() if x == d]:
            del run.pinned[i]
            run.parks[i] = 0
        self._cond.notify_all()

    def _all_dead(self):
        return all(d in self._dead for d in self._devices)

    def _no_gpu(self):
        """(code, message) for a map that no GPU is left to run: the error of the first GPU that failed."""
        if not self._dead:
            return 0, "no usable GPU left"
        code, message = next(iter(self._dead.values()))
        return code, "no usable GPU left: " + message

    def _fail(self, run, i, code, message, cause=None):
        error = BatchMapError(i, message, code)
        error.__cause__ = cause
        run.queue.append(Finished(i, error))

    def _held_on(self, d):
        return sum(st.held.device_bytes() for st in self._slots[d])

    def _update_peaks(self, run, d):
        dev = run.stats["devices"][d]
        dev["max_concurrent"] = max(dev["max_concurrent"], run.running[d])
        dev["peak_committed"] = max(dev["peak_committed"], run.committed[d])
        run.stats["peak_result_bytes"] = max(run.stats["peak_result_bytes"], run.waiting_bytes)
        run.stats["peak_in_flight_result_bytes"] = max(run.stats["peak_in_flight_result_bytes"],
                                                       run.waiting_bytes + run.in_progress_bytes)
