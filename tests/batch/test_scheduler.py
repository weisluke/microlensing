"""Login-node tests (no GPU) of the batch scheduler, driven through a fake backend:
python -m unittest tests.batch.test_scheduler -v"""
import _thread
import faulthandler
import gc
import sys
import threading
import time
import unittest
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TEST_TIMEOUT = 60   # seconds; a test that hangs kills the run with a traceback of every thread

from fake_backend import GiB, MiB, FakeBackend, FakeResult  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError, MapFailed, Scheduler  # noqa: E402


def maps(n, **extra):
    return [{"index": i, **extra} for i in range(n)]


def batch_threads():
    return [t for t in threading.enumerate() if t.name.startswith("ipm-batch-")]


def collect(sched, params):
    out = []
    for f in sched.results(params):
        out.append(f)
        f.release()
    return out


def runs(fake):
    """index -> list of (device, start position, end position, slot_id) in the fake's log."""
    log = fake.log_snapshot()
    open_runs = {}
    out = {}
    for pos, (event, device, slot_id, index, _) in enumerate(log):
        if event == "start":
            open_runs[(slot_id, index)] = pos
        elif event == "end":
            start = open_runs.pop((slot_id, index))
            out.setdefault(index, []).append((device, start, pos, slot_id))
    return out


def releases(fake, device, after=-1):
    """(position, slot_id) of each release() on the device that returned after the given log position."""
    return [(pos, e[2]) for pos, e in enumerate(fake.log_snapshot())
            if e[0] == "release" and e[1] == device and pos > after]


def max_running_after(fake, after, device):
    """The most maps running at once on the device at log positions after the given one."""
    running = peak = 0
    for pos, e in enumerate(fake.log_snapshot()):
        if e[1] == device and e[0] in ("start", "end"):
            running += 1 if e[0] == "start" else -1
            if pos > after:
                peak = max(peak, running)
    return peak


def max_running(fake):
    """The most maps running at once on all devices together."""
    running = peak = 0
    for e in fake.log_snapshot():
        if e[0] in ("start", "end"):
            running += 1 if e[0] == "start" else -1
            peak = max(peak, running)
    return peak


def overlaps(a, b):
    return a[1] < b[2] and b[1] < a[2]


def events(fake, name, device=None):
    """(position, slot_id) of each event of that name in the fake's log, on the device if one is given."""
    return [(pos, e[2]) for pos, e in enumerate(fake.log_snapshot())
            if e[0] == name and (device is None or e[1] == device)]


def gpu_failed(d):
    return f"GPU {d} failed (700): illegal address"


OOM_ALONE = "out of device memory with no other map on GPU 0"


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        faulthandler.dump_traceback_later(TEST_TIMEOUT, exit=True)
        self.addCleanup(faulthandler.cancel_dump_traceback_later)
        self.addCleanup(self.assert_no_batch_threads)

    def assert_no_batch_threads(self):
        self.assertEqual(batch_threads(), [])

    def make(self, fake, **kwargs):
        sched = Scheduler(fake, **kwargs)
        self.addCleanup(self.check_fake, fake)
        self.addCleanup(sched.close)
        return sched

    def check_fake(self, fake):
        self.assertEqual(fake.violations, [])
        for slot in fake.slots:
            names = set(slot.run_threads)
            self.assertLessEqual(len(names), 1, f"slot {slot.slot_id} ran on several threads: {names}")
            for name in names:
                self.assertTrue(name.startswith(f"ipm-batch-{slot.device}-"), name)

    @staticmethod
    def always_oom(index, **kwargs):
        """A fake on which map `index` always runs out of memory, and the memory allocated on its GPU at each of
        its tries (the list returned with the fake)."""
        seen = []
        fake = None

        def fail(p):
            if p["index"] != index:
                return None
            d = int(threading.current_thread().name.split("-")[2])
            with fake.lock:
                seen.append(fake.allocated(d))
            return MapFailed("out of memory", 2)

        fake = FakeBackend(fail_run_when=fail, **kwargs)
        return fake, seen

    def assert_successes(self, out, n, failed=()):
        self.assertEqual(sorted(f.index for f in out), list(range(n)))
        for f in out:
            if f.index in failed:
                self.assertIsInstance(f.result, BatchMapError)
            else:
                self.assertIsInstance(f.result, FakeResult)
                self.assertEqual(f.result.params["index"], f.index)

    def assert_stopped_cleanly(self, fake, sched, n):
        self.assertEqual(batch_threads(), [])
        log = fake.log_snapshot()
        starts = [e[3] for e in log if e[0] == "start"]
        ends = [e[3] for e in log if e[0] == "end"]
        self.assertEqual(sorted(starts), sorted(ends))
        self.assertLess(len(starts), n)
        closed_at = time.monotonic()
        time.sleep(0.15)
        self.assertEqual([e for e in fake.log_snapshot() if e[0] == "start" and e[4] > closed_at], [])
        self.assert_successes(collect(sched, maps(10)), 10)

    def test_all_maps_complete_with_their_indices(self):
        fake = FakeBackend()
        sched = self.make(fake)
        params = [{"index": i, "direct_deposit": i % 5 == 0} for i in range(20)]
        out = collect(sched, params)
        self.assert_successes(out, 20)
        self.assertEqual(sum(d["maps"] for d in sched.stats["devices"].values()), 20)
        self.assertEqual(sched.stats["direct_deposit_maps"], 4)
        for f in out:
            self.assertIs(f.result.params, params[f.index])

    def test_completion_order_not_batch_order(self):
        fake = FakeBackend()
        sched = self.make(fake)
        params = maps(10, time=0.01)
        params[0]["time"] = 0.2
        out = collect(sched, params)
        self.assert_successes(out, 10)
        self.assertNotEqual(out[0].index, 0)

    def test_concurrency_limited_by_memory(self):
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(24, pixel=3 * GiB, time=0.05)), 24)
        for d in (0, 1):
            dev = sched.stats["devices"][d]
            self.assertEqual(dev["budget"], 9 * GiB)
            self.assertEqual(dev["max_concurrent"], 3)
            self.assertLessEqual(fake.max_concurrency[d], 3)
            self.assertLessEqual(fake.peak_alloc[d], 9 * GiB)

    def test_slot_cap_and_maps_per_gpu(self):
        fake = FakeBackend()
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(48, time=0.05)), 48)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 8)
            self.assertLessEqual(fake.max_concurrency[d], 8)

        fake = FakeBackend()
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(16, time=0.05)), 16)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 2)
            self.assertLessEqual(fake.max_concurrency[d], 2)

    def test_no_concurrent_managed_access_limits_to_one_slot(self):
        fake = FakeBackend(concurrent_managed=False)
        sched = self.make(fake)
        with self.assertWarns(UserWarning):
            out = collect(sched, maps(6, time=0.02))
        self.assert_successes(out, 6)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 1)
            self.assertLessEqual(fake.max_concurrency[d], 1)

    def test_small_batches_spread_over_every_gpu(self):
        # 8 GPUs of 8 slots: 8 maps of equal length run one on each GPU and 16 maps two on each, instead of
        # filling the slots of the first GPUs. Every map is admitted long before the first one ends
        for n, per_gpu in ((8, 1), (16, 2)):
            fake = FakeBackend(devices=8)
            sched = self.make(fake)
            self.assert_successes(collect(sched, maps(n, time=0.5)), n)
            self.assertEqual({d: dev["maps"] for d, dev in sched.stats["devices"].items()},
                             {d: per_gpu for d in range(8)}, n)
            self.assertEqual(fake.max_concurrency, {d: per_gpu for d in range(8)}, n)

    def test_devices_argument_restricts_devices(self):
        fake = FakeBackend()
        sched = self.make(fake, devices=[1])
        self.assert_successes(collect(sched, maps(12, time=0.01)), 12)
        self.assertEqual({e[1] for e in fake.log_snapshot() if e[0] == "start"}, {1})
        self.assertEqual(list(sched.stats["devices"]), [1])

    def slow_consumer(self, sched, params, limit, delay=0.03):
        """Consume every map, releasing each one `delay` s after receiving it. Returns the items and the most
        result bytes seen waiting (queued, or received and not released) when each one was received."""
        out, seen = [], []
        for f in sched.results(params):
            out.append(f)
            seen.append(sched.stats["peak_result_bytes"])
            time.sleep(delay)
            f.release()
        self.assertTrue(all(s <= limit for s in seen), seen)
        return out

    def test_waiting_results_bounded_for_a_slow_consumer(self):
        # 2 GPUs of 8 slots, 1 GiB results and a 3 GiB limit, with a consumer slower than the maps: the finished
        # results waiting for it never exceed the limit, while the maps in progress do not count against it, so
        # all 16 slots start maps at once
        fake = FakeBackend()
        sched = self.make(fake, max_result_bytes=3 * GiB)
        self.assert_successes(self.slow_consumer(sched, maps(40, result=GiB, time=0.02), 3 * GiB), 40)
        stats = sched.stats
        self.assertEqual(stats["result_budget"], 3 * GiB)
        self.assertEqual(stats["peak_result_bytes"], 3 * GiB)
        self.assertEqual(max_running(fake), 16)
        self.assertGreaterEqual(stats["peak_in_flight_result_bytes"], 16 * GiB)
        self.assertLessEqual(stats["peak_in_flight_result_bytes"], 3 * GiB + 16 * GiB)   # limit + one per slot
        # a limit below one result: always room for one result, never for two, and the maps still overlap
        fake = FakeBackend(devices=1)
        sched = self.make(fake, max_result_bytes=1)
        self.assert_successes(self.slow_consumer(sched, maps(16, result=GiB, time=0.02), GiB), 16)
        self.assertEqual(sched.stats["result_budget"], 1)
        self.assertEqual(sched.stats["peak_result_bytes"], GiB)
        self.assertEqual(fake.max_concurrency[0], 8)
        self.assertLessEqual(sched.stats["peak_in_flight_result_bytes"], GiB + 8 * GiB)

    def test_large_results_run_side_by_side(self):
        # 40000^2 float maps have 6 GiB results: one GPU with 2 slots at the default 8 GiB limit runs 2 of them at
        # once, since the maps in progress are not counted, but never queues two (12 GiB > 8 GiB)
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(6, result=6 * GiB, time=0.1)), 6)
        self.assertEqual(fake.max_concurrency[0], 2)
        self.assertEqual(sched.stats["result_budget"], 8 * GiB)
        self.assertEqual(sched.stats["peak_result_bytes"], 6 * GiB)
        self.assertEqual(sched.stats["peak_in_flight_result_bytes"], 12 * GiB)
        # the same with a slow consumer: a finished map that does not fit waits in its slot
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(self.slow_consumer(sched, maps(6, result=6 * GiB, time=0.1), 6 * GiB, 0.15), 6)
        self.assertEqual(fake.max_concurrency[0], 2)
        self.assertEqual(sched.stats["peak_result_bytes"], 6 * GiB)
        # 8 GPUs: the limit is not raised, and every GPU runs maps at once
        fake = FakeBackend(devices=8)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(16, result=6 * GiB, time=0.1)), 16)
        self.assertEqual(max_running(fake), 16)
        self.assertEqual(sched.stats["result_budget"], 8 * GiB)
        self.assertEqual(sched.stats["peak_result_bytes"], 6 * GiB)

    def test_waiting_slot_threads_keep_no_finished_result(self):
        # 4 slots on 1 GPU: map 0 runs for 1 s, maps 1-7 end within about 0.1 s. While map 0 runs, the other 3
        # slot threads wait for a map: the results that the consumer received, released and dropped must be freed
        # by then, not kept until the threads exit
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=4)
        params = maps(8, time=0.02)
        params[0]["time"] = 1.0
        gen = sched.results(params)
        refs = []
        for _ in range(7):
            f = next(gen)
            self.assertNotEqual(f.index, 0)
            refs.append(weakref.ref(f.result))
            f.release()
            del f
        gc.collect()
        self.assertNotIn(0, [e[3] for e in fake.log_snapshot() if e[0] == "end"])   # map 0 still runs
        self.assertEqual(sum(r() is not None for r in refs), 0)
        f = next(gen)
        self.assertEqual(f.index, 0)
        f.release()
        gen.close()

    def test_gpu_whose_slots_wait_for_queue_room_takes_no_map_from_another(self):
        # 2 GPUs of 2 slots, 9 GiB budgets, a 3 GiB limit. The device memory places L (6 GiB) and X1 then X2 on
        # one GPU, B1 and B2 (4 GiB each) on the other. B1 and B2 finish while X1 and X2 wait (2 GiB), so both wait
        # in their slots. Once the consumer releases X1, S fits next to X2 but B1 and B2 do not: S must start on
        # the GPU of L at once, not be left to the other GPU, whose slots cannot start it
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake, maps_per_gpu=2, max_result_bytes=3 * GiB)
        params = [{"index": 0, "pixel": 6 * GiB, "result": 100 * MiB, "time": 1.5},
                  {"index": 1, "pixel": 4 * GiB, "result": int(2.5 * GiB), "time": 0.2},
                  {"index": 2, "pixel": 4 * GiB, "result": int(2.5 * GiB), "time": 0.2},
                  {"index": 3, "pixel": GiB, "result": GiB, "time": 0.05},
                  {"index": 4, "pixel": GiB, "result": GiB, "time": 0.05},
                  {"index": 5, "result": int(1.5 * GiB), "time": 0.05}]
        out, released = [], {}
        for f in sched.results(params):
            out.append(f)
            if f.index in (3, 4):
                time.sleep(0.35)
            released[f.index] = time.monotonic()
            f.release()
        self.assert_successes(out, 6)
        r = runs(fake)
        (big_l,), (b1,), (b2,), (x1,), (x2,), (s,) = (r[i] for i in range(6))
        self.assertEqual({b1[0], b2[0]}, {1 - big_l[0]})
        self.assertEqual({x1[0], x2[0], s[0]}, {big_l[0]})
        start_s = fake.log_snapshot()[s[1]][4]
        self.assertGreater(start_s, released[3])
        self.assertLess(start_s, released[4])

    def test_oversized_map_runs_alone_and_small_maps_continue(self):
        # 20 small maps first, so that idle slots of both GPUs hold memory when the big map comes up
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        params = maps(40, pixel=GiB, time=0.05)
        params[20].update(pixel=int(8.5 * GiB), time=0.5)
        self.assert_successes(collect(sched, params), 40)
        self.assertEqual(sched.stats["devices"][0]["budget"], 9 * GiB)
        r = runs(fake)
        (big,) = r[20]
        others = [run for i, rs in r.items() if i != 20 for run in rs]
        self.assertEqual([o for o in others if o[0] == big[0] and overlaps(o, big)], [])
        self.assertTrue([o for o in others if o[0] != big[0] and overlaps(o, big)])
        # the idle slots' memory was really freed before the big map started
        self.assertTrue([p for p, _ in releases(fake, big[0]) if p < big[1]])
        self.assertLessEqual(fake.peak_alloc[big[0]], 9 * GiB)

    def test_map_bigger_than_budget_still_attempted_alone(self):
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        params = maps(8, pixel=GiB, time=0.05)
        params[2].update(pixel=20 * GiB, time=0.1)
        self.assert_successes(collect(sched, params), 8)
        r = runs(fake)
        (big,) = r[2]
        self.assertEqual([o for i, rs in r.items() if i != 2 for o in rs if overlaps(o, big)], [])
        self.assertEqual(sched.stats["devices"][0]["peak_committed"], 20 * GiB)

    def test_slow_release_blocks_admission_on_its_gpu(self):
        # map 2 runs alone after maps 0 and 1; while their slots are still being freed (0.2 s each), maps 3 and
        # 4 must not start on top of memory that is not free yet
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, release_time=0.2)
        sched = self.make(fake, maps_per_gpu=4)
        params = [{"index": 0, "pixel": 6 * GiB, "time": 0.1},
                  {"index": 1, "pixel": 0, "star": 3 * GiB, "time": 0.1},
                  {"index": 2, "pixel": 0, "tree": 2 * GiB, "time": 0.4},
                  {"index": 3, "pixel": 3 * GiB, "time": 0.4},
                  {"index": 4, "pixel": 3 * GiB, "time": 0.4}]
        self.assert_successes(collect(sched, params), 5)
        dev = sched.stats["devices"][0]
        self.assertEqual(dev["budget"], 9 * GiB)
        self.assertLessEqual(dev["peak_committed"], dev["budget"])
        self.assertLessEqual(fake.peak_alloc[0], dev["budget"])
        r = runs(fake)
        (two,) = r[2]
        freed = [p for p, _ in releases(fake, 0) if p < two[1]]
        self.assertEqual(len(freed), 2)
        self.assertTrue(all(r[i][0][1] > max(freed) for i in (3, 4)))

    def test_grow_only_slot_gives_memory_back_after_a_big_map(self):
        # real slots keep their buffers: without giving them back, the slot that ran the big map would keep
        # re-admitting small maps alone and the GPU would stay serial for the rest of the batch
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True)
        sched = self.make(fake)
        params = [{"index": 0, "pixel": int(8.5 * GiB), "time": 0.2}] + \
                 [{"index": i, "pixel": GiB, "time": 0.05} for i in range(1, 41)]
        self.assert_successes(collect(sched, params), 41)
        (big,) = runs(fake)[0]
        self.assertTrue(releases(fake, 0, after=big[2]))
        self.assertGreaterEqual(max_running_after(fake, big[2], 0), 2)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    def test_slot_holding_more_than_needed_shrinks_while_others_run(self):
        # a 6 GiB map runs next to three 1 GiB maps (9 GiB budget) while four slots wait for memory. Small maps
        # keep running, so the run-alone release never fires: the slot that ran the big map has to give its
        # memory back while the others run, so that more than 4 maps run at once afterwards
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True)
        sched = self.make(fake)
        params = [{"index": 0, "pixel": 6 * GiB, "time": 0.3}] + \
                 [{"index": i, "pixel": GiB, "time": 0.05} for i in range(1, 61)]
        self.assert_successes(collect(sched, params), 61)
        r = runs(fake)
        (big,) = r[0]
        others = [o for i, rs in r.items() if i != 0 for o in rs]
        shrinks = [p for p, slot_id in releases(fake, 0, after=big[2]) if slot_id == big[3]]
        self.assertTrue(shrinks)
        self.assertTrue([o for o in others if o[1] < shrinks[0] < o[2]])
        self.assertGreaterEqual(max_running_after(fake, shrinks[0], 0), 6)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    # ---- running out of memory

    def test_oom_with_others_running_requeues_and_parks(self):
        # GPU 0 runs out of memory with more than 2 maps in progress. Each map that does frees its slot and goes
        # back to the front of the queue, and GPU 0 parks one slot each time
        fake = FakeBackend(oom_above={0: 2})
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(30, time=0.05)), 30)
        dev = sched.stats["devices"][0]
        ooms = events(fake, "oom", 0)
        self.assertGreaterEqual(dev["oom_retries"], 1)
        self.assertEqual(dev["oom_retries"], len(ooms))
        # an error needs 3 maps in progress, so every error parked at least one slot and GPU 0 kept at least 2
        self.assertGreaterEqual(dev["parked_slots"], dev["oom_retries"])
        self.assertLessEqual(dev["parked_slots"], 6)
        self.assertEqual(sched.stats["devices"][1]["oom_retries"], 0)
        self.assertLessEqual(max_running_after(fake, ooms[0][0], 0), 2)
        # the slot freed its memory before it did anything else
        log = fake.log_snapshot()
        for pos, slot_id in ooms:
            nxt = [e[0] for e in log[pos + 1:] if e[2] == slot_id and e[0] in ("start", "oom", "release")]
            self.assertEqual(nxt[0], "release", (pos, slot_id))

    def test_oom_retry_waits_for_fewer_maps_in_progress(self):
        # one GPU of 8 slots runs just 3 maps when map 2 runs out of memory. Its limit drops to 2 (not 7), so the
        # retry starts only once one of maps 0 and 1 has ended, never while both still run
        tries_of_2 = []
        always = []

        def fail(p):
            if p["index"] != 2:
                return None
            tries_of_2.append(p["index"])
            if always or len(tries_of_2) == 1:
                time.sleep(0.02)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail)
        sched = self.make(fake)
        params = [{"index": 0, "time": 0.2}, {"index": 1, "time": 0.4}, {"index": 2, "time": 0.05}]
        self.assert_successes(collect(sched, params), 3)
        dev = sched.stats["devices"][0]
        self.assertEqual((dev["oom_retries"], dev["parked_slots"]), (1, 6))
        r = runs(fake)
        first, retry = r[2]
        alongside = [r[i][0] for i in (0, 1)]
        self.assertTrue(all(overlaps(o, first) for o in alongside))
        still_running = [o for o in alongside if o[1] < retry[1] < o[2]]
        self.assertEqual(len(still_running), 1)   # map 0 had ended, map 1 still ran
        self.assertLess(r[0][0][2], retry[1])
        # when map 2 always runs out, it fails alone in the end and the GPU gets back all 6 + 1 parked slots
        always.append(True)
        out = collect(sched, params)
        self.assert_successes(out, 3, failed={2})
        dev = sched.stats["devices"][0]
        self.assertGreaterEqual(dev["oom_retries"], 2)
        self.assertEqual(dev["parked_slots"], 0)

    def test_oom_alone_fails_map(self):
        # map 3 runs out of memory however few maps run next to it: it is retried with fewer maps in progress
        # until it runs alone with no memory held by the idle slots, and only then fails. Its GPU then gets
        # back the slots its retries parked
        fake, seen = self.always_oom(devices=1, index=3)
        sched = self.make(fake)
        out = collect(sched, maps(20, time=0.05))
        self.assert_successes(out, 20, failed={3})
        (err,) = [f.result for f in out if f.index == 3]
        self.assertEqual((err.index, err.cuda_error, str(err)), (3, 2, OOM_ALONE))
        self.assertIsInstance(err.__cause__, MapFailed)
        r = runs(fake)
        tries = r[3]
        dev = sched.stats["devices"][0]
        self.assertGreaterEqual(len(tries), 3)
        self.assertEqual(dev["oom_retries"], len(tries) - 1)
        self.assertEqual(dev["parked_slots"], 0)
        self.assertEqual([o for i, rs in r.items() if i != 3 for o in rs if overlaps(o, tries[-1])], [])
        self.assertEqual(seen[-1], MiB)   # the last try had the GPU to itself: only its own 1 MiB allocated
        self.assertGreater(seen[-2], MiB)   # the try before had idle slots holding memory, which was freed
        # every try freed its own slot
        for _, _, end, slot_id in tries:
            self.assertTrue([p for p, s in releases(fake, 0, after=end) if s == slot_id])

    def test_oom_on_every_run_fails_every_map_and_stops(self):
        # no map fits, and each one takes 20 ms to find out: every map is retried while others run, then fails
        # alone, and the batch still ends
        def oom(p):
            time.sleep(0.02)
            return MapFailed("out of memory", 2)

        fake = FakeBackend(devices=1, fail_run_when=oom)
        sched = self.make(fake)
        out = collect(sched, maps(12))
        self.assert_successes(out, 12, failed=set(range(12)))
        for f in out:
            self.assertEqual((f.result.cuda_error, str(f.result)), (2, OOM_ALONE))
        dev = sched.stats["devices"][0]
        r = runs(fake)
        self.assertGreaterEqual(dev["oom_retries"], 1)
        self.assertEqual(dev["oom_retries"], sum(len(rs) for rs in r.values()) - 12)
        self.assertEqual(dev["parked_slots"], 0)   # each map gave back the slots its retries parked
        everything = [o for rs in r.values() for o in rs]
        for i, rs in r.items():   # each map failed on a try that ran alone
            self.assertEqual([o for o in everything if o is not rs[-1] and overlaps(o, rs[-1])], [], i)

    def test_always_oom_map_stays_on_its_gpu_and_gives_slots_back(self):
        # map 15 always runs out of memory on 2 GPUs of 8 slots. It is the 16th map started, so it first runs next
        # to 7 others on its GPU. Its retries stay on that GPU, the other GPU keeps all 8 slots, and the first GPU
        # gets its parked slots back once map 15 fails
        fake, seen = self.always_oom(index=15)
        sched = self.make(fake)
        out = collect(sched, maps(80, time=0.05))
        self.assert_successes(out, 80, failed={15})
        tries = runs(fake)[15]
        (d,) = {t[0] for t in tries}   # every try on the same GPU
        other = 1 - d
        (err,) = [f.result for f in out if f.index == 15]
        self.assertEqual((err.cuda_error, str(err)), (2, f"out of device memory with no other map on GPU {d}"))
        self.assertGreaterEqual(len(tries), 3)
        dev, odev = sched.stats["devices"][d], sched.stats["devices"][other]
        self.assertEqual(dev["oom_retries"], len(tries) - 1)
        self.assertEqual(dev["parked_slots"], 0)
        self.assertEqual((odev["oom_retries"], odev["parked_slots"]), (0, 0))
        self.assertEqual(max_running_after(fake, tries[0][1], other), 8)
        self.assertEqual(seen[-1], MiB)
        # with its slots back, the first GPU runs 8 maps at once again
        self.assertEqual(max_running_after(fake, tries[-1][2], d), 8)

    def test_oom_retry_moves_when_its_gpu_breaks(self):
        # a map runs out of memory on GPU 1 next to the 2 maps running there, and its retry waits for GPU 1.
        # Those 2 maps then break GPU 1: the waiting retry runs on GPU 0
        def fail(p):
            if threading.current_thread().name.startswith("ipm-batch-1-"):
                time.sleep(0.05)
                fake.device_errors[1] = 700
                return MapFailed("illegal address", 700)
            return None

        fake = FakeBackend(oom_above={1: 2}, fail_run_when=fail)
        sched = self.make(fake)
        out = collect(sched, maps(40, time=0.05))
        errors = [f for f in out if isinstance(f.result, BatchMapError)]
        self.assertEqual([str(f.result) for f in errors], [gpu_failed(1)])
        self.assert_successes(out, 40, failed={errors[0].index})
        retried = {e[3] for e in fake.log_snapshot() if e[0] == "oom"}
        self.assertEqual(len(retried), 1)   # the retry at the front kept GPU 1 from taking other maps
        self.assertNotIn(errors[0].index, retried)
        self.assertEqual({f.result.device for f in out if f.index in retried}, {0})
        self.assertEqual(len(events(fake, "start", 1)), 2)
        self.assertTrue(sched.stats["devices"][1]["dead"])

    def test_oom_retry_moves_when_its_gpu_has_no_usable_slot(self):
        # both slots of GPU 1 run out of memory and then cannot free it. GPU 1 still works but has no slot left,
        # so the two retries run on GPU 0
        def fail(p):
            if threading.current_thread().name.startswith("ipm-batch-1-"):
                time.sleep(0.05)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(fail_run_when=fail,
                           fail_release_when=lambda s: RuntimeError("cannot free") if s.device == 1 else None)
        sched = self.make(fake, maps_per_gpu=2)
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, maps(8, time=0.05))
        self.assert_successes(out, 8)
        self.assertEqual({f.result.device for f in out}, {0})
        self.assertEqual(len(events(fake, "release_failed", 1)), 2)
        self.assertFalse(sched.stats["devices"][1]["dead"])

    def test_oom_alone_frees_idle_slots_before_failing(self):
        # grow-only slots keep 2 GiB each after a first batch. A map whose estimate fits next to that memory
        # runs alone and runs out of memory while the other slots still hold it: the idle slots' memory is freed
        # and the map tried again, alone, before it can fail
        def fail(p):
            if p.get("oom") == "crowded":
                with fake.lock:
                    crowded = fake.allocated(0) > 3 * GiB
                return MapFailed("out of memory", 2) if crowded else None
            return MapFailed("out of memory", 2) if p.get("oom") == "always" else None

        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True, fail_run_when=fail)
        sched = self.make(fake, maps_per_gpu=4)
        warm = [{"index": i, "pixel": 2 * GiB, "time": 0.1} for i in range(4)]

        def warm_up():
            self.assert_successes(collect(sched, warm), 4)
            self.assertEqual(len({runs(fake)[i][-1][3] for i in range(4)}), 4)
            with fake.lock:
                self.assertEqual(fake.allocated(0), 8 * GiB)

        warm_up()
        (f,) = collect(sched, [{"index": 10, "pixel": GiB, "oom": "crowded"}])
        self.assertIsInstance(f.result, FakeResult)
        tries = runs(fake)[10]
        self.assertEqual(len(tries), 2)
        freed = {s for p, s in releases(fake, 0, after=tries[0][2]) if p < tries[1][1]}
        self.assertEqual(len(freed), 4)   # its own slot and the 3 idle ones
        self.assertEqual((sched.stats["devices"][0]["oom_retries"], sched.stats["devices"][0]["parked_slots"]), (1, 0))

        warm_up()
        (f,) = collect(sched, [{"index": 11, "pixel": GiB, "oom": "always"}])
        self.assertEqual((f.result.cuda_error, str(f.result)), (2, OOM_ALONE))
        tries = runs(fake)[11]
        self.assertEqual(len(tries), 2)   # once with the idle slots' memory, once without
        freed = {s for p, s in releases(fake, 0, after=tries[0][2]) if p < tries[1][1]}
        self.assertEqual(len(freed), 4)

    def test_check_device_blocks_admission_on_its_gpu(self):
        # map 0 fails with a CUDA error after 0.05 s and checking its GPU takes 0.3 s. Map 1 ends during the
        # check, but no map starts on that GPU until the check is over
        def fail(p):
            if p["index"] == 0:
                time.sleep(0.05)
                return MapFailed("invalid argument", 1)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail, check_time=0.3)
        sched = self.make(fake, maps_per_gpu=2)
        params = [{"index": 0}, {"index": 1, "time": 0.2}] + [{"index": i, "time": 0.02} for i in range(2, 8)]
        self.assert_successes(collect(sched, params), 8, failed={0})
        ((began, _),) = events(fake, "check")
        ((ended, _),) = events(fake, "checked")
        self.assertTrue(began < runs(fake)[1][0][2] < ended)
        self.assertEqual([p for p, _ in events(fake, "start") if began < p < ended], [])
        self.assertFalse(sched.stats["devices"][0]["dead"])

    def test_release_when_running_alone(self):
        # four slots hold 2 GiB each after maps 0-3. Map 4 needs 6 GiB of a 9 GiB budget, so it waits until
        # nothing runs; then every idle slot's memory is freed before it starts
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB)
        sched = self.make(fake, maps_per_gpu=4)
        params = maps(4, pixel=2 * GiB, time=0.1) + [{"index": 4, "pixel": 6 * GiB, "time": 0.1}] + \
            [{"index": i, "pixel": GiB, "time": 0.05} for i in range(5, 8)]
        self.assert_successes(collect(sched, params), 8)
        r = runs(fake)
        (big,) = r[4]
        holders = sorted(r[i][0][3] for i in range(4))
        self.assertEqual(len(set(holders)), 4)
        self.assertTrue(all(r[i][0][2] < big[1] for i in range(4)))
        self.assertEqual(sorted(slot_id for pos, slot_id in releases(fake, 0) if pos < big[1]), holders)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    # ---- broken GPUs

    def test_dead_device_moves_maps(self):
        # GPU 1 breaks at its 4th map. The map that reports the error first fails; the others in progress on
        # GPU 1 go back to the queue and run on GPU 0, and GPU 1 is not used again
        fake = FakeBackend(dead_after={1: 3},
                           fail_close_when=lambda s: RuntimeError("broken") if s.device == 1 else None)
        sched = self.make(fake)
        out = collect(sched, maps(40, time=0.05))
        errors = [f for f in out if isinstance(f.result, BatchMapError)]
        self.assertEqual(len(errors), 1)
        trigger = errors[0]
        self.assertEqual((trigger.result.cuda_error, str(trigger.result)), (700, gpu_failed(1)))
        self.assert_successes(out, 40, failed={trigger.index})
        ((death, _),) = events(fake, "dead")
        r = runs(fake)
        for f in out:
            if f.index != trigger.index and f.result.device != 0:
                (only,) = r[f.index]
                self.assertLess(only[2], death, f.index)   # succeeded on GPU 1 only before it broke
        self.assertTrue(all(len(rs) <= 2 for rs in r.values()))   # moved at most once
        late = [slot_id for pos, slot_id in events(fake, "start", 1) if pos > death]
        self.assertEqual(len(late), len(set(late)))   # each slot of GPU 1 started at most one map after it broke
        self.assertTrue(sched.stats["devices"][1]["dead"])
        self.assertFalse(sched.stats["devices"][0]["dead"])
        self.assertEqual(sum(d["maps"] for d in sched.stats["devices"].values()), 39)
        # closing a slot of the broken GPU raises: close() warns and closes every slot once
        with self.assertWarns(RuntimeWarning):
            sched.close()
        self.assertEqual([s.close_calls for s in fake.slots], [1] * len(fake.slots))

    def test_all_devices_dead_fails_remaining(self):
        fake = FakeBackend(devices=1, dead_after={0: 2})
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.3))
        self.assert_successes(out, 10, failed=set(range(10)))
        messages = [str(f.result) for f in out]
        self.assertEqual(messages.count(gpu_failed(0)), 1)
        self.assertEqual(messages.count("no usable GPU left: " + gpu_failed(0)), 9)
        self.assertEqual({f.result.cuda_error for f in out}, {700})
        self.assertTrue(sched.stats["devices"][0]["dead"])
        # the GPU stays out: a later batch fails at once without running anything
        started = len(events(fake, "start"))
        out = collect(sched, maps(3))
        self.assert_successes(out, 3, failed={0, 1, 2})
        self.assertEqual({(str(f.result), f.result.cuda_error) for f in out},
                         {("no usable GPU left: " + gpu_failed(0), 700)})
        self.assertEqual(len(events(fake, "start")), started)
        self.assertTrue(sched.stats["devices"][0]["dead"])

    def test_open_slot_failure_marks_device_dead(self):
        fake = FakeBackend(open_fails_for=(1,))
        sched = self.make(fake)
        out = collect(sched, maps(20, time=0.02))
        self.assert_successes(out, 20)
        self.assertEqual({f.result.device for f in out}, {0})
        self.assertEqual(events(fake, "start", 1), [])
        self.assertTrue(all(len(rs) == 1 for rs in runs(fake).values()))
        self.assertTrue(sched.stats["devices"][1]["dead"])
        self.assertEqual(sched.stats["devices"][0]["maps"], 20)
        tries = len(events(fake, "open_failed"))
        self.assertGreaterEqual(tries, 1)
        self.assertLessEqual(tries, 8)
        # GPU 1 stays out of the batch: the next call does not try to open it again
        out = collect(sched, maps(10, time=0.02))
        self.assert_successes(out, 10)
        self.assertEqual({f.result.device for f in out}, {0})
        self.assertEqual(len(events(fake, "open_failed")), tries)
        self.assertTrue(sched.stats["devices"][1]["dead"])

    def test_map_moves_once_then_fails(self):
        # one slot per GPU and no GPU can open one: the map moves from the first GPU to the second, then fails
        # with the second GPU's error
        fake = FakeBackend(open_fails_for=(0, 1))
        sched = self.make(fake, maps_per_gpu=1)
        (f,) = collect(sched, maps(1))
        tried = events(fake, "open_failed")
        self.assertEqual(len(tried), 2)
        devices = [fake.log_snapshot()[pos][1] for pos, _ in tried]
        self.assertEqual(sorted(devices), [0, 1])
        self.assertIsInstance(f.result, BatchMapError)
        self.assertEqual((f.result.cuda_error, str(f.result)),
                         (0, f"GPU {devices[1]} failed (0): cannot open a slot on fake GPU {devices[1]}"))
        self.assertTrue(all(sched.stats["devices"][d]["dead"] for d in (0, 1)))

    def test_every_map_moves_at_most_once_when_every_gpu_breaks(self):
        fake = FakeBackend(fail_run_when=lambda p: MapFailed("illegal address", 700), device_errors={0: 700, 1: 700})
        sched = self.make(fake, maps_per_gpu=2)
        out = collect(sched, maps(12, time=0.05))
        self.assert_successes(out, 12, failed=set(range(12)))
        self.assertTrue(all(len(rs) <= 2 for rs in runs(fake).values()))
        broken = {gpu_failed(0), gpu_failed(1)}
        firsts = set()
        for f in out:
            message = str(f.result)
            self.assertEqual(f.result.cuda_error, 700)
            if message.startswith("no usable GPU left: "):
                firsts.add(message[len("no usable GPU left: "):])
            else:
                self.assertIn(message, broken)
        self.assertLessEqual(len(firsts), 1)
        self.assertTrue(firsts <= broken)
        self.assertTrue(all(sched.stats["devices"][d]["dead"] for d in (0, 1)))

    def test_cuda_error_on_a_working_gpu_fails_only_its_map(self):
        fake = FakeBackend(fail_run_when=lambda p: MapFailed("invalid argument", 1) if p["index"] == 3 else None)
        sched = self.make(fake)
        out = collect(sched, maps(20, time=0.01))
        self.assert_successes(out, 20, failed={3})
        (err,) = [f.result for f in out if f.index == 3]
        self.assertEqual((err.cuda_error, str(err)), (1, "invalid argument"))
        self.assertEqual(len(events(fake, "check")), 1)
        self.assertFalse(any(d["dead"] for d in sched.stats["devices"].values()))

    # ---- release() failures

    def test_failed_release_keeps_memory_counted(self):
        # slot 0 cannot free its 2 GiB when map 4 has to run alone. The GPU still works, so slot 0 is not used
        # again and its memory still counts: the 3 GiB maps after map 4 must not run on top of it
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB,
                           fail_release_when=lambda s: RuntimeError("cannot free") if s.slot_id == 0 else None)
        sched = self.make(fake, maps_per_gpu=4)
        params = maps(4, pixel=2 * GiB, time=0.1) + [{"index": 4, "pixel": 6 * GiB, "time": 0.1}] + \
            [{"index": i, "pixel": 3 * GiB, "time": 0.05} for i in range(5, 11)]
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, params)
        self.assert_successes(out, 11)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)
        ((failed_at, _),) = events(fake, "release_failed")
        self.assertTrue([p for p, _ in events(fake, "check", 0) if p > failed_at])
        self.assertEqual([p for p, slot_id in events(fake, "start") if slot_id == 0 and p > failed_at], [])
        self.assertFalse(sched.stats["devices"][0]["dead"])

    def test_failed_release_on_a_broken_gpu_takes_it_out(self):
        # a slot of GPU 1 runs out of memory and cannot free it because GPU 1's context broke: GPU 1 leaves the
        # batch and its map runs on GPU 0. Maps already in progress on GPU 1 may still finish there
        fake = FakeBackend(oom_above={1: 2})

        def fail(slot):
            if slot.device != 1:
                return None
            fake.device_errors[1] = 700
            return MapFailed("illegal address", 700)

        fake.fail_release_when = fail
        sched = self.make(fake)
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, maps(30, time=0.05))
        self.assert_successes(out, 30)
        self.assertTrue(sched.stats["devices"][1]["dead"])
        self.assertEqual(sched.stats["devices"][1]["oom_retries"], 0)
        failed_at = events(fake, "release_failed", 1)[0][0]
        late = [slot_id for pos, slot_id in events(fake, "start", 1) if pos > failed_at]
        self.assertEqual(len(late), len(set(late)))
        moved = {e[3] for e in fake.log_snapshot() if e[0] == "oom"}
        self.assertTrue(moved)
        self.assertTrue(all(f.result.device == 0 for f in out if f.index in moved))

    # ---- other failures

    def test_estimate_failure_reported_and_batch_continues(self):
        threads = []

        def fail(p):
            threads.append(threading.current_thread())
            return MapFailed("bad estimate", 1) if p["index"] == 2 else None

        fake = FakeBackend(fail_estimate_when=fail)
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.01))
        self.assert_successes(out, 10, failed={2})
        (err,) = [f.result for f in out if f.index == 2]
        self.assertEqual((err.index, err.cuda_error, str(err)), (2, 1, "bad estimate"))
        self.assertEqual(set(threads), {threading.current_thread()})
        log = fake.log_snapshot()
        first_start = min(pos for pos, e in enumerate(log) if e[0] == "start")
        self.assertTrue(all(pos < first_start for pos, e in enumerate(log) if e[0] == "estimate"))
        self.assertNotIn(2, runs(fake))

    def test_run_failure_reported_and_batch_continues(self):
        fake = FakeBackend(fail_run_when=lambda p: MapFailed("bad", 0) if p["index"] == 3 else None)
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.01))
        self.assert_successes(out, 10, failed={3})
        (err,) = [f.result for f in out if f.index == 3]
        self.assertEqual((err.index, err.cuda_error, str(err)), (3, 0, "bad"))
        self.assertEqual(sum(d["maps"] for d in sched.stats["devices"].values()), 9)

    def test_unexpected_exception_becomes_batch_map_error(self):
        fake = FakeBackend(fail_run_when=lambda p: RuntimeError("boom") if p["index"] == 4 else None)
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.01))
        self.assert_successes(out, 10, failed={4})
        (err,) = [f.result for f in out if f.index == 4]
        self.assertEqual((err.index, err.cuda_error, str(err)), (4, 0, "boom"))
        self.assertIsInstance(err.__cause__, RuntimeError)

    def test_break_stops_cleanly(self):
        fake = FakeBackend()
        sched = self.make(fake)
        gen = sched.results(maps(50, time=0.05))
        for _ in range(2):
            next(gen).release()
        with self.assertRaises(RuntimeError):
            next(sched.results(maps(1)))
        gen.close()
        self.assert_stopped_cleanly(fake, sched, 50)

    def test_exception_in_consumer_stops_cleanly(self):
        fake = FakeBackend()
        sched = self.make(fake)
        with self.assertRaises(ValueError):
            for k, f in enumerate(sched.results(maps(50, time=0.05))):
                f.release()
                if k == 1:
                    raise ValueError("consumer failed")
        self.assert_stopped_cleanly(fake, sched, 50)

    def test_break_while_results_wait_in_slots_stops_cleanly(self):
        # a 1-byte limit lets one 1 GiB result wait at a time: the consumer stops after the first map while the
        # other 3 finished maps still wait in their slots for room in the queue
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=4, max_result_bytes=1)
        gen = sched.results(maps(10, result=GiB, time=0.05))
        f = next(gen)
        time.sleep(0.2)
        self.assertEqual(len(events(fake, "end")), 4)
        self.assertEqual(sched.stats["peak_result_bytes"], GiB)
        self.assertEqual(sched.stats["peak_in_flight_result_bytes"], 4 * GiB)
        gen.close()
        f.release()
        self.assert_stopped_cleanly(fake, sched, 10)

    def test_interrupt_while_waiting_stops_cleanly(self):
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=2)
        timer = threading.Timer(0.3, _thread.interrupt_main)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(KeyboardInterrupt):
            for f in sched.results(maps(6, time=1.0)):
                f.release()
        # delivered while the first two maps ran, so no further map started
        self.assertEqual(len([e for e in fake.log_snapshot() if e[0] == "start"]), 2)
        self.assert_stopped_cleanly(fake, sched, 6)

    def test_slots_persist_between_runs(self):
        fake = FakeBackend()
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(16, time=0.05)), 16)
        opened = fake.open_calls
        self.assertEqual(opened, 4)
        self.assert_successes(collect(sched, maps(16, time=0.05)), 16)
        self.assertEqual(fake.open_calls, opened)
        sched.close()
        sched.close()
        self.assertEqual([s.close_calls for s in fake.slots], [1] * opened)
        with self.assertRaises(RuntimeError):
            next(sched.results(maps(1)))

    def test_held_memory_counts_against_budget(self):
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(8, pixel=4 * GiB, time=0.05)), 8)
        self.assert_successes(collect(sched, maps(24, pixel=GiB, time=0.05)), 24)
        for d in (0, 1):
            dev = sched.stats["devices"][d]
            self.assertEqual(dev["budget"], 9 * GiB)
            self.assertGreaterEqual(dev["peak_committed"], 8 * GiB)
            self.assertLessEqual(dev["peak_committed"], dev["budget"])
            self.assertLessEqual(fake.peak_alloc[d], dev["budget"])


if __name__ == "__main__":
    unittest.main()
