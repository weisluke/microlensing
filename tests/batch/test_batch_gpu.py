"""GPU tests of batches run through the ctypes backend (skipped without a GPU): batched maps against serial
IPM.run() maps, save mode against IPM.save(), the memory estimates, every visible GPU, clean stops and the host
memory of save mode. Uses only the GPUs Slurm allocated (setUpModule checks CUDA_VISIBLE_DEVICES):
python -m unittest tests.batch.test_batch_gpu -v

Stars of two runs are compared after sorting by (x1, x2, mass), as their order after the tree sort is not
reproducible. Maps are judged by batch_rules against the noise between serial runs of the same map (the larger
of two serial runs' differences from a third, the reference)."""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu"))

import batch_rules  # noqa: E402
import bench_lib  # noqa: E402
import compare_maps  # noqa: E402
from microlensing.IPM import lib_ipm  # noqa: E402
from microlensing.IPM.batch import BatchedIPM  # noqa: E402
from microlensing.IPM.batch_backend import CtypesBackend  # noqa: E402
from microlensing.IPM.batch_params import expand  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError  # noqa: E402
from microlensing.IPM.ipm import IPM  # noqa: E402

LENS = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2)
MAP_NAMES = ("magnifications", "magnifications_minima", "magnifications_saddles")
MiB = 1 << 20
SERIAL_RUNS = 3   # per configuration: the reference and two runs whose larger difference from it is the noise
HIST_FACTOR = 3   # a histogram's L1 difference from the reference may be 3 times the serial runs' plus 1e-3 per pixel

TMP = None        # the module's scratch directory
STARFILE = None   # a star file written by a serial IPM run (setUpModule)


def nb(n, hl=10.0, **extra):
    """The nb lens on an n x n map with half-lengths hl."""
    return dict(LENS, half_length_y1=hl, half_length_y2=hl, num_pixels_y1=n, num_pixels_y2=n, **extra)


SALPETER = dict(rectangular=True, approx=True, mass_function="Salpeter", m_solar=0.5, m_lower=0.08, m_upper=1.5)


def star_file_map():
    """A map whose stars come from STARFILE (a field written for half-lengths 10, used at 8)."""
    return dict(kappa_tot=0.4, shear=0.4, half_length_y1=8.0, half_length_y2=8.0, num_pixels_y1=700,
                num_pixels_y2=700, starfile=STARFILE)


def device_count():
    return lib_ipm.lib.get_device_count() if hasattr(lib_ipm.lib, "get_device_count") else 0


def setUpModule():
    if device_count() == 0:
        raise unittest.SkipTest("no GPU visible")
    problem = bench_lib.check_all_gpus_allocated()
    if problem is not None:
        raise RuntimeError(problem)
    global TMP, STARFILE
    TMP = tempfile.mkdtemp(prefix="ipm_batch_gpu_")
    ipm = IPM(**nb(600), random_seed=77, write_stars=True, outfile_prefix=TMP + "/sf_")
    ipm.run()
    ipm.save()
    STARFILE = TMP + "/sf_ipm_stars.bin"
    if not Path(STARFILE).is_file():
        raise RuntimeError(f"the serial run wrote no star file {STARFILE}")


def tearDownModule():
    if TMP is not None:
        shutil.rmtree(TMP, ignore_errors=True)


def batch_of(maps, **batch_kw):
    """A BatchedIPM of the given per-map dicts (keys a map leaves out are None, i.e. IPM's default), checked to
    use the ctypes backend rather than the one-at-a-time fallback."""
    keys = sorted(set().union(*maps))
    seqs = {k: [m.get(k) for m in maps] for k in keys}
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # the fallback warns that the library has no batch support
        b = BatchedIPM(**seqs, **batch_kw)
    assert isinstance(b._backend, CtypesBackend), type(b._backend)
    return b


def serial(params, is_double):
    ipm = IPM(**params, is_double=is_double)
    ipm.run()
    return ipm


def sorted_stars(arr):
    arr = np.asarray(arr)
    return arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]


def batch_threads():
    return [t.name for t in threading.enumerate() if t.name.startswith("ipm-batch-")]


def device_free(d):
    out = (ctypes.c_ulonglong * 2)()
    cm = ctypes.c_int(0)
    assert lib_ipm.lib.get_device_info(d, out, ctypes.byref(cm))
    return out[0]


def gpu_processes():
    """({CUDA device: MiB this process uses}, {CUDA device: MiB other processes use}) from nvidia-smi, for the
    devices of a numeric CUDA_VISIBLE_DEVICES (CUDA_DEVICE_ORDER=PCI_BUS_ID); (None, None) if unavailable."""
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not visible or not all(x.isdigit() for x in visible) or shutil.which("nvidia-smi") is None:
        return None, None
    buses = {}
    for line in bench_lib.nvidia_smi("index,pci.bus_id").splitlines():
        index, bus = (x.strip() for x in line.split(","))
        buses[bus.lower()] = index
    cuda_of = {idx: d for d, idx in enumerate(visible)}
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    mine, others = {}, {}
    for line in out.splitlines():
        pid, bus, used = (x.strip() for x in line.split(","))
        d = cuda_of.get(buses.get(bus.lower()))
        if d is not None and used.isdigit():
            target = mine if int(pid) == os.getpid() else others
            target[d] = target.get(d, 0) + int(used)
    return mine, others


def _fmt(m):
    return " ".join(f"{k}={m[k]:.2e}" for k in ("mean", "p999", "p9999", "max", "sum_ratio"))


class _Equivalence:
    """Checks of batched results against serial runs of the same parameters."""

    def serial_refs(self, batch, is_double):
        """SERIAL_RUNS serial IPM.run() objects per map, shared by maps whose parameters differ only in their
        prefix: the first is the reference, the others give the noise."""
        refs, cache = [], {}
        for i in range(batch.num_maps):
            p = batch.params(i)
            key = repr(sorted((k, v) for k, v in p.items() if k != "outfile_prefix"))
            if key not in cache:
                cache[key] = tuple(serial(p, is_double) for _ in range(SERIAL_RUNS))
            refs.append(cache[key])
        return refs

    def assert_matches(self, label, params, result, refs, is_double):
        ref_a = refs[0]
        self.assertNotIsInstance(result, BatchMapError, f"{label}: {result}")
        self.assertTrue(result._host_only, label)
        for ref in refs[1:]:
            self.assertTrue(np.array_equal(sorted_stars(ref_a.stars.stars), sorted_stars(ref.stars.stars)),
                            f"{label}: the serial runs differ in their stars")
        self.assertEqual(result.num_stars, ref_a.num_stars, label)
        self.assertTrue(np.array_equal(sorted_stars(result.stars.stars), sorted_stars(ref_a.stars.stars)),
                        f"{label}: stars differ from the serial run")
        self.assertEqual(result.corner, ref_a.corner, label)
        expected_seed = 0 if params["starfile"] is not None else params["random_seed"]
        self.assertEqual(result.random_seed, expected_seed, label)
        self.assertEqual(ref_a.random_seed, expected_seed, label)
        for name in MAP_NAMES:
            a, c = getattr(ref_a, name), getattr(result, name)
            if a is None:
                self.assertIsNone(c, f"{label}: {name}")
                continue
            self.assertEqual(c.dtype, a.dtype, f"{label}: {name}")
            self.assertEqual(c.shape, a.shape, f"{label}: {name}")
            noise = batch_rules.noise(a, *(getattr(ref, name) for ref in refs[1:]))
            fails = batch_rules.judge(c, a, noise, is_double, parity=name != "magnifications")
            print(f"  {label} {name}: noise {_fmt(noise)} | batch {_fmt(compare_maps.map_metrics(c, a))}")
            self.assertEqual(fails, [], f"{label}: {name}")


class BatchEquivalenceTests(_Equivalence, unittest.TestCase):
    def test_batch_matches_serial_runs(self):
        # float: more maps than slots (2 per GPU), so slots run several different maps; map 5 repeats map 0's seed
        float_maps = [
            nb(1000, random_seed=1001),
            nb(600, random_seed=1002, **SALPETER),
            nb(800, random_seed=1003, write_parities=True),
            nb(600, random_seed=1004, num_rays_y=4),
            star_file_map(),
            nb(1000, random_seed=1001),
        ]
        # double (is_double applies to a whole batch): 4 maps on the 2 slots of GPU 0, with and without parities
        double_maps = [nb(600, random_seed=2001), nb(600, random_seed=2002, write_parities=True),
                       nb(600, random_seed=2003), nb(600, random_seed=2004, write_parities=True)]
        for label, maps, is_double, kw in (("float", float_maps, False, {}),
                                           ("double", double_maps, True, {"devices": [0]})):
            with batch_of(maps, is_double=is_double, maps_per_gpu=2, **kw) as b:
                refs = self.serial_refs(b, is_double)
                out = dict(b.run())
                stats = b.last_run_stats
            print(f"\n{label}: maps per GPU {({d: s['maps'] for d, s in stats['devices'].items()})}, "
                  f"max concurrent {({d: s['max_concurrent'] for d, s in stats['devices'].items()})}")
            self.assertEqual(sorted(out), list(range(len(maps))))
            for i in range(len(maps)):
                with self.subTest(precision=label, map=i):
                    self.assert_matches(f"{label} map {i}", b.params(i), out[i], refs[i], is_double)
            self.assertEqual(stats["direct_deposit_maps"], 0)
            if not is_double:   # equal seeds and parameters give equal star fields within a batch too
                self.assertTrue(np.array_equal(sorted_stars(out[5].stars.stars), sorted_stars(out[0].stars.stars)))

    def test_judge_rejects_another_seed(self):
        # the rule still discriminates: a serial map of the same configuration with another seed fails it
        for label, p, is_double in (("float parities", nb(600, write_parities=True), False),
                                    ("double", nb(600), True)):
            refs = [serial(dict(p, random_seed=9101), is_double) for _ in range(SERIAL_RUNS)]
            other = serial(dict(p, random_seed=9102), is_double)
            for name in MAP_NAMES:
                a = getattr(refs[0], name)
                if a is None:
                    continue
                with self.subTest(case=label, map=name):
                    noise = batch_rules.noise(a, *(getattr(r, name) for r in refs[1:]))
                    fails = batch_rules.judge(getattr(other, name), a, noise, is_double,
                                              parity=name != "magnifications")
                    print(f"  another seed, {label} {name}: {fails}")
                    self.assertNotEqual(fails, [])

    def test_reused_slot_matches_fresh_runs(self):
        # one slot runs every map: Salpeter with m_solar=0.5 (C++ scales m_lower/m_upper in place) alternates with
        # a star-file map (which frees and reallocates the stars) and differs in size from it
        maps = []
        for seed in (3001, 3002, 3003):
            maps += [nb(600, random_seed=seed, **SALPETER), star_file_map()]
        with batch_of(maps, devices=[0], maps_per_gpu=1) as b:
            refs = self.serial_refs(b, False)
            out = dict(b.run())
            stats = b.last_run_stats
        self.assertEqual(stats["devices"][0]["maps"], len(maps))
        self.assertEqual(stats["devices"][0]["max_concurrent"], 1)
        print()
        for i in range(len(maps)):
            with self.subTest(map=i):
                self.assert_matches(f"reused map {i}", b.params(i), out[i], refs[i], False)


def read_star_file(path, itemsize):
    """(header bytes, stars sorted by x1, x2, mass) of a star file: int num_stars, int rectangular, the complex
    corner and theta_star, then x1, x2 and mass of each star."""
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data, dtype=np.int32, count=1)[0])
    head = 8 + 3 * itemsize
    dtype = np.float64 if itemsize == 8 else np.float32
    stars = np.frombuffer(data, dtype=dtype, offset=head)
    if stars.size != 3 * n:
        raise ValueError(f"{path}: {stars.size} values for {n} stars")
    return data[:head], sorted_stars(stars.reshape(n, 3))


def read_histogram(path):
    counts = {}
    for line in Path(path).read_text().splitlines():
        value, count = line.split()
        counts[int(value)] = int(count)
    return counts


def histogram_l1(a, b):
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


class SaveModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="save_", dir=TMP))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_mode_files_match_ipm_save(self):
        flags = dict(write_stars=True, write_maps=True, write_parities=True, write_histograms=True)
        maps = [nb(600, random_seed=4001, **flags), nb(800, random_seed=4002, **flags),
                nb(600, random_seed=4003, **SALPETER, **flags)]
        serial_dirs = "BCD"[:SERIAL_RUNS]   # B is the reference, the others give the noise
        dirs = {k: self.tmp / k for k in "A" + serial_dirs}
        for d in dirs.values():
            d.mkdir()
        with batch_of(maps, outfile_prefix=f"{dirs['A']}/", maps_per_gpu=2) as b:
            recs = b.run(save=True)
            params = [b.params(i) for i in range(len(maps))]
        self.assertEqual([r.index for r in recs], list(range(len(maps))))
        for i, r in enumerate(recs):
            self.assertIsNone(r.error, f"map {i}: {r.error}")
            self.assertEqual(r.random_seed, params[i]["random_seed"])
            self.assertEqual(r.outfile_prefix, f"{dirs['A']}/{i}_")
        for i, p in enumerate(params):
            for k in serial_dirs:
                ipm = IPM(**dict(p, outfile_prefix=f"{dirs[k]}/{i}_"))
                ipm.run()
                ipm.save()
        names = sorted(f.name for f in dirs["A"].iterdir())
        for k in serial_dirs:
            self.assertEqual(names, sorted(f.name for f in dirs[k].iterdir()))
        per_map = 1 + 1 + 3 + 6   # parameter info, stars, 3 maps, 6 histograms
        self.assertEqual(len(names), per_map * len(maps), names)
        print()
        for name in names:
            a, b = dirs["A"] / name, dirs["B"] / name
            others = [dirs[k] / name for k in serial_dirs[1:]]
            i = int(name.split("_", 1)[0])
            npix = params[i]["num_pixels_y1"] * params[i]["num_pixels_y2"]
            with self.subTest(file=name):
                if name.endswith("_ipm_parameter_info.txt"):
                    la, lb = a.read_text().splitlines(), b.read_text().splitlines()
                    self.assertEqual(len(la), len(lb))
                    self.assertEqual([x for x in la if not x.startswith("t_shoot_cells ")],
                                     [x for x in lb if not x.startswith("t_shoot_cells ")])
                    self.assertEqual(sum(x.startswith("t_shoot_cells ") for x in la), 1)
                elif name.endswith("_ipm_stars.bin"):
                    ha, sa = read_star_file(a, 4)
                    hb, sb = read_star_file(b, 4)
                    self.assertEqual(ha, hb)
                    self.assertTrue(np.array_equal(sa, sb))
                elif name.endswith(".bin"):
                    self.assertEqual(a.read_bytes()[:8], b.read_bytes()[:8])
                    self.assertEqual(a.stat().st_size, b.stat().st_size)
                    ref, cand = bench_lib.read_map(b), bench_lib.read_map(a)
                    noise = batch_rules.noise(ref, *(bench_lib.read_map(x) for x in others))
                    parity = not name.endswith("_ipm_magnifications.bin")
                    print(f"  {name}: noise {_fmt(noise)} | batch {_fmt(compare_maps.map_metrics(cand, ref))}")
                    self.assertEqual(batch_rules.judge(cand, ref, noise, False, parity=parity), [])
                elif name.endswith(".txt"):
                    ha, hb = read_histogram(a), read_histogram(b)
                    self.assertEqual(sum(ha.values()), npix)
                    self.assertEqual(sum(hb.values()), npix)
                    # the bins are 1/1000 of a magnification wide, so serial noise alone moves a few per cent of
                    # the pixels into a neighbouring bin: the limit is relative to the serial runs' L1
                    l1 = histogram_l1(ha, hb)
                    l1_noise = max(histogram_l1(read_histogram(x), hb) for x in others)
                    limit = HIST_FACTOR * l1_noise + 1e-3 * npix
                    print(f"  {name}: L1 batch {l1}, L1 serial {l1_noise}, limit {limit:.0f} ({npix} pixels)")
                    self.assertLessEqual(l1, limit)
                else:
                    self.fail(f"unexpected file {name}")

    def test_save_mode_host_memory_flat(self):
        child = (
            "import sys\n"
            "from microlensing.IPM.batch import BatchedIPM\n"
            "n, out = int(sys.argv[1]), sys.argv[2]\n"
            "with BatchedIPM(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=10, half_length_y2=10,\n"
            "                num_pixels_y1=2000, num_pixels_y2=2000, num_maps=n, random_seed=5001,\n"
            "                write_maps=True, outfile_prefix=out + '/') as b:\n"
            "    recs = b.run(save=True)\n"
            "bad = [r.index for r in recs if r.error is not None]\n"
            "if bad or len(recs) != n:\n"
            "    sys.exit(f'failed maps {bad} of {len(recs)}')\n"
            "with open('/proc/self/status') as f:\n"
            "    print('VmHWM_kB', next(l.split()[1] for l in f if l.startswith('VmHWM:')))\n"
        )
        peaks = {}
        for n in (16, 128):
            out = self.tmp / f"n{n}"
            out.mkdir()
            proc = subprocess.run([sys.executable, "-c", child, str(n), str(out)], cwd=str(REPO),
                                  capture_output=True, text=True, timeout=900)
            self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
            self.assertEqual(len(list(out.glob("*_ipm_magnifications.bin"))), n)
            line = next(x for x in proc.stdout.splitlines() if x.startswith("VmHWM_kB"))
            peaks[n] = int(line.split()[1]) / 1024
            shutil.rmtree(out)
        print(f"\nsave mode VmHWM: N=16 {peaks[16]:.0f} MiB, N=128 {peaks[128]:.0f} MiB, "
              f"difference {peaks[128] - peaks[16]:.0f} MiB")
        self.assertLess(abs(peaks[128] - peaks[16]), 512)


class BackendTests(unittest.TestCase):
    def test_open_slot_failure_names_the_gpu_and_cause(self):
        backend = CtypesBackend(False, 0)
        missing = device_count()
        with self.assertRaisesRegex(RuntimeError, f"GPU {missing}: there is no GPU {missing} among the"):
            backend.open_slot(missing)

    def test_released_slot_holds_nothing_and_runs_again(self):
        backend = CtypesBackend(False, 0)
        p = expand(dict(nb(600), random_seed=[9001]))[0]
        est = backend.estimate(p)
        slot = backend.open_slot(0)
        try:
            first = slot.run(p)
            self.assertGreater(first.held.device_bytes(), 0)
            for k in ("pixel", "star", "tree"):
                self.assertLessEqual(getattr(first.held, k), getattr(est, k), k)
            self.assertEqual(first.result_bytes, est.result)   # the maps and stars of a map without parities
            slot.release()
            alloc = (ctypes.c_ulonglong * 3)()
            backend.lib.get_allocated_bytes(slot.ipm.obj, alloc)
            self.assertEqual(list(alloc), [0, 0, 0])
            second = slot.run(p)
            self.assertTrue(np.array_equal(sorted_stars(second.result.stars.stars),
                                           sorted_stars(first.result.stars.stars)))
            self.assertEqual(second.result.random_seed, 9001)
        finally:
            slot.close()
        self.assertIsNone(slot.ipm)


class SchedulingTests(unittest.TestCase):
    def test_estimates_prevent_direct_deposit(self):
        with batch_of([nb(5000, hl=25.0)] * 32, random_seed=6001, devices=[0]) as b:
            for i, r in b.run():
                self.assertNotIsInstance(r, BatchMapError, f"map {i}: {r}")
            stats = b.last_run_stats
        dev = stats["devices"][0]
        print(f"\n5000^2 x 32 on GPU 0: max concurrent {dev['max_concurrent']}, peak committed "
              f"{dev['peak_committed'] / MiB:.0f} MiB of a {dev['budget'] / MiB:.0f} MiB budget, "
              f"direct deposit maps {stats['direct_deposit_maps']}")
        self.assertEqual(dev["maps"], 32)
        self.assertEqual(stats["direct_deposit_maps"], 0)
        self.assertLessEqual(dev["peak_committed"], dev["budget"])
        self.assertGreater(dev["max_concurrent"], 1)

    def test_uses_every_visible_gpu(self):
        n_dev = device_count()
        if n_dev < 2:
            self.skipTest("fewer than 2 GPUs visible")
        with batch_of([nb(600)] * 16, random_seed=7001) as b:
            out = dict(b.run())
            stats = b.last_run_stats
        self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
        per_gpu = {d: s["maps"] for d, s in stats["devices"].items()}
        print(f"\n16 maps on every GPU: {per_gpu}")
        self.assertEqual(sorted(per_gpu), list(range(n_dev)))
        self.assertTrue(all(v > 0 for v in per_gpu.values()), per_gpu)
        self.assertEqual(sum(per_gpu.values()), 16)
        with batch_of([nb(600)] * 4, random_seed=7002, devices=[1]) as b:
            out = dict(b.run())
            stats = b.last_run_stats
        self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
        self.assertEqual(list(stats["devices"]), [1])
        self.assertEqual(stats["devices"][1]["maps"], 4)

    def test_break_then_rerun(self):
        # 2000^2 maps, so that the slots hold some 270 MiB per GPU: a slot that close() left open would show. The
        # driver keeps a cache of device memory that nvidia-smi gives to no process and that moves in steps of
        # about 128 MiB (after earlier tests put managed memory on a GPU, measured 2026-10-22), so the free memory
        # may move by up to 256 MiB. This process's own usage must come back within 64 MiB after close(), and must
        # not grow over 3 further close()/rerun cycles
        free_tolerance = 256 * MiB
        maps = [nb(2000)] * 20
        n_dev = device_count()
        with batch_of(maps[:4], random_seed=8000) as warm:   # each GPU's first batch also loads code onto it
            list(warm.run())
        before = {d: device_free(d) for d in range(n_dev)}
        before_mine, before_others = gpu_processes()
        b = batch_of(maps, random_seed=8001)
        try:
            got = []
            for i, r in b.run():
                self.assertNotIsInstance(r, BatchMapError, f"map {i}: {r}")
                got.append(i)
                if len(got) == 2:
                    break
            self.assertEqual(batch_threads(), [])
            out = dict(b.run())   # the next run of the same batch, on the slots the first one opened
            self.assertEqual(sorted(out), list(range(20)))
            self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
            self.assertEqual(batch_threads(), [])
            holding_mine, _ = gpu_processes()
        finally:
            b.close()
        after = {d: device_free(d) for d in range(n_dev)}
        after_mine, after_others = gpu_processes()
        cycles_mine = []   # this process's usage after each further close()
        for _ in range(3):
            try:
                out = dict(b.run())   # a closed batch opens new slots
                self.assertEqual(sorted(out), list(range(20)))
                self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
            finally:
                b.close()
            self.assertEqual(batch_threads(), [])
            cycles_mine.append(gpu_processes()[0])
        print(f"\nfree memory before / after close (MiB): "
              + ", ".join(f"GPU {d} {before[d] / MiB:.0f} / {after[d] / MiB:.0f}" for d in range(n_dev))
              + f"\nthis process (nvidia-smi, MiB): before {before_mine}, holding {holding_mine}, "
              f"after {after_mine}, after 3 more close()/rerun cycles {cycles_mine}; "
              f"other processes before {before_others}, after {after_others}")
        if before_mine is None:   # no per-process memory from nvidia-smi: only the free memory can be checked
            for d in range(n_dev):
                self.assertLessEqual(abs(after[d] - before[d]), free_tolerance, f"GPU {d}")
            return
        for d in range(n_dev):
            # this process's own memory on each GPU, which other processes on a shared GPU cannot disturb
            self.assertGreater(holding_mine.get(d, 0), before_mine.get(d, 0) + 100, f"GPU {d}: slots not seen")
            self.assertLessEqual(abs(after_mine.get(d, 0) - before_mine.get(d, 0)), 64, f"GPU {d}")
            for k, mine in enumerate(cycles_mine):   # no growth from one close() to the next
                self.assertLessEqual(mine.get(d, 0) - after_mine.get(d, 0), 64, f"GPU {d}, cycle {k + 1}")
            # the free memory, on a GPU that no other process uses: another process's allocations would move it
            if not before_others.get(d) and not after_others.get(d):
                self.assertLessEqual(abs(after[d] - before[d]), free_tolerance, f"GPU {d}")
            else:
                print(f"GPU {d}: free-memory check skipped, another process uses this GPU "
                      f"({before_others.get(d)} / {after_others.get(d)} MiB before / after)")


if __name__ == "__main__":
    unittest.main()
