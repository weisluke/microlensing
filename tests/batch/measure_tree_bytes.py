"""Measure the tree allocation per star of IPM runs on a GPU worker, to set IPM<T>::tree_bytes_per_star:
python tests/batch/measure_tree_bytes.py [device]
tree_bytes_per_star = ceil(margin * max / 64) * 64 per precision (max = the largest value printed for it).
The margin is 1.5 in the plan; 1.1 is the largest round figure for which the estimate of stars1e6_f stays within
1.5 times its allocation (test_exports_gpu.py)."""
import ctypes
import math
import sys

from microlensing.IPM import lib_ipm, lib_ipm_double

# name -> (setters, value); the lenses of the nb (notebook), hm100_f and stars1e6_f configurations of tests/gpu/configs.py
NB = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25, half_length_y2=25)
HM100 = dict(kappa_tot=0.5, shear=0.49, kappa_star=0.1, rectangular=1, approx=1, half_length_y1=12.5, half_length_y2=12.5)
STARS1E6 = dict(kappa_tot=0.3, shear=0.3, kappa_star=0.27, half_length_y1=505, half_length_y2=505)
CONFIGS = {
    "nb_1000": dict(NB, num_pixels_y1=1000, num_pixels_y2=1000),
    "nb_5000": dict(NB, num_pixels_y1=5000, num_pixels_y2=5000),
    "hm100": dict(HM100, num_pixels_y1=1000, num_pixels_y2=1000),
    "stars1e6": dict(STARS1E6, num_pixels_y1=5000, num_pixels_y2=5000),
}


def apply(lib, obj, params, write_parities=0):
    for name, value in params.items():
        getattr(lib, "set_" + name)(obj, value)
    lib.set_write_histograms(obj, 1)
    lib.set_write_parities(obj, write_parities)


def run_worker(lib, params, device, write_parities=0):
    """Run one map on a fresh worker; return (num_stars, allocated bytes (pixel, star, tree)); the worker is deleted."""
    obj = lib.IPM_init()
    try:
        if not lib.set_worker(obj, device):
            raise RuntimeError("set_worker failed")
        apply(lib, obj, params, write_parities)
        if not lib.run(obj, 0):
            raise RuntimeError("run failed")
        alloc = (ctypes.c_ulonglong * 3)()
        lib.get_allocated_bytes(obj, alloc)
        return lib.get_num_stars(obj), tuple(alloc)
    finally:
        lib.IPM_delete(obj)


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for label, lib in (("float", lib_ipm.lib), ("double", lib_ipm_double.lib)):
        per_star = []
        for name, params in CONFIGS.items():
            n, alloc = run_worker(lib, params, device)
            per_star.append(alloc[2] / n)
            print(f"{label} {name}: num_stars={n} tree_bytes={alloc[2]} per_star={alloc[2] / n:.2f}")
        print(f"{label}: max per star {max(per_star):.2f} -> constant with margin 1.1: {math.ceil(1.1 * max(per_star) / 64) * 64}, with 1.5: {math.ceil(1.5 * max(per_star) / 64) * 64}")


if __name__ == "__main__":
    main()
