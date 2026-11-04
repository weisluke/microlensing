#!/usr/bin/env python3
"""Configuration matrix for the IPM benchmark (spec section 6).

Arguments listed here are those beyond the common ones; `-v` and `-o` are added by run_ipm().
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

SEED = "20261015"
COMMON = ("-rs", SEED, "-ws", "1", "-wm", "1", "-wh", "1")
NOTEBOOK = ("-k", "0.4", "-y", "0.4", "-ks", "0.2", "-hly1", "25", "-hly2", "25", "-npy1", "5000", "-npy2", "5000")
HM100 = ("-k", "0.5", "-y", "0.49", "-ks", "0.1", "-r", "1", "-a", "1",
         "-hly1", "12.5", "-hly2", "12.5", "-npy1", "1000", "-npy2", "1000")
HM1000 = ("-k", "0.7252", "-y", "0.2729", "-ks", "0.1", "-r", "1", "-a", "1",
          "-hly1", "12.5", "-hly2", "12.5", "-npy1", "1000", "-npy2", "1000")
NRY10 = ("-k", "0.4", "-y", "0.4", "-ks", "0.2", "-hly1", "10", "-hly2", "10",
         "-npy1", "2000", "-npy2", "2000", "-nry", "10")
BIG = ("-k", "0.4", "-y", "0.4", "-ks", "0.2", "-hly1", "40", "-hly2", "40", "-npy1", "8000", "-npy2", "8000")
STARS1E6 = ("-k", "0.3", "-y", "0.3", "-ks", "0.27", "-hly1", "505", "-hly2", "505", "-npy1", "5000", "-npy2", "5000")
SF = ("-k", "0.4", "-y", "0.4", "-hly1", "25", "-hly2", "25", "-npy1", "5000", "-npy2", "5000")


@dataclass(frozen=True)
class Config:
    name: str
    build: str                 # "float" or "double"
    args: tuple                # arguments beyond COMMON, -v and -o
    sets: frozenset            # subset of {"ladder", "full", "truth", "memory"}
    truth: str = None          # double configuration with identical arguments (float configs only)
    starfile_from: str = None  # configuration whose run1 star file is passed with -sf
    parities: bool = False     # writes minima/saddles maps


def _c(name, build, args, sets, **kw):
    return Config(name, build, tuple(args), frozenset(sets), **kw)


CONFIGS = {c.name: c for c in [
    _c("nb_f", "float", NOTEBOOK, {"ladder", "full"}, truth="nb_d"),
    _c("nb_d", "double", NOTEBOOK, {"ladder", "full"}),
    _c("rect_f", "float", NOTEBOOK + ("-r", "1", "-a", "1"), {"ladder", "full"}, truth="rect_d"),
    _c("hm100_f", "float", HM100, {"ladder", "full"}, truth="hm100_d"),
    _c("circ_a0_f", "float", NOTEBOOK + ("-a", "0"), {"full"}, truth="circ_a0_d"),
    _c("rect_a0_f", "float", NOTEBOOK + ("-r", "1", "-a", "0"), {"full"}, truth="rect_a0_d"),
    _c("rect_d", "double", NOTEBOOK + ("-r", "1", "-a", "1"), {"full"}),
    _c("rect_a0_d", "double", NOTEBOOK + ("-r", "1", "-a", "0"), {"full"}),
    _c("par_f", "float", NOTEBOOK + ("-wp", "1"), {"full"}, truth="par_d", parities=True),
    _c("nry10_f", "float", NRY10, {"full"}, truth="nry10_d"),
    _c("big_f", "float", BIG, {"full", "memory"}, truth="big_d"),
    _c("hm1000_f", "float", HM1000, {"full", "memory"}, truth="hm1000_d"),
    _c("hm100_d", "double", HM100, {"full"}),
    _c("sf_f", "float", SF, {"full"}, truth="sf_d", starfile_from="nb_f"),
    _c("stars1e6_f", "float", STARS1E6, {"full", "memory"}, truth="stars1e6_d"),
    # double-truth references for float configurations without a double twin in the matrix
    _c("circ_a0_d", "double", NOTEBOOK + ("-a", "0"), {"truth"}),
    _c("par_d", "double", NOTEBOOK + ("-wp", "1"), {"truth"}, parities=True),
    _c("nry10_d", "double", NRY10, {"truth"}),
    _c("big_d", "double", BIG, {"truth"}),
    _c("hm1000_d", "double", HM1000, {"truth"}),
    _c("sf_d", "double", SF, {"truth"}, starfile_from="nb_f"),
    _c("stars1e6_d", "double", STARS1E6, {"truth"}),
]}

# The quadtree root and the default image plane are centred on (0, 0), so lattice corners on x1 = 0 or x2 = 0 evaluate to exactly
# 0.0 and sit on the tree's dividing lines at every level; the original kernel's rounding then picks either neighbouring leaf.
# A 2.5e-5-pixel shift of the source centre moves every corner off those lines without changing the physics.
SHIFT = ("-cy1", "1e-7", "-cy2", "1e-7")

LADDER = ["nb_f", "nb_d", "rect_f", "hm100_f"]
# R2p changes only the double build: its ladder entry is nb_d and its regression covers every double configuration of the full matrix
RUNG_LADDER_SKIP = {"R2p": {"nb_f", "rect_f", "hm100_f"}}


def configs_in(set_name):
    return [c for c in CONFIGS.values() if set_name in c.sets]


def ladder_configs(rung=None):
    skip = RUNG_LADDER_SKIP.get(rung, set())
    return [CONFIGS[n] for n in LADDER if n not in skip]


def regression_configs(rung, set_name):
    if rung == "R2p" and set_name == "ladder":
        return [c for c in CONFIGS.values() if c.build == "double" and "full" in c.sets]
    return ladder_configs(rung) if set_name == "ladder" else configs_in("full")


def exe_for(build_dir, build):
    return Path(build_dir) / ("ipm_gpu" if build == "float" else "ipm_gpu_double")


def full_args(cfg, ref_root):
    args = list(COMMON) + list(cfg.args)
    if cfg.starfile_from:
        args += ["-sf", str(Path(ref_root) / cfg.starfile_from / "run1" / "ipm_stars.bin")]
    return args


def shifted_args(cfg, ref_root):
    return full_args(cfg, ref_root) + list(SHIFT)


def shift_center():
    """The (center_y1, center_y2) floats of SHIFT, for the Python wrapper's keywords."""
    args = dict(zip(SHIFT[::2], SHIFT[1::2]))
    return float(args["-cy1"]), float(args["-cy2"])


def double_regression_configs():
    """Double configurations under regression, in CONFIGS order: each also gets the shifted-centre equivalence check."""
    return [c for c in CONFIGS.values() if c.build == "double" and ("ladder" in c.sets or "full" in c.sets)]


def ordered(configs):
    """Star-file users after the configurations whose star file they read."""
    return sorted(configs, key=lambda c: (c.starfile_from is not None, list(CONFIGS).index(c.name)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--command", help="print '<exe> <args>' for a configuration")
    ap.add_argument("--build")
    ap.add_argument("--ref-root")
    a = ap.parse_args(argv)
    if a.list:
        for c in CONFIGS.values():
            print(f"{c.name:12s} {c.build:6s} {','.join(sorted(c.sets)):14s} {' '.join(c.args)}")
    if a.command:
        c = CONFIGS[a.command]
        print(exe_for(a.build, c.build), " ".join(full_args(c, a.ref_root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
