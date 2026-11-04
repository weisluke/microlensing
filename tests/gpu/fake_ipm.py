#!/usr/bin/env python3
"""Stand-in for ipm_gpu in unit tests: writes the output files a real run writes.
Environment switches: FAKE_IPM_FAIL=1 (exit 3), FAKE_IPM_NO_PARAM=1 (no parameter file),
FAKE_IPM_NO_MAP=1 (no map), FAKE_IPM_VALUE (pixel value, default 2.0), FAKE_IPM_SLEEP (seconds), FAKE_IPM_ALLOC_MB (allocate and touch that many MB before sleeping),
FAKE_IPM_SHAPE (rows x cols of the written maps, default 3x4; the -npy arguments are ignored),
FAKE_IPM_STDOUT (text printed to stdout before the final line).
With -wp 1 the minima and saddles maps hold 1/4 and 3/4 of the value, so that minima + saddles = total as in a real run."""
import os
import struct
import sys
import time

import numpy as np

args = sys.argv[1:]


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


prefix = opt("-o", "./")
npy2, npy1 = (int(v) for v in os.environ.get("FAKE_IPM_SHAPE", "3x4").split("x"))
ballast = bytearray(int(os.environ.get("FAKE_IPM_ALLOC_MB", "0")) * 2**20)
for i in range(0, len(ballast), 4096):
    ballast[i] = 1
time.sleep(float(os.environ.get("FAKE_IPM_SLEEP", "0.02")))
if os.environ.get("FAKE_IPM_FAIL"):
    print("fake failure", file=sys.stderr)
    sys.exit(3)
value = float(os.environ.get("FAKE_IPM_VALUE", "2.0"))
maps = {"ipm_magnifications.bin": value}
if opt("-wp", "0") == "1":
    maps.update({"ipm_magnifications_minima.bin": 0.25 * value, "ipm_magnifications_saddles.bin": 0.75 * value})
if not os.environ.get("FAKE_IPM_NO_MAP"):
    for name, v in maps.items():
        with open(prefix + name, "wb") as f:
            f.write(struct.pack("ii", npy2, npy1))
            f.write(np.full((npy2, npy1), v, dtype=np.float32).tobytes())
with open(prefix + "ipm_stars.bin", "wb") as f:
    f.write(b"\0" * 16)
if not os.environ.get("FAKE_IPM_NO_PARAM"):
    with open(prefix + "ipm_parameter_info.txt", "w") as f:
        f.write("kappa_tot 0.4\nnum_stars 137\nt_shoot_cells 0.123\n")
print(os.environ.get("FAKE_IPM_STDOUT", ""), end="")
print("fake run done")
