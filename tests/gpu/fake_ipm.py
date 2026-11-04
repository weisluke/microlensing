#!/usr/bin/env python3
"""Stand-in for ipm_gpu in unit tests: writes the output files a real run writes.
Environment switches: FAKE_IPM_FAIL=1 (exit 3), FAKE_IPM_NO_PARAM=1 (no parameter file),
FAKE_IPM_NO_MAP=1 (no map), FAKE_IPM_VALUE (pixel value, default 2.0), FAKE_IPM_SLEEP (seconds),
FAKE_IPM_SHAPE (rows x cols of the written maps, default 3x4; the -npy arguments are ignored)."""
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
time.sleep(float(os.environ.get("FAKE_IPM_SLEEP", "0.02")))
if os.environ.get("FAKE_IPM_FAIL"):
    print("fake failure", file=sys.stderr)
    sys.exit(3)
value = float(os.environ.get("FAKE_IPM_VALUE", "2.0"))
names = ["ipm_magnifications.bin"]
if opt("-wp", "0") == "1":
    names += ["ipm_magnifications_minima.bin", "ipm_magnifications_saddles.bin"]
if not os.environ.get("FAKE_IPM_NO_MAP"):
    for name in names:
        with open(prefix + name, "wb") as f:
            f.write(struct.pack("ii", npy2, npy1))
            f.write(np.full((npy2, npy1), value, dtype=np.float32).tobytes())
with open(prefix + "ipm_stars.bin", "wb") as f:
    f.write(b"\0" * 16)
if not os.environ.get("FAKE_IPM_NO_PARAM"):
    with open(prefix + "ipm_parameter_info.txt", "w") as f:
        f.write("kappa_tot 0.4\nnum_stars 137\nt_shoot_cells 0.123\n")
print("fake run done")
