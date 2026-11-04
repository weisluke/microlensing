"""The rule that judges a batched map against serial IPM.run() maps.

Serial runs of the same map are not bit-identical (float atomics add in a different order each run), so a
batched map is judged against the noise between serial runs of its own configuration: noise(ref_a, ref_b, ...)
is the metrics of each further serial run against ref_a (the largest of each metric when there are several),
and the candidate's metrics against ref_a may be at most 3 times that noise plus a floor:
- mean, p999 and p9999: cand <= 3 * noise + floor
- the sum: |cand sum_ratio| <= 3 * |noise sum_ratio| + floor
- non-finite or negative pixels always fail.
max is never a failure: the worst float pixel of two serial runs differs by up to 0.14 relative.

The floors are 1e-7 for float and 1e-13 for double, except two float floors that the serial noise of IPM itself
needs (Task 11 ruling): the sum of a float map, 1e-6 (1e-5 for a parity map), and p9999 of a float parity map,
3e-2. A triangle near a critical curve can take either parity from one run to the next, which moves its area
between the minima and saddle maps: the parity maps' sums and their top 0.01% of pixels jump by quantized amounts,
so that a single pair of serial runs often shows far less noise than a third run. The GPU tests therefore make
three serial runs per configuration and pass the larger noise of the two pairs: noise(ref_a, ref_b, ref_c).

Measured false failures on serial runs only (10 runs per configuration, every ordered tuple, 2026-10-21 on A100):
with noise from two pairs and these floors, 0 of 5040 tuples for every map of nb 1000^2, Salpeter 600^2, 800^2 and
600^2 with parities (total, minima, saddles), num_rays_y=4, a star file, Salpeter 600^2 with parities, and double
600^2 with and without parities. With one pair and floors of 1e-7, 26-38% of the float parity-map tuples, up to 7.4%
of the float total maps and 1% of the double parity maps failed. A serial map of another seed fails the rule
(tests/batch/test_batch_gpu.py)."""
import sys
from pathlib import Path

_GPU_DIR = str(Path(__file__).resolve().parent.parent / "gpu")
if _GPU_DIR not in sys.path:
    sys.path.insert(0, _GPU_DIR)

import compare_maps  # noqa: E402

FACTOR = 3
PERCENTILE_KEYS = ("mean", "p999", "p9999")
FLOORS = {
    # (is_double, parity map): floors per metric
    (False, False): {"mean": 1e-7, "p999": 1e-7, "p9999": 1e-7, "sum_ratio": 1e-6},
    (False, True): {"mean": 1e-7, "p999": 1e-7, "p9999": 3e-2, "sum_ratio": 1e-5},
    (True, False): {"mean": 1e-13, "p999": 1e-13, "p9999": 1e-13, "sum_ratio": 1e-13},
    (True, True): {"mean": 1e-13, "p999": 1e-13, "p9999": 1e-13, "sum_ratio": 1e-13},
}


def noise(ref_a, ref_b, *more_refs) -> dict:
    """The metrics of the further serial runs (ref_b, ...) against the first (ref_a): with several, the largest
    value of each metric (of |sum_ratio| for the sum)."""
    runs = [compare_maps.map_metrics(r, ref_a) for r in (ref_b,) + more_refs]
    out = dict(runs[0])
    for m in runs[1:]:
        for key in PERCENTILE_KEYS + ("max", "offmask_max"):
            out[key] = max(out[key], m[key])
        if abs(m["sum_ratio"]) > abs(out["sum_ratio"]):
            out["sum_ratio"] = m["sum_ratio"]
    return out


def judge(candidate, ref_a, noise_metrics, is_double, parity=False) -> list:
    """The failures of candidate against ref_a under noise_metrics (from noise()); an empty list passes.
    parity: the map is a minima or saddle map."""
    floors = FLOORS[(bool(is_double), bool(parity))]
    m = compare_maps.map_metrics(candidate, ref_a)
    fails = compare_maps._common_failures(m)
    for key in PERCENTILE_KEYS:
        limit = FACTOR * noise_metrics[key] + floors[key]
        if not m[key] <= limit:
            fails.append(f"{key}={m[key]:.3e} > {FACTOR} x noise {noise_metrics[key]:.3e} + {floors[key]:.0e}")
    limit = FACTOR * abs(noise_metrics["sum_ratio"]) + floors["sum_ratio"]
    if not abs(m["sum_ratio"]) <= limit:
        fails.append(f"|sum_ratio|={abs(m['sum_ratio']):.3e} > {FACTOR} x noise "
                     f"{abs(noise_metrics['sum_ratio']):.3e} + {floors['sum_ratio']:.0e}")
    return fails
