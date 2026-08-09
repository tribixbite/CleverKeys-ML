#!/usr/bin/env python3
"""Measure the realized shared-affine distribution of both train.py samplers.

`ALT_LAYOUT_EVAL.md` §7.2b found the legacy rejection sampler silently truncated
and biased the x-scale on en_qwerty (accepted sx mean 0.955 vs the nominal
0.85–1.15, 31.5 % first-draw rejects) because the key centers span cx 0.05–0.95.
This script runs both samplers against the real layout for N draws and prints,
per axis: acceptance rate, identity-fallback rate, and the realized scale /
translate quantiles — the before/after evidence for the Phase-G fix. It also
asserts the containment invariant (every transformed center in [0,1]) for every
draw of the coupled sampler, so the fix is verified, not just described.

Usage:
  python affine_stats.py --draws 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_LAYOUT  # noqa: E402
from train import (AFFINE_TRIES, SCALE_HI, SCALE_LO, TRANS_ABS,  # noqa: E402
                   SwipeDataset, load_layout_centers)


def summarize(name: str, arr: np.ndarray) -> str:
    q = np.quantile(arr, [0.0, 0.05, 0.5, 0.95, 1.0])
    return (f"    {name}: mean {arr.mean():.4f}  min {q[0]:.4f}  p5 {q[1]:.4f}  "
            f"median {q[2]:.4f}  p95 {q[3]:.4f}  max {q[4]:.4f}")


def run(sampler: str, centers: np.ndarray, draws: int, seed: int) -> None:
    ds = SwipeDataset.__new__(SwipeDataset)          # sampler needs no npz
    ds.centers = centers
    ds.affine_sampler = sampler
    ds.k = centers.shape[0]
    bounds = []
    for ax in range(2):
        lo, hi = float(centers[:, ax].min()), float(centers[:, ax].max())
        s_max = 1.0 / max(hi - lo, 1e-9)
        if lo < 0.5:
            s_max = min(s_max, (0.5 + TRANS_ABS) / (0.5 - lo))
        if hi > 0.5:
            s_max = min(s_max, (0.5 + TRANS_ABS) / (hi - 0.5))
        bounds.append((lo, hi, max(SCALE_LO, min(SCALE_HI, s_max))))
    ds._axis_bounds = bounds

    np.random.seed(seed)
    out = np.empty((draws, 4))
    identity = 0
    for i in range(draws):
        sx, sy, tx, ty, _ = ds._sample_affine()
        out[i] = (sx, sy, tx, ty)
        if sx == 1.0 and sy == 1.0 and tx == 0.0 and ty == 0.0:
            identity += 1
        # Containment invariant — must hold for every draw of either sampler
        # (legacy identity fallbacks trivially satisfy it).
        nx = (centers[:, 0] - 0.5) * sx + 0.5 + tx
        ny = (centers[:, 1] - 0.5) * sy + 0.5 + ty
        assert nx.min() >= -1e-9 and nx.max() <= 1 + 1e-9, (sx, tx)
        assert ny.min() >= -1e-9 and ny.max() <= 1 + 1e-9, (sy, ty)

    # First-draw acceptance for the legacy loop, measured independently so the
    # 31.5 % figure is reproduced rather than quoted.
    np.random.seed(seed)
    rej = 0
    probes = min(draws, 200_000)
    cx, cy = centers[:, 0], centers[:, 1]
    for _ in range(probes):
        sx, sy = np.random.uniform(SCALE_LO, SCALE_HI, 2)
        tx, ty = np.random.uniform(-TRANS_ABS, TRANS_ABS, 2)
        nx = (cx - 0.5) * sx + 0.5 + tx
        ny = (cy - 0.5) * sy + 0.5 + ty
        if not (nx.min() >= 0 and nx.max() <= 1 and ny.min() >= 0 and ny.max() <= 1):
            rej += 1

    print(f"[{sampler}] {draws} draws  identity-fallback {identity / draws:.2%}  "
          f"(first-draw reject rate of the legacy criterion: {rej / probes:.1%})")
    for name, col in (("sx", 0), ("sy", 1), ("tx", 2), ("ty", 3)):
        print(summarize(name, out[:, col]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    centers = load_layout_centers(args.layout)
    for ax, name in ((0, "x"), (1, "y")):
        lo, hi = centers[:, ax].min(), centers[:, ax].max()
        print(f"layout {name}: centers span [{lo:.4f}, {hi:.4f}]  "
              f"span-bound s_max = {1.0 / (hi - lo):.4f}")
    print(f"nominal: scale U({SCALE_LO}, {SCALE_HI})  translate U(±{TRANS_ABS})  "
          f"legacy tries {AFFINE_TRIES}")
    for sampler in ("legacy", "coupled"):
        run(sampler, centers, args.draws, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
