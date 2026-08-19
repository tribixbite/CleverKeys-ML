#!/usr/bin/env python3
"""Assert the P6 full-pool holdouts are bit-identical to the P4 90/10 ones.

`script_synth.side_of` maps the holdout split to the reserved donor half
regardless of `--train-donor-side`, and the split RNG is seeded per split, so
changing the TRAIN donor side must leave `holdout.npz` untouched.  If that
holds, the P4 -> P6 comparison is exactly paired on the same 10,000 rows and a
McNemar is legitimate; if it does not, every before/after number in P6 is
confounded and the run stops here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

W = Path.home() / "ctc-train"
ARRAYS = ("features", "targets", "target_lengths", "words", "donor_row",
          "donor_group", "drawn_duration_ms")

bad = 0
for code in sys.argv[1:]:
    mine = 0
    a_p, b_p = W / f"cache_{code}_v2/holdout.npz", W / f"cache_{code}_v2full/holdout.npz"
    with np.load(a_p, allow_pickle=True) as a, np.load(b_p, allow_pickle=True) as b:
        for k in ARRAYS:
            x, y = a[k], b[k]
            if x.shape != y.shape:
                print(f"[{code}] {k}: SHAPE {x.shape} != {y.shape}")
                bad += 1; mine += 1
                continue
            if x.dtype.kind in "fiu":
                d = float(np.abs(x.astype("float64") - y.astype("float64")).max())
                if d != 0.0:
                    print(f"[{code}] {k}: max|Δ| = {d:.3e}")
                    bad += 1; mine += 1
            elif not (x == y).all():
                print(f"[{code}] {k}: content differs")
                bad += 1; mine += 1
    print(f"[{code}] holdout bit-identical to the P4 cache: "
          f"{'NO' if mine else 'yes'} ({len(np.load(a_p)['words'])} rows)")
print("HOLDOUT-INVARIANCE:", "FAIL" if bad else "PASS")
raise SystemExit(1 if bad else 0)
