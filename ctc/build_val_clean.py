#!/usr/bin/env python3
"""Carve per-arm ``val_clean`` masks — val rows with no contributor overlap with an arm.

Aggregate val accuracy is not comparable across arms whose training pools differ in
contributor overlap: T0 shares a contributor with 75 % of val, T1/T2/T2b are
contributor-disjoint by construction. So for each arm we mark the val-9918 rows whose
contributing session appears nowhere in that arm's training rows, and evaluate on that
subset as well as in aggregate.

Two masks matter per arm:
  * its **own** mask — the honest generalization number for that arm;
  * the **shared T0-clean mask** — the intersection every arm can be scored on, which
    is what makes the head-to-head cross-arm comparison fair (T0 is the dirty control,
    so its clean subset is the binding constraint).

Sessions come from two sources: FUTO rows via the corpus hash->session index, HWS rows
via the ``session`` field carried in the filtered HWS pools. Rows whose session cannot
be resolved are reported and treated as *potentially overlapping* (conservative).

Output: ``cache/val_clean_masks.json`` — ``{arm: {"clean": [bool…], "n_clean": int, …}}``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from scan_futo_sessions import trace_hash  # noqa: E402

NST = Path("/home/will/git/swype/neural-swipe-typing/data")

#: arm -> training jsonl (T0 is the canonical split; tiers come from build_tiers.py)
ARMS = {
    "T0": NST / "train_hwsfuto.jsonl",
    "T1": Path("data/tier_t1.jsonl"),
    "T2": Path("data/tier_t2.jsonl"),
    "T2b": Path("data/tier_t2b.jsonl"),
}


def h(o) -> bytes:
    p = o["points"]
    return trace_hash(o["word"], [q["x"] for q in p], [q["y"] for q in p])


def build_resolver(workdir: Path):
    """-> ``hash -> 'futo:<id>' | 'hws:<id>'`` covering both corpora."""
    d = np.load(resolve(workdir, "cache/futo_session_index.npz"), allow_pickle=False)
    voc = np.array([str(v) for v in d["session_vocab"]])
    S = d["session"]
    res: Dict[bytes, str] = {}
    for i, row in enumerate(d["hashes"]):
        res.setdefault(bytes(row), "futo:" + voc[int(S[i])])
    for f in ("train_hws_filtered", "val_hws_filtered"):
        with open(NST / "hws" / f"{f}.jsonl") as fh:
            for line in fh:
                o = json.loads(line)
                res.setdefault(h(o), "hws:" + str(o.get("session")))
    return res


def sessions_of(path: Path, res) -> tuple[Set[str], int, int]:
    """Distinct contributor sessions used by a training file (+ resolved/total)."""
    out: Set[str] = set()
    n = ok = 0
    with open(path) as f:
        for line in f:
            n += 1
            s = res.get(h(json.loads(line)))
            if s is not None:
                out.add(s)
                ok += 1
    return out, ok, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--val", type=Path, default=NST / "val_hwsfuto.jsonl")
    ap.add_argument("--out", default="cache/val_clean_masks.json")
    ap.add_argument("--arms", default="T0,T1,T2,T2b")
    args = ap.parse_args()

    res = build_resolver(args.workdir)
    print(f"resolver: {len(res)} hash->session entries")

    val_sess = []
    with open(args.val) as f:
        for line in f:
            val_sess.append(res.get(h(json.loads(line))))
    n_val = len(val_sess)
    unresolved = sum(1 for s in val_sess if s is None)
    print(f"val rows: {n_val} ({unresolved} with unresolved session -> treated as dirty)")

    out: Dict[str, object] = {"n_val": n_val, "val_unresolved": unresolved}
    masks: Dict[str, np.ndarray] = {}
    for arm in args.arms.split(","):
        path = resolve(args.workdir, ARMS[arm])
        if not path.exists():
            print(f"  {arm}: MISSING {path} — skipped")
            continue
        sess, ok, n = sessions_of(path, res)
        clean = np.array([s is not None and s not in sess for s in val_sess], bool)
        masks[arm] = clean
        out[arm] = {"train_rows": n, "train_sessions_resolved": ok,
                    "train_sessions": len(sess),
                    "n_clean": int(clean.sum()),
                    "pct_clean": round(float(clean.mean()) * 100, 2),
                    "clean": clean.tolist()}
        print(f"  {arm}: {n} rows, {len(sess)} sessions ({ok} rows resolved) "
              f"-> val_clean {int(clean.sum())}/{n_val} ({clean.mean() * 100:.1f}%)")

    if "T0" in masks:
        shared = masks["T0"].copy()
        for m in masks.values():
            shared &= m
        out["SHARED"] = {"n_clean": int(shared.sum()),
                         "pct_clean": round(float(shared.mean()) * 100, 2),
                         "clean": shared.tolist(),
                         "note": "intersection of all arm masks; the fair cross-arm subset"}
        print(f"  SHARED (all arms clean): {int(shared.sum())}/{n_val} "
              f"({shared.mean() * 100:.1f}%)")

    p = resolve(args.workdir, args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
