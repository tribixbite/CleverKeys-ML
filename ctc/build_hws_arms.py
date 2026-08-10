#!/usr/bin/env python3
"""Build the Phase I-B How-We-Swipe filtering arms (englishLevel / quality gates).

The canonical HWS training rows only ever got basic hygiene (``is_err = 0``,
``>= 3`` points, ``len(word) >= 2``, exact-trace dedup vs the canonical
holdout).  The user's originally-intended filtering — native-speaker selection
and motion-quality gating — was never applied because ``metadata.tsv`` was not
on this machine (``DATA_TIERS.md`` §1.2).  The full OSF release
(``fetch_hws_full.py``) carries ``englishLevel`` in every per-user ``.json``
sidecar, so the filtering is finally measurable.  This script builds the HWS
side of each arm; the FUTO side is held fixed (the T3 benchmark pool).

Arms (HWS pool only; each is a strict subset of the control):

  control    every kept trace of the full release — MUST byte-identically
             reproduce ``tier_t3hws.jsonl`` (78,155 rows), which is the
             validation that this script sits on the audited code path.
  nativeadv  participants with self-reported englishLevel native|advanced.
  native     englishLevel native only.
  quality    all levels + the HWS-derived motion gates below.

Motion gates for the ``quality`` arm — derived from the measured distributions
of the 84,612 basic-hygiene traces of the release itself (percentiles in
``PHASE_I_DATA.md``), NOT copied from the FUTO cascade that Phase A measured as
*negative*.  The FUTO bounds are demonstrably wrong for HWS: its speed floor
(0.001 u/ms) sits at the HWS 25th percentile and would discard a quarter of the
corpus.  These bounds instead trim only the degenerate tails (~2 % total):

  duration  [150, 10000] ms      (p1 = 158, p99.5 = 9,997)
  points    [8, 512]             (p0.9 = 8-9; 512 = p99.7, the FUTO cap)
  path len  >= 0.10              (p0.9; below this the finger did not move —
                                  a tap logged against a multi-letter word)
  speed     [0.0002, 0.008] u/ms (p0.5 / p99.9; u = letter-area widths)

Every gate's drop count is reported separately (first failing gate wins), so
the cascade stays attributable.

Outputs (under ``--workdir``, default ``~/ctc-train``):

  data/hws_arm_<arm>.jsonl        canonical {"word", "points"} rows
  data/hws_arm_<arm>.stats.json   drop accounting + participant counts
  data/hws_uid_levels.json        uid -> englishLevel for all 1,338 users
  data/tier_t3futo.jsonl          the FUTO-only prefix of tier_t3.jsonl
                                  (with --t3futo; line-count verified)

Training composition per arm (mirrors the adopted T3+3xHWS recipe, where
``--train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz`` = FUTO + 3x HWS):

  --train-npz train_t3futo.npz,hws_arm_<arm>.npz,hws_arm_<arm>.npz,hws_arm_<arm>.npz

Note one deliberate delta vs the control invocation: featurization-time
self-dedup (prepare_data) runs per-npz here, so a FUTO row bit-identical to an
HWS row would survive where the joint train_t3.npz build dropped it.  Measured
below at build time and reported; the two corpora sit on different coordinate
grids, so the expected overlap is zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tiers import (HWS_FULL_LOGS, hash_row, load_pool_hashes,  # noqa: E402
                         load_pool_hashes_xyt, parse_hws_log, trace_hash_xyt)
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

NST = Path("/home/will/git/swype/neural-swipe-typing/data")

#: HWS-derived motion gates (see module docstring for the percentile basis).
HWS_MIN_DURATION_MS, HWS_MAX_DURATION_MS = 150.0, 10_000.0
HWS_MIN_POINTS, HWS_MAX_POINTS = 8, 512
HWS_MIN_PATHLEN = 0.10
HWS_MIN_SPEED, HWS_MAX_SPEED = 0.0002, 0.008

LEVELS = ("native", "advanced", "intermediate", "beginner", "na")

ARM_LEVELS: Dict[str, Optional[Set[str]]] = {
    "control": None,                        # no level filter
    "nativeadv": {"native", "advanced"},
    "native": {"native"},
    "quality": None,                        # no level filter; motion gates instead
}


def load_levels() -> Dict[str, str]:
    """uid -> normalized self-reported englishLevel for every release user."""
    levels: Dict[str, str] = {}
    for p in HWS_FULL_LOGS.glob("*.json"):
        with p.open(errors="replace") as fh:
            raw = json.load(fh).get("englishLevel") or "NA"
        levels[p.stem] = raw.strip().lower()
    if len(levels) != 1338:
        raise SystemExit(f"expected 1338 sidecars, found {len(levels)}")
    return levels


def quality_reason(word: str, pts: List[dict]) -> Optional[str]:
    """First failing HWS motion gate, or None if the trace passes all four."""
    n = len(pts)
    if n < HWS_MIN_POINTS or n > HWS_MAX_POINTS:
        return "bad_points"
    dur = pts[-1]["t"]
    if dur < HWS_MIN_DURATION_MS or dur > HWS_MAX_DURATION_MS:
        return "bad_duration"
    xs = np.fromiter((p["x"] for p in pts), float, n)
    ys = np.fromiter((p["y"] for p in pts), float, n)
    plen = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
    if plen < HWS_MIN_PATHLEN:
        return "short_path"
    speed = plen / dur                       # dur >= 150 here, so finite
    if speed < HWS_MIN_SPEED or speed > HWS_MAX_SPEED:
        return "bad_speed"
    return None


def build_arm(arm: str, data_dir: Path, levels: Dict[str, str],
              holdout: Set[bytes], holdout_xyt: Set[bytes]) -> None:
    """One arm's jsonl + stats, on the exact write_hws_release keep-path."""
    keep_levels = ARM_LEVELS[arm]
    gates = arm == "quality"
    out = data_dir / f"hws_arm_{arm}.jsonl"
    st: Dict[str, object] = dict(
        arm=arm, logs=0, users_kept=0, traces=0, len1=0, level_drop=0,
        bad_points=0, bad_duration=0, short_path=0, bad_speed=0,
        leak_xy=0, leak_xyt=0, kept=0,
        users_by_level={l: 0 for l in LEVELS}, kept_by_level={l: 0 for l in LEVELS},
    )
    t0 = time.time()
    with open(out, "w") as w:
        for log in sorted(HWS_FULL_LOGS.glob("*.log")):
            st["logs"] += 1
            lvl = levels[log.stem]
            if keep_levels is not None and lvl not in keep_levels:
                # count the whole user's would-be rows as level_drop lazily:
                # cheaper to just skip the parse; per-level totals come from
                # the control arm's kept_by_level.
                continue
            st["users_kept"] += 1
            st["users_by_level"][lvl] += 1
            for word, pts in parse_hws_log(log):
                st["traces"] += 1
                if len(word) < 2:
                    st["len1"] += 1
                    continue
                if gates:
                    reason = quality_reason(word, pts)
                    if reason is not None:
                        st[reason] += 1
                        continue
                if hash_row(word, pts) in holdout:
                    st["leak_xy"] += 1
                    continue
                if trace_hash_xyt(word, [p["x"] for p in pts],
                                  [p["y"] for p in pts],
                                  [p["t"] for p in pts]) in holdout_xyt:
                    st["leak_xyt"] += 1
                    continue
                w.write(json.dumps({"word": word, "points": pts}) + "\n")
                st["kept"] += 1
                st["kept_by_level"][lvl] += 1
    print(f"[{arm}] {json.dumps({k: v for k, v in st.items() if not isinstance(v, dict)})}"
          f"  ({time.time() - t0:.0f}s) -> {out}")
    out.with_suffix(".stats.json").write_text(json.dumps(st, indent=1))


def extract_t3futo(data_dir: Path) -> None:
    """The FUTO-only prefix of tier_t3.jsonl, split at the recorded futo_kept.

    ``build_t3`` writes the whole FUTO pass before any HWS row, so the first
    ``futo_kept`` lines ARE the FUTO half; the remainder must equal
    ``hws_kept`` and byte-match ``tier_t3hws.jsonl`` (both checked here).
    """
    stats = json.loads((data_dir / "tier_t3.stats.json").read_text())
    n_futo, n_hws = stats["futo_kept"], stats["hws_kept"]
    src = data_dir / "tier_t3.jsonl"
    out = data_dir / "tier_t3futo.jsonl"
    h_tail = hashlib.sha256()
    n = 0
    with open(src, "rb") as f, open(out, "wb") as w:
        for i, line in enumerate(f):
            if i < n_futo:
                w.write(line)
            else:
                h_tail.update(line)
            n = i + 1
    if n != n_futo + n_hws:
        raise SystemExit(f"tier_t3.jsonl has {n} lines, stats say {n_futo}+{n_hws}")
    ref = hashlib.sha256((data_dir / "tier_t3hws.jsonl").read_bytes()).hexdigest()
    if h_tail.hexdigest() != ref:
        raise SystemExit("tier_t3.jsonl HWS tail != tier_t3hws.jsonl — layout drifted")
    print(f"[t3futo] {n_futo} FUTO lines -> {out}; HWS tail sha256 verified == tier_t3hws.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--arms", default="control,nativeadv,native,quality")
    ap.add_argument("--t3futo", action="store_true",
                    help="also extract data/tier_t3futo.jsonl from tier_t3.jsonl")
    args = ap.parse_args()

    data_dir = resolve(args.workdir, Path("data"))
    levels = load_levels()
    (data_dir / "hws_uid_levels.json").write_text(json.dumps(levels, indent=0,
                                                             sort_keys=True))

    holdout = load_pool_hashes(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes(NST / "test_hwsfuto.jsonl")
    holdout_xyt = load_pool_hashes_xyt(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes_xyt(NST / "test_hwsfuto.jsonl")
    print(f"holdout: {len(holdout)} xy-hashes, {len(holdout_xyt)} xyt-hashes")

    if args.t3futo:
        extract_t3futo(data_dir)

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARM_LEVELS:
            raise SystemExit(f"unknown arm {arm!r} (choices: {sorted(ARM_LEVELS)})")
        build_arm(arm, data_dir, levels, holdout, holdout_xyt)

    # Validation: the control arm must reproduce tier_t3hws.jsonl byte-for-byte.
    ctrl = data_dir / "hws_arm_control.jsonl"
    ref = data_dir / "tier_t3hws.jsonl"
    if ctrl.exists() and ref.exists():
        a = hashlib.sha256(ctrl.read_bytes()).hexdigest()
        b = hashlib.sha256(ref.read_bytes()).hexdigest()
        verdict = "IDENTICAL" if a == b else "MISMATCH"
        print(f"[check] hws_arm_control vs tier_t3hws: {verdict} ({a[:16]} / {b[:16]})")
        if a != b:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
