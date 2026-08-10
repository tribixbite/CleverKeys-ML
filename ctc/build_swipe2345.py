#!/usr/bin/env python3
"""Build the Phase-J FUTO swipe-2..5 training pools (DATASET_SCOUT.md §2/§5).

Pools (all MIT, collection runs added to ``futo-org/swipe.futo.org`` 2026-06-15):

  sw234      swipe-2 (informal register) + swipe-3 (unique words + deliberate
             misspellings, swiped as spelled) + swipe-4 (confusable words —
             purpose-built for the t1↔t3 gap). English, canonical QWERTY frame.
  sw5q       swipe-5 rows with ``layout == 'qwerty'`` only. The five layouts of
             our alt-layout eval suite (dvorak/azerty/qwertz/german/spanish) are
             **structurally excluded** — training on them destroys the eval
             (DATASET_SCOUT.md §2.3) — as are all other non-qwerty layouts.
  realalt    swipe-5 clearflow / kasroz / toki_pona — real human swipes on
             layouts no arm has ever trained on and no committed eval uses.
             Split 80/20 by SESSION (deterministic blake2b hash) so the holdout
             is contributor-disjoint from the training side of the same pool.

Hygiene (the T3 philosophy — hygiene yes, curation no; four exclusion-curation
negatives stand):

  * ``dual_finger == 1`` rows dropped (different input mode + schema fork).
  * ``distance >= 100000`` dropped: the corpus encodes validity FAILURES as
    sentinel codes 100001/100002/100003 (measured; swipe-1 train itself carries
    only 1.9 % of them after FUTO's own validity filtering, the unfiltered
    swipe-2..5 carry 7–12 %). This reproduces the validity class swipe-1 was
    already filtered by upstream — an invalid-label gate, not a quality opinion.
    Real-valued distances are kept in full, whatever their size.
  * >= 3 points (the HWS parse minimum), canonicalized word length >= 2.
  * Exact-trace dedup vs the canonical val/test holdout under BOTH hash
    conventions (T3 rule; expected 0 — these are post-holdout collection runs).

Verification performed here, not assumed (DATASET_SCOUT claims re-checked):

  * FULL session-set disjointness vs the raw 5.1 GB swipe-1 train corpus (the
    scout sampled 68 ids; this compares the complete sets).
  * Endpoint-proximity of every emitted pool against its layout geometry
    (PHASE_H §2.3 metric: fraction of traces whose first/last point lands on the
    word's first/last key), so a frame regression cannot pass silently.

Usage:
  python build_swipe2345.py [--skip-session-scan]
Then featurize each emitted jsonl via prepare_data.py --extra-train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tiers import NST, canonical_points, hash_row, load_pool_hashes, \
    load_pool_hashes_xyt  # noqa: E402
from prepare_data import trace_hash as trace_hash_xyt  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

SRC = Path.home() / "ctc-train" / "data" / "futo_swipe2345"
RAW_SWIPE1 = NST / "futo" / "train.jsonl"

#: swipe-5 layouts that are (or feed) a committed evaluation — NEVER trained on.
EVAL_LAYOUTS = {"dvorak", "azerty", "qwertz", "german", "spanish"}
#: swipe-5 layouts kept as the realalt pools.
REALALT_LAYOUTS = ("clearflow", "kasroz", "toki_pona")
#: sentinel distance codes = FUTO validity-check failures (measured 100001..3).
DIST_SENTINEL = 100000.0
HOLDOUT_FRAC = 0.2


def session_split_holdout(session: str) -> bool:
    """Deterministic per-session 80/20 split for the realalt pools."""
    h = hashlib.blake2b(session.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 100 < int(HOLDOUT_FRAC * 100)


def load_layout_letters(path: Path) -> Tuple[List[str], np.ndarray]:
    """FUTO layout json -> (letters present, [n,2] centers) for a–z keys only."""
    obj = json.loads(path.read_text())
    letters, cxy = [], []
    for k in obj["keys"]:
        ch = str(k.get("letter", "")).lower()
        if len(ch) == 1 and "a" <= ch <= "z":
            letters.append(ch)
            cxy.append((float(k["cx"]), float(k["cy"])))
    return letters, np.asarray(cxy, np.float32)


def endpoint_stats(rows: List[dict], letters: List[str], centers: np.ndarray,
                   n_sample: int = 2000) -> Dict[str, float]:
    """PHASE_H §2.3 endpoint-proximity: hit-rate + mean distance, first/last key."""
    idx = {c: i for i, c in enumerate(letters)}
    rng = np.random.RandomState(0)
    pick = rng.choice(len(rows), size=min(n_sample, len(rows)), replace=False)
    s_hit = e_hit = 0
    s_d, e_d = [], []
    n = 0
    for i in pick:
        o = rows[i]
        w = [c for c in o["word"].lower() if c in idx]
        if not w:
            continue
        pts = o["points"]
        p0 = np.array([pts[0]["x"], pts[0]["y"]], np.float32)
        p1 = np.array([pts[-1]["x"], pts[-1]["y"]], np.float32)
        d0 = np.linalg.norm(centers - p0, axis=1)
        d1 = np.linalg.norm(centers - p1, axis=1)
        s_hit += int(d0.argmin() == idx[w[0]])
        e_hit += int(d1.argmin() == idx[w[-1]])
        s_d.append(float(d0[idx[w[0]]]))
        e_d.append(float(d1[idx[w[-1]]]))
        n += 1
    return {"n": n, "start_hit": round(s_hit / max(n, 1), 3),
            "end_hit": round(e_hit / max(n, 1), 3),
            "start_d": round(float(np.mean(s_d)), 4),
            "end_d": round(float(np.mean(e_d)), 4)}


def scan_swipe1_sessions() -> Set[str]:
    """Complete session-id set of the raw swipe-1 train corpus (one full pass)."""
    t0 = time.time()
    out: Set[str] = set()
    with open(RAW_SWIPE1) as f:
        for line in f:
            # cheap targeted parse: "session":"..." appears once per row
            j = line.find('"session":"')
            if j >= 0:
                k = line.find('"', j + 11)
                out.add(line[j + 11:k])
    print(f"[sessions] swipe-1 train: {len(out)} unique sessions "
          f"({time.time() - t0:.0f}s)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--skip-session-scan", action="store_true",
                    dest="skip_session_scan",
                    help="skip the full swipe-1 session comparison (5.1 GB pass)")
    args = ap.parse_args()

    holdout = load_pool_hashes(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes(NST / "test_hwsfuto.jsonl")
    holdout_xyt = load_pool_hashes_xyt(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes_xyt(NST / "test_hwsfuto.jsonl")
    print(f"holdout traces: {len(holdout)} (xy) / {len(holdout_xyt)} (xyt)")

    # ── pass 1: load + gate every run ───────────────────────────────────────────
    pools: Dict[str, List[dict]] = {"sw234": [], "sw5q": []}
    realalt: Dict[str, List[dict]] = {n: [] for n in REALALT_LAYOUTS}
    sessions: Dict[str, Set[str]] = {}
    stats: Dict[str, Counter] = {}
    sw5_layout_kept = Counter()
    sw5_lang = Counter()
    for run in (2, 3, 4, 5):
        st = Counter()
        sess: Set[str] = set()
        with open(SRC / f"swipe{run}.jsonl") as f:
            for line in f:
                o = json.loads(line)
                st["rows_in"] += 1
                sess.add(o["session"])
                if o.get("dual_finger"):
                    st["dual_finger"] += 1
                    continue
                if float(o["distance"]) >= DIST_SENTINEL:
                    st["distance_sentinel"] += 1
                    continue
                d = o["data"]
                if len(d) < 3:
                    st["too_few_points"] += 1
                    continue
                word = o["word"].lower()
                cw = "".join(c for c in word if c.isalpha())
                if len(cw) < 2:
                    st["short_word"] += 1
                    continue
                pts = canonical_points(d)
                if hash_row(word, pts) in holdout:
                    st["leak_xy"] += 1
                    continue
                if trace_hash_xyt(word, [p["x"] for p in pts],
                                  [p["y"] for p in pts],
                                  [p["t"] for p in pts]) in holdout_xyt:
                    st["leak_xyt"] += 1
                    continue
                row = {"word": word, "points": pts}
                if run == 5:
                    lay = o.get("layout", "")
                    if lay in EVAL_LAYOUTS:
                        st["eval_layout_excluded"] += 1
                        continue
                    if lay == "qwerty":
                        # en only: a pl/fr/de word swiped on plain QWERTY would
                        # have its accents stripped (not folded) by the
                        # normalizer — 'über' -> 'ber' — i.e. a wrong CTC label.
                        sw5_lang[o.get("language", "?")] += 1
                        if o.get("language") != "en":
                            st["nonen_qwerty_excluded"] += 1
                            continue
                        pools["sw5q"].append(row)
                        sw5_layout_kept[lay] += 1
                    elif lay in realalt:
                        row["session"] = o["session"]
                        realalt[lay].append(row)
                        sw5_layout_kept[lay] += 1
                    else:
                        st["other_layout"] += 1
                        continue
                    st["kept"] += 1
                else:
                    pools["sw234"].append(row)
                    st["kept"] += 1
        sessions[f"swipe{run}"] = sess
        stats[f"swipe{run}"] = st
        print(f"[swipe-{run}] {dict(st)}  sessions={len(sess)}")
    print(f"[swipe-5] kept by layout: {dict(sw5_layout_kept)}  "
          f"qwerty languages: {dict(sw5_lang)}")

    # ── session disjointness, measured on the complete sets ─────────────────────
    ver: Dict[str, object] = {}
    for a in sessions:
        for b in sessions:
            if a < b:
                inter = sessions[a] & sessions[b]
                if inter:
                    print(f"[sessions] ⚠ {a} ∩ {b} = {len(inter)}")
                ver[f"{a}∩{b}"] = len(inter)
    if not args.skip_session_scan:
        s1 = scan_swipe1_sessions()
        for name, ss in sessions.items():
            inter = ss & s1
            ver[f"{name}∩swipe1"] = len(inter)
            flag = " ⚠ NOT DISJOINT" if inter else ""
            print(f"[sessions] {name} ∩ swipe-1-train: {len(inter)}{flag}")

    # ── emit + endpoint validation ──────────────────────────────────────────────
    qletters, qcenters = load_layout_letters(
        Path(__file__).resolve().parent / "en_qwerty.json")
    out_dir = resolve(args.workdir, Path("data"))
    endpoints: Dict[str, Dict[str, float]] = {}
    for name in ("sw234", "sw5q"):
        rows = pools[name]
        out = out_dir / f"tier_{name}.jsonl"
        with open(out, "w") as w:
            for r in rows:
                w.write(json.dumps(r) + "\n")
        endpoints[name] = endpoint_stats(rows, qletters, qcenters)
        print(f"[{name}] {len(rows)} rows -> {out}  endpoints {endpoints[name]}")
    for lay in REALALT_LAYOUTS:
        rows = realalt[lay]
        letters, centers = load_layout_letters(SRC / f"layout_{lay}.json")
        endpoints[lay] = endpoint_stats(rows, letters, centers)
        tr_n = ho_n = 0
        tr_s: Set[str] = set()
        ho_s: Set[str] = set()
        with open(out_dir / f"realalt_{lay}_train.jsonl", "w") as wt, \
             open(out_dir / f"realalt_{lay}_heldout.jsonl", "w") as wh:
            for r in rows:
                s = r.pop("session")
                # toki_pona has only 2 contributor sessions — unsplittable by
                # session, so it becomes a pure ZERO-SHOT eval pool (all rows
                # held out, none trained). Its 14-letter alphabet also makes it
                # the most alien committed geometry, which is what a zero-shot
                # probe wants.
                if lay == "toki_pona" or session_split_holdout(s):
                    wh.write(json.dumps(r) + "\n")
                    ho_n += 1
                    ho_s.add(s)
                else:
                    wt.write(json.dumps(r) + "\n")
                    tr_n += 1
                    tr_s.add(s)
        assert not (tr_s & ho_s), f"{lay}: session split not disjoint"
        print(f"[realalt/{lay}] {len(letters)} letter keys; train {tr_n} rows "
              f"({len(tr_s)} sessions) / heldout {ho_n} rows ({len(ho_s)} "
              f"sessions)  endpoints {endpoints[lay]}")

    rec = {"stats": {k: dict(v) for k, v in stats.items()},
           "sw5_layout_kept": dict(sw5_layout_kept),
           "sw5q_languages": dict(sw5_lang),
           "session_checks": ver, "endpoints": endpoints,
           "gates": {"distance_sentinel": DIST_SENTINEL, "min_points": 3,
                     "min_word": 2, "dual_finger": "excluded",
                     "eval_layouts_excluded": sorted(EVAL_LAYOUTS)},
           "holdout_frac": HOLDOUT_FRAC}
    (out_dir / "swipe2345_build.stats.json").write_text(json.dumps(rec, indent=1))
    print(f"wrote {out_dir / 'swipe2345_build.stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
