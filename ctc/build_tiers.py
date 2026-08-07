#!/usr/bin/env python3
"""Build the tiered training pools (T1 / T2 / T2b) for the data scale-up ladder.

Tiers (all evaluated against the SAME canonical val-9918 / test-2400):

  T0   canonical 110,876 — the existing control, already cached as ``train.npz``.
       55,438 HWS + 55,438 FUTO, a deliberately balanced curated merge.
  T1   the user's curation re-applied at scale: the HWS half of T0 plus the FULL
       688,025-row filtered+normalised FUTO pool.
  T2   "more data, less curation": HF swipe-1 train with basic hygiene only
       (``potentially_invalid_sentence`` dropped) — tests whether the curation
       that discarded ~250 k rows was net-positive.
  T2b  T2 plus the recoverable quality rules (duration window, point cap), so the
       curation question decomposes into "which filter earned its keep".

Contamination control (applied to every tier):
  a. **Exact-trace dedup** against canonical val/test — also re-applied inside
     ``prepare_data.py``, so a tier physically cannot smuggle a holdout trace.
  b. **Session disjointness** — every contributor session that produced a
     canonical val/test trace is excluded wholesale. Sessions are recovered by
     hashing each row against the ``futo_session_index`` built by
     ``scan_futo_sessions.py``. Rows that cannot be mapped (the FUTO pool was
     renormalised for ~15 % of rows, losing bit-identity with the raw corpus) are
     kept by default and dropped under ``--strict-session``; both counts are
     reported so the choice is explicit rather than silent.

HWS-side note: the HWS half cannot be expanded — all 61,597 filtered HWS rows are
already consumed by T0 (55,438 train + 6,159 holdout), with zero left over.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from scan_futo_sessions import trace_hash  # noqa: E402

NST = Path("/home/will/git/swype/neural-swipe-typing/data")
CCO = Path("/home/will/git/swype/cc-old-data")

#: Quality rules recovered VERBATIM from the user's actual FUTO filter
#: (scripts/filter_and_normalize_dataset.py, whose stats file reproduces the
#: 939,550 -> 764,473 cascade). These are the real thresholds; an earlier guess
#: of 80/3000ms + 200 points came from a different, unused cascade.
MIN_WORD_LEN, MAX_WORD_LEN = 2, 20
MIN_POINTS, MAX_POINTS = 8, 512
MIN_DURATION_MS, MAX_DURATION_MS = 40.0, 4000.0
MIN_SPEED, MAX_SPEED = 0.001, 0.01
MAX_CANVAS_WIDTH = 900


def canonicalize_word(word: str) -> str:
    """Lowercase and strip the punctuation the user's filter stripped."""
    w = word.lower()
    for ch in "'.,;:!?()":
        w = w.replace(ch, "")
    return w


def build_valid_set():
    """NLTK words + wordfreq top-400k, canonicalised — the user's dictionary gate.

    Returns ``None`` (gate disabled, reported as such) if neither corpus loads.
    """
    import re
    vs = set()
    try:
        from wordfreq import top_n_list
        vs |= {w for w in top_n_list("en", 400000) if re.fullmatch(r"[a-z]{2,20}", w)}
    except Exception as e:  # noqa: BLE001
        print(f"  [dict] wordfreq unavailable ({e})")
    try:
        from nltk.corpus import words as nltk_words
        vs |= {canonicalize_word(w) for w in nltk_words.words()}
    except Exception as e:  # noqa: BLE001
        print(f"  [dict] nltk words unavailable ({e})")
    return vs or None


def hash_row(word: str, pts) -> bytes:
    return trace_hash(word, [p["x"] for p in pts], [p["y"] for p in pts])


def load_pool_hashes(path: Path) -> Set[bytes]:
    """Hash a canonical-format jsonl (``{word, points:[{t,x,y}]}``)."""
    out = set()
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out.add(hash_row(o["word"], o["points"]))
    return out


def session_lookup(workdir: Path):
    """-> ``(hash -> session_id, tainted_session_ids)`` from the corpus index."""
    d = np.load(resolve(workdir, "cache/futo_session_index.npz"), allow_pickle=False)
    voc = np.array([str(v) for v in d["session_vocab"]])
    S = d["session"]
    h2s: Dict[bytes, int] = {}
    for i, row in enumerate(d["hashes"]):
        h2s.setdefault(bytes(row), int(S[i]))
    tainted_names = set(
        np.load(resolve(workdir, "cache/futo_tainted_sessions.npz"),
                allow_pickle=False)["names"].tolist())
    tainted = {i for i, n in enumerate(voc) if n in tainted_names}
    return h2s, tainted


def emit(rows, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def build_t1(args, holdout: Set[bytes], h2s, tainted) -> None:
    """HWS half of T0 + the full filtered FUTO pool, contamination-controlled."""
    hws_pool = load_pool_hashes(NST / "hws" / "train_hws_filtered.jsonl") | \
        load_pool_hashes(NST / "hws" / "val_hws_filtered.jsonl")
    # HWS participants are NOT disjoint from the holdout (98.4 % of the HWS half
    # shares a participant with val/test), so the HWS side needs its own session
    # exclusion or T1 inherits T0's contamination. Build participant -> rows and
    # the set of participants that produced a holdout trace.
    hws_sess = {}
    for f in ("train_hws_filtered", "val_hws_filtered"):
        with open(NST / "hws" / f"{f}.jsonl") as fh:
            for line in fh:
                o = json.loads(line)
                hws_sess[hash_row(o["word"], o["points"])] = str(o.get("session"))
    hws_tainted = {hws_sess[k] for k in holdout if k in hws_sess}
    print(f"[T1] HWS participants touching holdout: {len(hws_tainted)}")
    out = resolve(args.workdir, "data/tier_t1.jsonl")
    stats = dict(hws_kept=0, hws_leak=0, hws_taint=0, futo_in=0, futo_leak=0, futo_taint=0,
                 futo_unmapped=0, futo_kept=0)
    t0 = time.time()
    with open(out, "w") as w:
        for line in open(NST / "train_hwsfuto.jsonl"):
            o = json.loads(line)
            hh = hash_row(o["word"], o["points"])
            if hh not in hws_pool:
                continue
            if hh in holdout:
                stats["hws_leak"] += 1
                continue
            if not args.keep_hws_overlap and hws_sess.get(hh) in hws_tainted:
                stats["hws_taint"] += 1
                continue
            w.write(json.dumps({"word": o["word"], "points": o["points"]}) + "\n")
            stats["hws_kept"] += 1
        for line in open(CCO / "train_futo_filtered_norm.jsonl"):
            o = json.loads(line)
            stats["futo_in"] += 1
            h = hash_row(o["word"], o["points"])
            if h in holdout:
                stats["futo_leak"] += 1
                continue
            s = h2s.get(h)
            if s is None:
                stats["futo_unmapped"] += 1
                if args.strict_session:
                    continue
            elif s in tainted:
                stats["futo_taint"] += 1
                continue
            w.write(json.dumps({"word": o["word"], "points": o["points"]}) + "\n")
            stats["futo_kept"] += 1
    stats["total"] = stats["hws_kept"] + stats["futo_kept"]
    print(f"[T1] {stats}  ({time.time() - t0:.0f}s) -> {out}")
    (out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=1))


def build_t2(args, holdout: Set[bytes], h2s, tainted, quality: bool) -> None:
    """HF swipe-1 train, converted to canonical rows, hygiene + contamination."""
    tag = "t2b" if quality else "t2"
    out = resolve(args.workdir, f"data/tier_{tag}.jsonl")
    st = dict(rows_in=0, invalid_sentence=0, leak=0, taint=0, unmapped=0,
              bad_duration=0, too_many_points=0, not_portrait=0, canvas_dims=0,
              canvas_wide=0, bad_speed=0, invalid_word=0, not_in_dictionary=0, kept=0)
    valid_set = build_valid_set() if quality else None
    if quality:
        st["dictionary_terms"] = len(valid_set) if valid_set else 0
    t0 = time.time()
    with open(out, "w") as w, open(args.corpus) as f:
        for line in f:
            o = json.loads(line)
            st["rows_in"] += 1
            if o.get("potentially_invalid_sentence"):
                st["invalid_sentence"] += 1
                continue
            d = o["data"]
            if quality:
                # Metadata gates, verbatim from the user's filter.
                if o.get("orientation", "") != "portrait-primary":
                    st["not_portrait"] += 1
                    continue
                cw, ch = o.get("canvas_width", 0), o.get("canvas_height", 0)
                if cw <= ch:
                    st["canvas_dims"] += 1
                    continue
                if cw > MAX_CANVAS_WIDTH:
                    st["canvas_wide"] += 1
                    continue
                if not (MIN_POINTS <= len(d) <= MAX_POINTS):
                    st["too_many_points"] += 1
                    continue
                dur = d[-1]["t"] - d[0]["t"]
                if dur < MIN_DURATION_MS or dur > MAX_DURATION_MS:
                    st["bad_duration"] += 1
                    continue
                # Mean path speed in normalised units per ms.
                px = np.array([p["x"] for p in d]); py = np.array([p["y"] for p in d])
                arc = float(np.hypot(np.diff(px), np.diff(py)).sum())
                spd = arc / dur if dur > 0 else 0.0
                if spd < MIN_SPEED or spd > MAX_SPEED:
                    st["bad_speed"] += 1
                    continue
                cw_ = canonicalize_word(o["word"])
                if not (MIN_WORD_LEN <= len(cw_) <= MAX_WORD_LEN) or not cw_.isalpha():
                    st["invalid_word"] += 1
                    continue
                if valid_set is not None and cw_ not in valid_set:
                    st["not_in_dictionary"] += 1
                    continue
            # canonical row: x,y verbatim (frame is identical), t relative, word lowered
            t_0 = d[0]["t"]
            pts = [{"t": float(p["t"] - t_0), "x": p["x"], "y": p["y"]} for p in d]
            h = hash_row(o["word"], pts)
            if h in holdout:
                st["leak"] += 1
                continue
            s = h2s.get(h)
            if s is None:
                st["unmapped"] += 1
                if args.strict_session:
                    continue
            elif s in tainted:
                st["taint"] += 1
                continue
            w.write(json.dumps({"word": o["word"].lower(), "points": pts}) + "\n")
            st["kept"] += 1
    print(f"[{tag.upper()}] {st}  ({time.time() - t0:.0f}s) -> {out}")
    (out.with_suffix(".stats.json")).write_text(json.dumps(st, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--corpus", type=Path,
                    default=NST / "futo" / "train.jsonl",
                    help="HF swipe-1 train.jsonl (raw schema) for T2/T2b")
    ap.add_argument("--tiers", default="t1,t2,t2b")
    ap.add_argument("--keep-hws-overlap", action="store_true", dest="keep_hws_overlap",
                    help="keep HWS rows from participants who also appear in the "
                         "holdout (reproduces the original, contaminated T1)")
    ap.add_argument("--strict-session", action="store_true", dest="strict_session",
                    help="also drop rows whose session cannot be recovered")
    args = ap.parse_args()

    holdout = load_pool_hashes(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes(NST / "test_hwsfuto.jsonl")
    print(f"canonical holdout traces: {len(holdout)}")
    h2s, tainted = session_lookup(args.workdir)
    print(f"corpus hash->session entries: {len(h2s)}; tainted sessions: {len(tainted)}")

    want = set(args.tiers.split(","))
    if "t1" in want:
        build_t1(args, holdout, h2s, tainted)
    if "t2" in want:
        build_t2(args, holdout, h2s, tainted, quality=False)
    if "t2b" in want:
        build_t2(args, holdout, h2s, tainted, quality=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
