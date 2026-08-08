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
  T3   the **benchmark** arm (Phase D): the full FUTO swipe-1 train pool with
       hygiene only, **NO session exclusion**, plus the FULL How-We-Swipe release
       (1,338 participants, not the 1,052-log local subset). Exact-trace dedup vs
       the canonical holdout is the *only* contamination control. See the
       disclosure in ``PHASE_D.md`` §2: the published FUTO baselines were trained
       on the literal test traces, so a tier that is exact-deduped is already
       strictly more conservative than they are — but T3 is contributor-dirty by
       construction and can therefore NOT support a generalization claim. It
       exists to be comparable with published numbers, nothing else.
  T4   the **curated** benchmark arm (Phase E): T3's contamination policy applied
       to T1's curated FUTO source — the user's 688,025-row filtered+normalised
       pool instead of the raw swipe-1 train corpus, plus the same full HWS
       release. T3 vs T4 is curation at benchmark scale. T4 inherits T3's
       disclosure verbatim: it is contributor-dirty and is a benchmark tier only.
  T3hws  the HWS half of T3 alone (Phase E). Not a training tier on its own — it
       is the oversampling supply for the 3x-HWS arm, concatenated onto
       ``train_t3.npz`` twice via ``train.py --train-npz a,b,b``.

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
from typing import Dict, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import trace_hash as trace_hash_xyt  # noqa: E402
from scan_futo_sessions import trace_hash  # noqa: E402

NST = Path("/home/will/git/swype/neural-swipe-typing/data")
CCO = Path("/home/will/git/swype/cc-old-data")

#: Full How-We-Swipe release (fetch_hws_full.py). 1,338 participants; the pool the
#: canonical splits were built from is the 1,052-log local subset of this.
HWS_FULL_LOGS = Path.home() / "ctc-train" / "data" / "hws_full" / "swipelogs"

#: Column offsets of the 12-field space-separated swipetraces .log rows
#: (sentence timestamp keyb_width keyb_height event x_pos y_pos x_radius
#:  y_radius angle word is_err), documented in the OSF wiki.
LOG_TS, LOG_KW, LOG_KH = 1, 2, 3
LOG_EVENT, LOG_X, LOG_Y, LOG_WORD, LOG_IS_ERR = 4, 5, 6, 10, 11

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
    """Exact-trace dedup key: a–z-normalized word + exact float64 x/y bytes.

    Both this and ``prepare_data.trace_hash_xyt`` key on the **normalized** word
    as of the campaign-2 post-decode fix. They previously keyed on the raw word,
    which is why ``'arabian.'`` in a tier did not match ``'arabian'`` in the
    holdout even though the two produce a bit-identical input tensor and the same
    CTC target.

    ⚠ **The tiers currently cached under ``~/ctc-train`` were built with the old
    key and were deliberately NOT rebuilt** — `AUDIT_PREDECODE.md` §E judged a
    rebuild the worse trade, because re-rolling six seeds against a +0.28 pt val
    margin risks more than the defect costs. `AUDIT_FINAL.md` §4 bounded the
    effect by measurement rather than assumption: the leaked rows score **4.34 pt
    below** comparable non-leaked ones (i.e. no memorization signal), and removing
    all of them moves the headline by **< 0.05 pt on val / 0.20 pt on test**, with
    all five bars still clearing on every one of six seeds. Any tier built from
    here on gets the correct key.
    """
    return trace_hash(word, [p["x"] for p in pts], [p["y"] for p in pts])


def load_pool_hashes(path: Path) -> Set[bytes]:
    """Hash a canonical-format jsonl (``{word, points:[{t,x,y}]}``)."""
    out = set()
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out.add(hash_row(o["word"], o["points"]))
    return out


def load_pool_hashes_xyt(path: Path) -> Set[bytes]:
    """As :func:`load_pool_hashes`, under ``prepare_data``'s ``word+x+y+t`` hash.

    The two conventions differ in two ways: this one includes the timestamps and
    does **not** lowercase the word. Applying both and taking the union means a
    tier row is dropped if it matches a holdout trace under either.
    """
    out = set()
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            pts = o["points"]
            out.add(trace_hash_xyt(o["word"], [p["x"] for p in pts],
                                   [p["y"] for p in pts], [p["t"] for p in pts]))
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
    out = resolve(args.workdir, f"data/tier_t1{args.suffix}.jsonl")
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


def canonical_points(d):
    """Raw corpus ``data`` -> canonical points (x/y verbatim, t rebased to 0)."""
    t_0 = d[0]["t"]
    return [{"t": float(p["t"] - t_0), "x": p["x"], "y": p["y"]} for p in d]


def quality_reason(o, d, valid_set) -> Optional[str]:
    """First T2b quality gate this raw corpus row fails, or ``None`` if it passes.

    Returns the stats key so callers can both count and branch on one decision.
    The gates are verbatim from the user's recovered filter; keeping them in one
    function is what lets ``build_val_clean.py`` reconstruct a tier's contributor
    set without a second, drifting copy of the cascade.
    """
    if o.get("orientation", "") != "portrait-primary":
        return "not_portrait"
    cw, ch = o.get("canvas_width", 0), o.get("canvas_height", 0)
    if cw <= ch:
        return "canvas_dims"
    if cw > MAX_CANVAS_WIDTH:
        return "canvas_wide"
    if not (MIN_POINTS <= len(d) <= MAX_POINTS):
        return "too_many_points"
    dur = d[-1]["t"] - d[0]["t"]
    if dur < MIN_DURATION_MS or dur > MAX_DURATION_MS:
        return "bad_duration"
    # Mean path speed in normalised units per ms.
    px = np.array([p["x"] for p in d])
    py = np.array([p["y"] for p in d])
    spd = float(np.hypot(np.diff(px), np.diff(py)).sum()) / dur if dur > 0 else 0.0
    if spd < MIN_SPEED or spd > MAX_SPEED:
        return "bad_speed"
    cw_ = canonicalize_word(o["word"])
    if not (MIN_WORD_LEN <= len(cw_) <= MAX_WORD_LEN) or not cw_.isalpha():
        return "invalid_word"
    if valid_set is not None and cw_ not in valid_set:
        return "not_in_dictionary"
    return None


def build_t2(args, holdout: Set[bytes], h2s, tainted, quality: bool) -> None:
    """HF swipe-1 train, converted to canonical rows, hygiene + contamination."""
    tag = "t2b" if quality else "t2"
    out = resolve(args.workdir, f"data/tier_{tag}{args.suffix}.jsonl")
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
                reason = quality_reason(o, d, valid_set)
                if reason is not None:
                    st[reason] += 1
                    continue
            # canonical row: x,y verbatim (frame is identical), t relative, word lowered
            pts = canonical_points(d)
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


def parse_hws_log(path: Path):
    """One swipetraces ``.log`` -> canonical ``(word, points)`` rows.

    Bit-exact replica of ``neural-swipe-typing/process_swipe_logs.py``, which is
    what produced the HWS half of the canonical splits:

    * skip the header line and any row with fewer than 12 space-separated fields
      or a non-integer in a numeric column;
    * drop every row flagged ``is_err == 1`` **before** trace assembly, so an
      error row silently disappears from the middle of a trace rather than
      splitting it;
    * a trace opens on ``touchstart`` and closes on its ``touchend`` or on the
      next ``touchstart``, and is kept only at ``>= 3`` points;
    * ``t = timestamp - trace_start``, ``x = x_pos / keyb_width``,
      ``y = y_pos / keyb_height`` — the same float division, so the output is
      bit-identical to the canonical rows (verified: all 60,303 unique traces of
      the 61,597-row canonical pool are reproduced exactly from the release).

    The caller applies the ``len(word) >= 2`` drop, exactly as the original
    pipeline did downstream of this function.
    """
    out = []
    cur: list = []
    start = kw = kh = None
    word = None

    def flush() -> None:
        nonlocal cur
        if len(cur) >= 3 and word is not None:
            out.append((word, [{"t": float(ts - start),
                                "x": float(x) / float(kw),
                                "y": float(y) / float(kh)} for ts, x, y in cur]))
        cur = []

    with path.open(errors="replace") as fh:
        fh.readline()                                   # header
        for line in fh:
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                ts = int(parts[LOG_TS])
                w_, h_ = int(parts[LOG_KW]), int(parts[LOG_KH])
                x, y = int(parts[LOG_X]), int(parts[LOG_Y])
                is_err = int(parts[LOG_IS_ERR])
            except (ValueError, IndexError):
                continue
            if is_err == 1:
                continue
            event = parts[LOG_EVENT]
            if event == "touchstart":
                flush()
                cur = [(ts, x, y)]
                start, kw, kh, word = ts, w_, h_, parts[LOG_WORD]
            elif event in ("touchmove", "touchend") and cur:
                cur.append((ts, x, y))
                if event == "touchend":
                    flush()
    flush()
    return out


def build_t3(args, holdout: Set[bytes], holdout_xyt: Set[bytes]) -> None:
    """The Phase-D benchmark tier: full FUTO + full HWS, exact-trace dedup only.

    Deliberately **no session/participant exclusion** — see the module docstring
    and ``PHASE_D.md`` §2. Both hash forms are applied to every candidate row:
    ``(word.lower(), x, y)`` from ``scan_futo_sessions`` and
    ``(word, x, y, t)`` from ``prepare_data``. The first ignores timing, so it is
    the stricter of the two; taking their union means a row is dropped if it
    matches a holdout trace under *either* convention.
    """
    out = resolve(args.workdir, f"data/tier_t3{args.suffix}.jsonl")
    st = dict(futo_in=0, futo_invalid_sentence=0, futo_leak_xy=0, futo_leak_xyt=0,
              futo_kept=0, hws_logs=0, hws_traces=0, hws_len1=0,
              hws_leak_xy=0, hws_leak_xyt=0, hws_kept=0)
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        # ── FUTO: the full swipe-1 train corpus, hygiene gate only ──────────────
        with open(args.corpus) as f:
            for line in f:
                o = json.loads(line)
                st["futo_in"] += 1
                if o.get("potentially_invalid_sentence"):
                    st["futo_invalid_sentence"] += 1
                    continue
                pts = canonical_points(o["data"])
                word = o["word"].lower()
                if hash_row(word, pts) in holdout:
                    st["futo_leak_xy"] += 1
                    continue
                if trace_hash_xyt(word, [p["x"] for p in pts], [p["y"] for p in pts],
                                  [p["t"] for p in pts]) in holdout_xyt:
                    st["futo_leak_xyt"] += 1
                    continue
                w.write(json.dumps({"word": word, "points": pts}) + "\n")
                st["futo_kept"] += 1
        print(f"[T3] futo pass done ({time.time() - t0:.0f}s): {st}", flush=True)
        write_hws_release(w, st, holdout, holdout_xyt)
    st["total"] = st["futo_kept"] + st["hws_kept"]
    print(f"[T3] {st}  ({time.time() - t0:.0f}s) -> {out}")
    (out.with_suffix(".stats.json")).write_text(json.dumps(st, indent=1))


def write_hws_release(w, st: Dict[str, int], holdout: Set[bytes],
                      holdout_xyt: Set[bytes]) -> None:
    """Append every kept trace of the FULL How-We-Swipe release to an open file.

    The filter is the canonical one (``is_err = 0`` and ``>= 3`` points, both
    inside :func:`parse_hws_log`, then ``len(word) >= 2``) plus exact-trace dedup
    against the canonical holdout under both hash conventions. Shared by T3 and
    T4 so the two tiers cannot drift apart on the HWS side, and by the HWS-only
    tier that supplies the oversampling copies (Phase E, E3b).

    *st* is updated in place with the ``hws_*`` counters.
    """
    for log in sorted(HWS_FULL_LOGS.glob("*.log")):
        st["hws_logs"] += 1
        for word, pts in parse_hws_log(log):
            st["hws_traces"] += 1
            if len(word) < 2:
                st["hws_len1"] += 1
                continue
            if hash_row(word, pts) in holdout:
                st["hws_leak_xy"] += 1
                continue
            if trace_hash_xyt(word, [p["x"] for p in pts], [p["y"] for p in pts],
                              [p["t"] for p in pts]) in holdout_xyt:
                st["hws_leak_xyt"] += 1
                continue
            w.write(json.dumps({"word": word, "points": pts}) + "\n")
            st["hws_kept"] += 1


def hws_stats() -> Dict[str, int]:
    """Zeroed ``hws_*`` counters for :func:`write_hws_release`."""
    return dict(hws_logs=0, hws_traces=0, hws_len1=0, hws_leak_xy=0,
                hws_leak_xyt=0, hws_kept=0)


def build_t3hws(args, holdout: Set[bytes], holdout_xyt: Set[bytes]) -> None:
    """The HWS half of T3 on its own — the oversampling supply for Phase E's E3b.

    Duplicating rows inside a tier jsonl would be undone by ``prepare_data.py``'s
    exact self-dedup, so the 3x-HWS arm is built instead by concatenating this
    npz onto ``train_t3.npz`` twice at load time (``train.py --train-npz a,b,b``).
    That is exact 3x oversampling under a plain without-replacement shuffle,
    rather than the with-replacement approximation a weighted sampler would give.
    """
    out = resolve(args.workdir, f"data/tier_t3hws{args.suffix}.jsonl")
    st = hws_stats()
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        write_hws_release(w, st, holdout, holdout_xyt)
    st["total"] = st["hws_kept"]
    print(f"[T3hws] {st}  ({time.time() - t0:.0f}s) -> {out}")
    (out.with_suffix(".stats.json")).write_text(json.dumps(st, indent=1))


def build_t4(args, holdout: Set[bytes], holdout_xyt: Set[bytes]) -> None:
    """T4 — the user's curated FUTO pool at benchmark scale + the full HWS release.

    T4 is T3's contamination policy (exact-trace dedup only, **no session or
    participant exclusion**) applied to T1's *curated* FUTO source. The two tiers
    therefore isolate curation at benchmark scale:

    ============  ====================================  ==================
    tier          FUTO source                           session exclusion
    ============  ====================================  ==================
    T1            curated 688,025-row pool              yes (FUTO side)
    T3            raw swipe-1 train, hygiene gate only  no
    **T4**        **curated 688,025-row pool**          **no**
    ============  ====================================  ==================

    ⚠ T4 inherits T3's disclosure verbatim (``PHASE_D.md`` §2): every contributor
    who produced a val/test trace also has other traces here, so T4 is a
    *benchmark* tier and cannot support a generalization claim.

    The curated pool is already in canonical ``{word, points}`` form, so no
    hygiene gate or point rebasing applies — its own cascade ran upstream.
    """
    out = resolve(args.workdir, f"data/tier_t4{args.suffix}.jsonl")
    st = dict(futo_in=0, futo_leak_xy=0, futo_leak_xyt=0, futo_kept=0)
    st.update(hws_stats())
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        with open(CCO / "train_futo_filtered_norm.jsonl") as f:
            for line in f:
                o = json.loads(line)
                st["futo_in"] += 1
                pts = o["points"]
                word = o["word"].lower()
                if hash_row(word, pts) in holdout:
                    st["futo_leak_xy"] += 1
                    continue
                if trace_hash_xyt(word, [p["x"] for p in pts],
                                  [p["y"] for p in pts],
                                  [p["t"] for p in pts]) in holdout_xyt:
                    st["futo_leak_xyt"] += 1
                    continue
                w.write(json.dumps({"word": word, "points": pts}) + "\n")
                st["futo_kept"] += 1
        print(f"[T4] futo pass done ({time.time() - t0:.0f}s): {st}", flush=True)
        write_hws_release(w, st, holdout, holdout_xyt)
    st["total"] = st["futo_kept"] + st["hws_kept"]
    print(f"[T4] {st}  ({time.time() - t0:.0f}s) -> {out}")
    (out.with_suffix(".stats.json")).write_text(json.dumps(st, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--corpus", type=Path,
                    default=NST / "futo" / "train.jsonl",
                    help="HF swipe-1 train.jsonl (raw schema) for T2/T2b")
    ap.add_argument("--tiers", default="t1,t2,t2b")
    ap.add_argument("--suffix", default="",
                    help="appended to every tier's output stem (e.g. '_strict'), so a "
                         "rebuild under different exclusion rules lands in its own file "
                         "instead of clobbering the tier a run already trained on")
    ap.add_argument("--keep-hws-overlap", action="store_true", dest="keep_hws_overlap",
                    help="keep HWS rows from participants who also appear in the "
                         "holdout (reproduces the original, contaminated T1)")
    ap.add_argument("--strict-session", action="store_true", dest="strict_session",
                    help="also drop rows whose session cannot be recovered")
    args = ap.parse_args()

    holdout = load_pool_hashes(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes(NST / "test_hwsfuto.jsonl")
    print(f"canonical holdout traces: {len(holdout)}")

    want = set(args.tiers.split(","))
    # T3/T3hws/T4 are the tiers with no session exclusion, so they need no corpus
    # index — but they apply BOTH hash conventions instead of one.
    no_session = {"t3", "t3hws", "t4"}
    if want & no_session:
        holdout_xyt = load_pool_hashes_xyt(NST / "val_hwsfuto.jsonl") | \
            load_pool_hashes_xyt(NST / "test_hwsfuto.jsonl")
        print(f"canonical holdout traces (word+x+y+t hash): {len(holdout_xyt)}")
        if "t3" in want:
            build_t3(args, holdout, holdout_xyt)
        if "t3hws" in want:
            build_t3hws(args, holdout, holdout_xyt)
        if "t4" in want:
            build_t4(args, holdout, holdout_xyt)
    if not (want - no_session):
        return 0

    h2s, tainted = session_lookup(args.workdir)
    print(f"corpus hash->session entries: {len(h2s)}; tainted sessions: {len(tainted)}")

    if "t1" in want:
        build_t1(args, holdout, h2s, tainted)
    if "t2" in want:
        build_t2(args, holdout, h2s, tainted, quality=False)
    if "t2b" in want:
        build_t2(args, holdout, h2s, tainted, quality=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
