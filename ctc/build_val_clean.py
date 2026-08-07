#!/usr/bin/env python3
"""Carve per-arm ``val_clean`` masks — val rows with no contributor overlap with an arm.

Aggregate val accuracy is not comparable across arms whose training pools differ in
contributor overlap: T0 shares a contributor with 75 % of val, the session-excluded
tiers do not. So for each arm we mark the val-9918 rows whose contributing session
appears nowhere in that arm's training rows, and score that subset alongside the
aggregate.

Sessions are read from each row's **own** ``session`` field, never inferred from a
trace hash. That distinction is the whole point of this rewrite:

  * the canonical splits and both tier source pools carry ``session`` verbatim, at
    100 % coverage, so every val row and every training row has a known contributor;
  * the previous hash-resolver version could not resolve 705 val rows (~15 % of the
    FUTO pool was renormalised and lost bit-identity with the raw corpus) and counted
    them dirty, which threw away 456 usable rows and, worse, mislabelled 38 T0 rows
    as *clean* when their contributor is in fact in T0.

That gap is not cosmetic. 219 contributor sessions produced 249 val-FUTO rows yet
never entered ``futo_tainted_sessions.npz``, because the holdout traces that would
have exposed them were exactly the ones whose hash no longer matches the corpus. The
27,356 corpus rows those sessions hold are still inside T1/T2/T2b, so the tiers are
**not** fully session-disjoint from val, and only a session-field mask can see it.

Because a tier's jsonl carries no ``session`` field, each arm's contributor set is
reconstructed by replaying that tier's keep-rule over its SOURCE file and reading the
session off the source row. The keep-rules are imported from ``build_tiers.py`` (not
re-implemented) so the two cannot drift.

Output: ``cache/val_clean_masks.json`` — ``{arm: {"clean": [bool…], "n_clean": …}}``.
Existing arms are preserved unless recomputed, so a mask a run was already scored on
cannot be silently replaced.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tiers import (CCO, NST, build_valid_set, canonical_points,  # noqa: E402
                         hash_row, load_pool_hashes, quality_reason)
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

CORPUS = NST / "futo" / "train.jsonl"
FUTO_POOL = CCO / "train_futo_filtered_norm.jsonl"

#: Worker globals (inherited through fork; never pickled).
_HOLDOUT: Set[bytes] = set()
_TAINT: Set[str] = set()
_H2S: Dict[bytes, str] = {}
_VALID = None


def _init(holdout, taint, h2s, valid) -> None:
    global _HOLDOUT, _TAINT, _H2S, _VALID
    _HOLDOUT, _TAINT, _H2S, _VALID = holdout, taint, h2s, valid


def key(source: str, session: str) -> str:
    """Namespaced contributor id — HWS and FUTO session ids are different spaces."""
    return f"{source}:{session}"


def _line_range(path: Path, start: int, end: int):
    """Yield parsed rows of a byte range, starting at the first whole line."""
    with open(path, "rb") as f:
        if start:
            f.seek(start - 1)
            f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield json.loads(line)


def _corpus_chunk(job: Tuple[int, int]) -> Tuple[Set[str], Set[str]]:
    """T2 / T2b keep-rules over a corpus byte range -> their contributor sets."""
    t2: Set[str] = set()
    t2b: Set[str] = set()
    for o in _line_range(CORPUS, *job):
        if o.get("potentially_invalid_sentence"):
            continue
        d = o["data"]
        pts = canonical_points(d)
        h = hash_row(o["word"], pts)
        if h in _HOLDOUT:
            continue
        s = _H2S.get(h)
        if s is not None and s in _TAINT:
            continue
        k = key("futo", o.get("session") or "")
        t2.add(k)
        if quality_reason(o, d, _VALID) is None:
            t2b.add(k)
    return t2, t2b


def _futo_pool_chunk(job: Tuple[int, int]) -> Set[str]:
    """T1's FUTO keep-rule over the filtered pool (unmapped rows are KEPT)."""
    out: Set[str] = set()
    for o in _line_range(FUTO_POOL, *job):
        h = hash_row(o["word"], o["points"])
        if h in _HOLDOUT:
            continue
        s = _H2S.get(h)
        if s is not None and s in _TAINT:
            continue
        out.add(key("futo", o.get("session") or ""))
    return out


def chunks(path: Path, jobs: int) -> List[Tuple[int, int]]:
    size = path.stat().st_size
    step = size // jobs + 1
    return [(a, min(a + step, size)) for a in range(0, size, step)]


def hws_sets(holdout: Set[bytes]) -> Tuple[Dict[bytes, str], Set[str], Set[bytes]]:
    """-> ``(hash->hws session, participants touching holdout, hws pool hashes)``."""
    sess: Dict[bytes, str] = {}
    for f in ("train_hws_filtered", "val_hws_filtered"):
        with open(NST / "hws" / f"{f}.jsonl") as fh:
            for line in fh:
                o = json.loads(line)
                sess[hash_row(o["word"], o["points"])] = str(o.get("session"))
    tainted = {sess[k] for k in holdout if k in sess}
    return sess, tainted, set(sess)


def t0_sessions() -> Set[str]:
    """T0's contributors, straight off the canonical training split."""
    out: Set[str] = set()
    with open(NST / "train_hwsfuto.jsonl") as f:
        for line in f:
            o = json.loads(line)
            out.add(key(o["source"], o["session"]))
    return out


def t1_hws_sessions(holdout: Set[bytes], hws_pool: Set[bytes],
                    hws_sess: Dict[bytes, str], hws_tainted: Set[str],
                    strict: bool) -> Set[str]:
    """HWS contributors surviving build_t1's rule (strict = participant exclusion)."""
    out: Set[str] = set()
    with open(NST / "train_hwsfuto.jsonl") as f:
        for line in f:
            o = json.loads(line)
            h = hash_row(o["word"], o["points"])
            if h not in hws_pool or h in holdout:
                continue
            if strict and hws_sess.get(h) in hws_tainted:
                continue
            out.add(key("hws", o["session"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--val", type=Path, default=NST / "val_hwsfuto.jsonl")
    ap.add_argument("--out", default="cache/val_clean_masks.json")
    ap.add_argument("--arms", default="T0,T1,T1strict,T2,T2b")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    t_start = time.time()
    holdout = load_pool_hashes(NST / "val_hwsfuto.jsonl") | \
        load_pool_hashes(NST / "test_hwsfuto.jsonl")
    d = np.load(resolve(args.workdir, "cache/futo_session_index.npz"), allow_pickle=False)
    voc = [str(v) for v in d["session_vocab"]]
    S = d["session"]
    h2s: Dict[bytes, str] = {}
    for i, row in enumerate(d["hashes"]):
        h2s.setdefault(bytes(row), voc[int(S[i])])
    taint = set(np.load(resolve(args.workdir, "cache/futo_tainted_sessions.npz"),
                        allow_pickle=False)["names"].tolist())
    hws_sess, hws_tainted, hws_pool = hws_sets(holdout)
    print(f"[prep] holdout {len(holdout)}  h2s {len(h2s)}  futo-taint {len(taint)}  "
          f"hws-taint {len(hws_tainted)}  ({time.time() - t_start:.0f}s)", flush=True)

    # val contributors, read directly — no resolution step, no unresolved rows.
    val_keys: List[str] = []
    with open(args.val) as f:
        for line in f:
            o = json.loads(line)
            val_keys.append(key(o["source"], o["session"]))
    n_val = len(val_keys)
    print(f"val rows: {n_val}  distinct contributors: {len(set(val_keys))}")

    want = set(args.arms.split(","))
    valid = build_valid_set() if {"T2b"} & want else None
    arm_sessions: Dict[str, Set[str]] = {}

    if "T0" in want:
        arm_sessions["T0"] = t0_sessions()
    if {"T1", "T1strict"} & want:
        t = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(args.jobs, initializer=_init,
                      initargs=(holdout, taint, h2s, valid)) as pool:
            futo: Set[str] = set()
            for part in pool.imap_unordered(_futo_pool_chunk,
                                            chunks(FUTO_POOL, args.jobs)):
                futo |= part
        print(f"[T1] futo contributors {len(futo)} ({time.time() - t:.0f}s)", flush=True)
        if "T1" in want:
            arm_sessions["T1"] = futo | t1_hws_sessions(
                holdout, hws_pool, hws_sess, hws_tainted, strict=False)
        if "T1strict" in want:
            arm_sessions["T1strict"] = futo | t1_hws_sessions(
                holdout, hws_pool, hws_sess, hws_tainted, strict=True)
    if {"T2", "T2b"} & want:
        t = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(args.jobs, initializer=_init,
                      initargs=(holdout, taint, h2s, valid)) as pool:
            t2: Set[str] = set()
            t2b: Set[str] = set()
            for a, b in pool.imap_unordered(_corpus_chunk, chunks(CORPUS, args.jobs)):
                t2 |= a
                t2b |= b
        print(f"[T2/T2b] contributors {len(t2)} / {len(t2b)} "
              f"({time.time() - t:.0f}s)", flush=True)
        if "T2" in want:
            arm_sessions["T2"] = t2
        if "T2b" in want:
            arm_sessions["T2b"] = t2b

    # Merge into any existing file so a mask a run was already scored on survives.
    out_path = resolve(args.workdir, args.out)
    out: Dict[str, object] = {}
    if out_path.exists():
        out = json.loads(out_path.read_text())
    out["n_val"] = n_val
    out.pop("val_unresolved", None)          # no longer possible: 100 % coverage
    masks: Dict[str, np.ndarray] = {}
    for arm, sess in arm_sessions.items():
        clean = np.array([k not in sess for k in val_keys], bool)
        masks[arm] = clean
        out[arm] = {"train_sessions": len(sess), "n_clean": int(clean.sum()),
                    "pct_clean": round(float(clean.mean()) * 100, 2),
                    "clean": clean.tolist()}
        print(f"  {arm:<9} {len(sess):>6} contributors -> val_clean "
              f"{int(clean.sum())}/{n_val} ({clean.mean() * 100:.1f}%)")

    have = {k: np.array(v["clean"], bool) for k, v in out.items()
            if isinstance(v, dict) and "clean" in v and k != "SHARED"}
    if have:
        shared = np.ones(n_val, bool)
        for m in have.values():
            shared &= m
        out["SHARED"] = {"n_clean": int(shared.sum()),
                         "pct_clean": round(float(shared.mean()) * 100, 2),
                         "clean": shared.tolist(),
                         "note": "intersection of every arm mask present in this file"}
        print(f"  {'SHARED':<9} {'':>6} -> val_clean {int(shared.sum())}/{n_val} "
              f"({shared.mean() * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"wrote {out_path}  ({time.time() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
