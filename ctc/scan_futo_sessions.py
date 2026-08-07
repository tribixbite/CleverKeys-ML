#!/usr/bin/env python3
"""Index the FUTO swipe-1 train corpus by trace hash -> session (contamination control).

Single parallel pass over the 5.2 GB ``train.jsonl``. For every row it records the
content hash of ``(word.lower(), exact float64 x/y bytes)`` together with the row's
``session`` and byte offset, and writes:

  * ``futo_session_index.npz`` — parallel arrays (hash, session_id, line_no) for
    every corpus row, plus the session-id vocabulary.

That index answers both contamination questions in one shot:

  a. **Exact-trace leakage** — which corpus rows reproduce a canonical val/test trace.
  b. **Session leakage** — which *sessions* those rows belong to, so that every
     other row from the same contributor can be dropped as well. A session is one
     volunteer's contribution run, so leaving siblings in would let the model
     memorise a specific person's hand geometry on the exact words we score.

The hash is taken in the CONVERTED frame, which is safe because the corpus frame
and our canonical frame are provably identical (x/y copied verbatim; see
``convert_futo_swipe1.py`` for the verification).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

DEFAULT_CORPUS = Path("/home/will/git/swype/neural-swipe-typing/data/futo/train.jsonl")


def trace_hash(word: str, xs, ys) -> bytes:
    """16-byte content hash of a trace: lowercased word + exact float64 x/y bytes."""
    m = hashlib.blake2b(digest_size=16)
    m.update(word.lower().encode("utf-8"))
    m.update(b"\0")
    m.update(np.asarray(xs, np.float64).tobytes())
    m.update(np.asarray(ys, np.float64).tobytes())
    return m.digest()


def _scan_chunk(job: Tuple[str, int, int]) -> Tuple[List[bytes], List[str], int, int]:
    """Scan a byte range, starting at the first full line at or after ``start``."""
    path, start, end = job
    hashes: List[bytes] = []
    sessions: List[str] = []
    n = 0
    bad = 0
    with open(path, "rb") as f:
        if start:
            f.seek(start - 1)
            f.readline()          # discard the partial line
        else:
            f.seek(0)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            try:
                o = json.loads(line)
                d = o["data"]
                hashes.append(trace_hash(o["word"], [p["x"] for p in d],
                                         [p["y"] for p in d]))
                sessions.append(o.get("session") or "")
            except Exception:     # noqa: BLE001  malformed line
                bad += 1
                continue
            n += 1
    return hashes, sessions, n, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help="FUTO swipe-1 train.jsonl (raw HF row schema)")
    ap.add_argument("--out", default="cache/futo_session_index.npz")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()

    corpus = Path(args.corpus).expanduser()
    size = corpus.stat().st_size
    step = size // args.jobs + 1
    jobs = [(str(corpus), a, min(a + step, size)) for a in range(0, size, step)]
    print(f"[scan] {corpus} ({size / 1e9:.2f} GB) in {len(jobs)} chunks", flush=True)

    t0 = time.time()
    all_h: List[bytes] = []
    all_s: List[str] = []
    total = bad = 0
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for hs, ss, n, b in pool.imap(_scan_chunk, jobs):
            all_h.extend(hs)
            all_s.extend(ss)
            total += n
            bad += b
    print(f"[scan] {total} rows ({bad} malformed) in {time.time() - t0:.1f}s", flush=True)

    vocab = sorted(set(all_s))
    sid = {s: i for i, s in enumerate(vocab)}
    out = resolve(args.workdir, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        hashes=np.frombuffer(b"".join(all_h), dtype=np.uint8).reshape(-1, 16),
        session=np.array([sid[s] for s in all_s], np.int32),
        session_vocab=np.array(vocab),
        n_rows=np.array(total), n_malformed=np.array(bad),
    )
    print(f"[scan] {len(vocab)} distinct sessions -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
