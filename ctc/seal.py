#!/usr/bin/env python3
"""Content-addressed seal on the held-out test split.

Campaign 2's seal was enforced by ``"test" in Path(args.test).name`` — a
**filename substring**. `AUDIT_FINAL.md` §6.7 recorded that as a structural gap:
the seal "was held by discipline", not by a mechanism. A filename check is
defeated by copying the split to any other name, by slicing it, or by feeding
rows through a different loader, and it cannot notice that a "new" evaluation
file is actually the sealed rows.

This module replaces it with a check on the **content of the rows actually
loaded**, wherever they came from:

* every row is reduced to the same normalized trace hash the dedup uses
  (a–z-normalized word + exact float64 x/y bytes), truncated to 8 bytes;
* the sealed split's 2,400 hashes are committed in ``test2400_seal.json``;
* a decode is refused when the loaded rows overlap the sealed set by more than
  :data:`OVERLAP_LIMIT`.

Because the check is on hashes of the loaded rows, it fires on a renamed copy, on
a shuffled copy, and on any slice of the split — a 120-row prefix is 100 %
overlapping and is refused, where the filename guard passed it. (An undisclosed
120-row smoke decode is exactly what `AUDIT_FINAL.md` §6.7 turned up.)

**The seal on test-2400 is spent** (`AUDIT_FINAL.md` §7): the pre-registered
decode happened and no further decode of that split is legitimate for any
variant, preset, checkpoint or stratum. The guard therefore exists to stop an
*accidental* re-decode; ``--unseal-test`` is the deliberate, logged override and
carries the same warning.

**The unsealing ledger.** Each sealed record carries an ``unsealings`` list — the
append-only log of every authorised read of that split: when, who authorised it,
what was decoded, and where the numbers were published. It is the answer to "how
worn is this split?", which the fingerprints alone cannot give. ``--emit``
preserves it across a fingerprint regeneration; entries are added by hand when a
decode is authorised, and are never removed.

Regenerate the fingerprint (does not decode anything — it only hashes rows):

    python seal.py --emit --split data/test_hwsfuto.jsonl --name test-2400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import normalize_word  # noqa: E402

#: Committed fingerprint of every sealed split, beside this file.
SEAL_FILE = Path(__file__).resolve().parent / "test2400_seal.json"

#: Fraction of loaded rows that may appear in a sealed split before the decode is
#: refused. Non-zero on purpose: **7 traces are bit-exactly shared between
#: val-9918 and test-2400** (`AUDIT_FINAL.md` §6.7), so val legitimately overlaps
#: the sealed set by 7/9918 = 0.07 %. 1 % leaves ~14x headroom over that while
#: still refusing any slice of the split, the smallest of which (one row) is
#: 100 % overlapping.
OVERLAP_LIMIT = 0.01

#: Truncated-hash width. 8 bytes over a 2,400-row set makes a false positive
#: ~2400/2^64 per row — far below one row in any corpus this pipeline handles.
DIGEST_BYTES = 8


def row_fingerprint(word: str, xs: Sequence[float], ys: Sequence[float]) -> str:
    """One row -> its truncated hex trace hash.

    Deliberately the **timing-independent** ``(normalized word, x, y)`` form:
    it is the stricter of the two conventions in this pipeline (``DATA_TIERS.md``),
    so a row that was re-timed but not re-drawn is still recognised as the sealed
    trace.
    """
    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    h.update(normalize_word(word).encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(xs, np.float64).tobytes())
    h.update(np.asarray(ys, np.float64).tobytes())
    return h.hexdigest()


def fingerprint_rows(rows) -> List[str]:
    """``load_test``-shaped rows ``(word, xs, ys, ts)`` -> their fingerprints."""
    return [row_fingerprint(w, xs, ys) for w, xs, ys, _ in rows]


def load_seals(path: Path = SEAL_FILE) -> Dict[str, Dict[str, object]]:
    """Read the committed fingerprint file; empty dict when it is absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def check(rows, allow: bool, what: str = "this run",
          seal_path: Path = SEAL_FILE) -> None:
    """Refuse to proceed when *rows* overlap a sealed split.

    :param rows: the rows about to be decoded, as ``load_test`` returns them.
    :param allow: the caller's ``--unseal-test`` flag. True still prints the
        overlap, so an intentional unsealing is always visible in the log.
    :param what: label for the message (the script or split name).
    :raises SystemExit: when the overlap exceeds :data:`OVERLAP_LIMIT` and
        *allow* is false.
    """
    seals = load_seals(seal_path)
    if not seals:
        print(f"[seal] WARNING: {seal_path.name} missing — no seal enforced")
        return
    fps = fingerprint_rows(rows)
    n = max(len(fps), 1)
    for name, rec in seals.items():
        sealed: Set[str] = set(rec["fingerprints"])
        hits = sum(1 for f in fps if f in sealed)
        frac = hits / n
        if frac <= OVERLAP_LIMIT:
            continue
        msg = (f"[seal] {what}: {hits}/{len(fps)} loaded rows ({frac:.1%}) are in "
               f"the SEALED split '{name}' (n={rec['n']})")
        if allow:
            print(msg + "  — proceeding because --unseal-test was passed.")
            print("[seal] the test-2400 seal is SPENT (AUDIT_FINAL.md §7); this "
                  "decode is not a legitimate basis for any published number.")
            return
        raise SystemExit(
            msg + "\n[seal] refusing to decode. The seal on test-2400 was spent by "
            "the pre-registered decode in AUDIT_FINAL.md §7, and no further decode "
            "of it is legitimate.\n[seal] pass --unseal-test to override "
            "deliberately (it will be logged).")


def add_argument(ap: argparse.ArgumentParser) -> None:
    """Register the shared ``--unseal-test`` flag on a script's parser."""
    ap.add_argument("--unseal-test", action="store_true", dest="unseal_test",
                    help="decode rows that belong to a sealed split. The "
                         "test-2400 seal is already spent (AUDIT_FINAL.md §7), so "
                         "this cannot produce a publishable number.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--emit", action="store_true",
                    help="hash --split and write it into the seal file")
    ap.add_argument("--split", default="data/test_hwsfuto.jsonl")
    ap.add_argument("--name", default="test-2400")
    args = ap.parse_args()

    from futo_decoder_eval import load_test
    path = resolve(args.workdir, args.split)
    rows = load_test(path)
    fps = fingerprint_rows(rows)
    words_sha = hashlib.sha256(
        "\n".join(normalize_word(w) for w, _, _, _ in rows).encode()).hexdigest()
    print(f"{path}: {len(rows)} rows, {len(set(fps))} unique fingerprints, "
          f"words_sha256 {words_sha[:16]}…")
    if not args.emit:
        return 0
    seals = load_seals()
    # The unsealing ledger outlives a fingerprint regeneration: it records reads of
    # the split, not its contents, and losing it would erase how worn the split is.
    prior = seals.get(args.name, {}).get("unsealings", [])
    seals[args.name] = {
        "source": str(path),
        "n": len(rows),
        "words_sha256": words_sha,
        "digest_bytes": DIGEST_BYTES,
        "note": "a-z-normalized word + float64 x/y; see seal.py",
        "unsealings": prior,
        "fingerprints": sorted(set(fps)),
    }
    SEAL_FILE.write_text(json.dumps(seals, indent=1, sort_keys=True))
    print(f"wrote {SEAL_FILE} ({SEAL_FILE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
