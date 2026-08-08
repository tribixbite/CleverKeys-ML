#!/usr/bin/env python3
"""Cache exact-port featurization of the hwsfuto JSONL splits -> npz.

Featurization uses ``futo_decoder_eval.featurize`` — the SAME function the Kotlin
``CtcFeaturizer`` is a bit-identical port of — so train-time and app-time
featurization agree.

Audit fixes applied here:
  * #3  cross-split dedup: train rows whose (word + exact float64 x/y/t bytes)
        hash appears in val or test are dropped, as are exact self-duplicates
        within train. val/test are NEVER filtered.
  * #11 CTC feasibility: a row is dropped when ``len(word) + adjacent_repeats``
        exceeds the encoder's T'=32 output frames (such targets are silently
        zeroed by ``zero_infinity=True`` and would waste the sample).
  * #16 ``--workdir`` pathing; the layout defaults to the script's directory.
  * #17 provenance JSON embedded in every npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import featurize  # noqa: E402  exact resampler port
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve, sha256_file  # noqa: E402

#: Encoder output frames (T'); CTC needs T' >= len(word) + adjacent repeats.
T_OUT = 32

SPLITS = ("train", "val", "test")


def normalize_word(w: str) -> str:
    """Lowercase and strip everything outside ``a-z`` (``don't`` -> ``dont``).

    Mirrors ``futo_decoder_eval.load_combined_vocab`` and the Kotlin
    ``CtcLexiconTrie.loadStrippingNonAlphabet``.
    """
    return "".join(c for c in w.lower() if "a" <= c <= "z")


def adjacent_repeats(word: str) -> int:
    """Count adjacent equal characters — each needs one extra CTC blank frame."""
    return sum(1 for a, b in zip(word, word[1:]) if a == b)


def trace_hash(word: str, xs: List[float], ys: List[float], ts: List[float]) -> bytes:
    """Content hash over the NORMALIZED word plus the exact float64 path bytes.

    Exact bytes (not rounded text) so the dedup cannot collapse genuinely
    distinct traces, and cannot miss bit-identical ones (audit finding #3).

    The word is keyed through :func:`normalize_word`, i.e. on the same a–z form
    the CTC target is built from. Keying on the *raw* word was the campaign-2
    dedup defect: ``'arabian.'`` in a training tier and ``'arabian'`` in the
    holdout hash to different values, so a row with a bit-identical input tensor
    **and an identical label** survived the dedup. It let 588 val and 145 test
    rows into `train_t3`; see `AUDIT_PREDECODE.md` §3 and `AUDIT_FINAL.md` §4.

    ⚠ Two rows that differ only in punctuation now collide, which is exactly the
    intent: they produce the same training target from the same path, so keeping
    both is duplication, and keeping one across a split boundary is leakage.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_word(word).encode("utf-8"))
    h.update(b"\x00")
    for arr in (xs, ys, ts):
        h.update(np.asarray(arr, np.float64).tobytes())
        h.update(b"\x00")
    return h.digest()


def iter_rows(path: Path):
    """Yield ``(raw_word, xs, ys, ts)`` for every non-blank JSONL line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pts = obj["points"]
            yield (obj["word"],
                   [float(p["x"]) for p in pts],
                   [float(p["y"]) for p in pts],
                   [float(p["t"]) for p in pts])


def source_meta(path: Path) -> Dict[str, object]:
    """Filename/size/mtime of a (possibly symlinked) source split."""
    real = path.resolve()
    st = real.stat()
    return {"path": str(path), "resolved": str(real), "size": st.st_size,
            "mtime": st.st_mtime}


def _featurize_batch(batch):
    """Worker: featurize a list of (xs, ys, ts) tuples -> list of [2,64] arrays."""
    return [featurize(xs, ys, ts) for xs, ys, ts in batch]


def prepare(jsonl_path: Path, out_path: Path, exclude: Set[bytes],
            dedup_self: bool, provenance_extra: Dict[str, object],
            jobs: int = 1) -> Set[bytes]:
    """Featurize one split into ``out_path``; return the split's trace hashes.

    :param exclude: hashes whose rows must be dropped (cross-split leakage).
    :param dedup_self: drop exact duplicates appearing twice inside this split.
    :param provenance_extra: static provenance fields merged into the npz record.
    """
    feats: List[np.ndarray] = []
    words: List[str] = []
    hashes: Set[bytes] = set()
    seen: Set[bytes] = set()
    pending: List[tuple] = []          # rows awaiting parallel featurization
    n_total = 0
    d_empty = d_infeasible = d_leak = d_self = 0

    t0 = time.time()
    for raw_word, xs, ys, ts in iter_rows(jsonl_path):
        n_total += 1
        w = normalize_word(raw_word)
        h = trace_hash(raw_word, xs, ys, ts)
        hashes.add(h)
        if not w:
            d_empty += 1
            continue
        if len(w) + adjacent_repeats(w) > T_OUT:
            d_infeasible += 1
            continue
        if h in exclude:
            d_leak += 1
            continue
        if dedup_self:
            if h in seen:
                d_self += 1
                continue
            seen.add(h)
        if jobs > 1:
            pending.append((xs, ys, ts))
        else:
            feats.append(featurize(xs, ys, ts))      # [2, 64] float32
        words.append(w)

    if jobs > 1 and pending:
        # Featurization is pure Python and dominates a ~700 k-row tier; fan it out.
        chunk = max(1, len(pending) // (jobs * 8))
        batches = [pending[i:i + chunk] for i in range(0, len(pending), chunk)]
        with mp.get_context("fork").Pool(jobs) as pool:
            for out in pool.imap(_featurize_batch, batches):
                feats.extend(out)
        pending.clear()

    if not words:
        raise SystemExit(f"{jsonl_path}: no rows survived filtering")

    features = np.stack(feats).astype(np.float32)    # [N, 2, 64]
    # Ragged targets -> flat buffer + lengths (CTC-loss friendly).
    tgt_flat = np.concatenate([np.frombuffer(w.encode("ascii"), np.uint8) - ord("a")
                               for w in words]).astype(np.int64)
    tgt_len = np.array([len(w) for w in words], np.int64)

    prov = dict(provenance_extra)
    prov.update({
        "split_source": source_meta(jsonl_path),
        "rows_in": n_total,
        "rows_out": len(words),
        "dropped_empty_word": d_empty,
        "dropped_ctc_infeasible": d_infeasible,
        "dropped_cross_split_duplicate": d_leak,
        "dropped_self_duplicate": d_self,
        "t_out_frames": T_OUT,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=features, targets=tgt_flat,
                        target_lengths=tgt_len, words=np.array(words),
                        provenance=np.array(json.dumps(prov, sort_keys=True)))
    print(f"{jsonl_path.name}: {n_total} in -> {len(words)} cached "
          f"(empty {d_empty}, infeasible {d_infeasible}, cross-split-dup {d_leak}, "
          f"self-dup {d_self}) in {time.time() - t0:.1f}s -> {out_path}")
    return hashes


def collect_hashes(path: Path) -> Set[bytes]:
    """Hash every row of a split without featurizing it."""
    return {trace_hash(w, xs, ys, ts) for w, xs, ys, ts in iter_rows(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                    help="runtime dir holding data/ and cache/ (default: ~/ctc-train)")
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT,
                    help="canonical layout json (hashed into provenance only)")
    ap.add_argument("--data", type=Path, default=Path("data"),
                    help="split dir, relative to --workdir unless absolute")
    ap.add_argument("--cache", type=Path, default=Path("cache"),
                    help="npz output dir, relative to --workdir unless absolute")
    ap.add_argument("--extra-train", default="", dest="extra_train",
                    help="featurize ONE extra training jsonl (a data tier built by "
                         "build_tiers.py) instead of the three canonical splits; "
                         "still deduped against canonical val/test")
    ap.add_argument("--out-name", default="", dest="out_name",
                    help="npz basename for --extra-train (default: input stem)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel featurization workers for --extra-train")
    args = ap.parse_args()

    data_dir = resolve(args.workdir, args.data)
    cache_dir = resolve(args.workdir, args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    static_prov = {
        "generator": "prepare_data.py",
        "featurizer_sha256": sha256_file(here / "futo_decoder_eval.py"),
        "layout": str(args.layout),
        "layout_sha256": sha256_file(args.layout),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    paths = {s: data_dir / f"{s}_hwsfuto.jsonl" for s in SPLITS}
    for s, p in paths.items():
        if not p.exists():
            raise SystemExit(f"missing split: {p}")

    # Hold-out hashes must be known BEFORE train is filtered (audit fix #3).
    print("[dedup] hashing held-out splits ...", flush=True)
    holdout: Set[bytes] = set()
    for s in ("val", "test"):
        h = collect_hashes(paths[s])
        print(f"  {s}: {len(h)} unique traces")
        holdout |= h
    print(f"  holdout union: {len(holdout)} unique traces", flush=True)

    if args.extra_train:
        # Tier mode: featurize one externally-built training pool. The same
        # holdout exclusion and self-dedup apply, so a tier can never smuggle a
        # canonical val/test trace in even if build_tiers.py missed one.
        src = resolve(args.workdir, args.extra_train)
        name = args.out_name or src.stem
        static_prov["tier_source"] = str(src)
        prepare(src, cache_dir / f"{name}.npz", exclude=holdout, dedup_self=True,
                provenance_extra=static_prov, jobs=args.jobs)
        return 0

    # val/test are cached verbatim (never filtered), train is deduped against them.
    for s in SPLITS:
        prepare(paths[s], cache_dir / f"{s}.npz",
                exclude=holdout if s == "train" else set(),
                dedup_self=(s == "train"),
                provenance_extra=static_prov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
