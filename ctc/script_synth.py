#!/usr/bin/env python3
"""Residual-transplant synthesis for ANY script — the Phase-O generalization of
``cyrillic_synth.py``.

The counterfactual is unchanged: **a script with no swipe corpus at all.** Real
human deviation (undershoot, corner-cutting, jitter — the tangent/normal
residual decomposition ``layout_aug.warp_path`` computes) is sampled from
ENGLISH traces and re-anchored onto the ideal polylines of the target script's
words on the target script's own layout.  ``cyrillic_synth.py`` is left in place
as the historical record of the ru run; this file is the same mechanism with the
three ru-specific facts (alphabet, projection, lexicon) lifted into
:mod:`script_registry`.

What is new here, and why
-------------------------
Phase I-B could validate its synthesis against a real Yandex corpus.  No other
script has one, so Phase O has to build its own honest holdout — and a holdout
drawn from the same generator is worth nothing unless it is disjoint in the
things that could leak.  Two disjointness rules are enforced:

* **Donor split.** The English donor pool is partitioned by a deterministic hash
  of the donor row index; ``--donor-side train`` and ``--donor-side holdout``
  draw from disjoint halves.  A holdout trace therefore carries motor noise from
  a human trace the training set never saw.
* **Independent draws.** Separate RNG seeds for train / selection-val / holdout,
  so the word draws are independent too.

What that holdout can and cannot establish is stated once, here, and repeated in
every table it feeds: it measures **generalization to fresh samples of this
generator**, not to real human swipes in the target script.  The size of *that*
gap is knowable for exactly one script — Russian, which has both a synthesis
holdout and a real corpus — and Phase O measures it there and carries it as the
calibration for everything else (PHASE_O.md §2.1).

Outputs (under ``--workdir``, default ``~/ctc-train``)::

  cache_<code>/train_synth.npz   [N,2,64] features + targets   (--rows, seed 1234)
  cache_<code>/val.npz           5,000 rows for greedy checkpoint selection (seed 999)
  cache_<code>/holdout.npz       10,000 rows for the final decode read  (seed 777)
  cache_<code>/synth_stats.json  generation + endpoint-proximity + control
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_registry as SR  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from layout_aug import warp_path  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import T_OUT, adjacent_repeats  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Fraction of the English donor pool reserved for holdout synthesis.
DONOR_HOLDOUT_FRAC = 0.1

#: Seeds, one per split — independent word draws by construction.
SEEDS = {"train": 1234, "val": 999, "holdout": 777}


def collapse(seq: np.ndarray) -> np.ndarray:
    """Adjacent-repeat collapse — same rule as ``layout_aug._polyline_letters``."""
    if len(seq) <= 1:
        return seq
    keep = np.ones(len(seq), bool)
    keep[1:] = seq[1:] != seq[:-1]
    return seq[keep]


def build_donor_index(npz_paths: Sequence[Path], side: str
                      ) -> Tuple[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]]:
    """English donors -> (features [N,2,64], az polyline seqs, vertex-count index).

    *side* is ``"train"``, ``"holdout"`` or ``"all"``.  The split is a
    deterministic stride over the global donor row index, so it does not depend
    on load order beyond the fixed ``--donors`` list, and the two sides are
    exactly complementary.
    """
    stride = int(round(1.0 / DONOR_HOLDOUT_FRAC))
    feats_all: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    by_count: Dict[int, List[int]] = defaultdict(list)
    kept = 0
    row = 0
    for p in npz_paths:
        with np.load(p) as d:
            f = np.array(d["features"])
            words = [str(w) for w in d["words"]]
        keep_mask = np.zeros(len(words), bool)
        for i in range(len(words)):
            is_holdout = (row + i) % stride == 0
            keep_mask[i] = is_holdout if side == "holdout" else (
                True if side == "all" else not is_holdout)
        feats_all.append(f[keep_mask])
        for i, w in enumerate(words):
            if not keep_mask[i]:
                continue
            t = np.frombuffer(w.encode("ascii"), np.uint8).astype(np.int64) - 97
            seq = collapse(t)
            seqs.append(seq)
            by_count[len(seq)].append(kept)
            kept += 1
        row += len(words)
    feats = np.concatenate(feats_all) if len(feats_all) > 1 else feats_all[0]
    assert len(feats) == kept, (len(feats), kept)
    return feats, seqs, {k: np.asarray(v) for k, v in by_count.items()}


def synthesize(n_rows: int, rng: np.random.Generator,
               lexicon: Sequence[Tuple[str, float]], letters: str,
               donor_feats: np.ndarray, donor_seqs: List[np.ndarray],
               donor_by_count: Dict[int, np.ndarray],
               qwerty: np.ndarray, centers: np.ndarray,
               progress: int = 100_000,
               ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Draw words, transplant donor residuals, return ([N,2,64], words, stats)."""
    idx = {c: i for i, c in enumerate(letters)}
    words_arr = np.array([w for w, _ in lexicon])
    weights = np.array([f for _, f in lexicon], np.float64)
    weights /= weights.sum()
    seqs = [collapse(np.array([idx[c] for c in w], np.int64)) for w in words_arr]
    st = dict(drawn=0, no_donor=0, made=0)
    out_feats = np.empty((n_rows, 2, 64), np.float32)
    out_words: List[str] = []
    t0 = time.time()
    while len(out_words) < n_rows:
        k = rng.choice(len(words_arr), p=weights)
        st["drawn"] += 1
        seq = seqs[k]
        pool = donor_by_count.get(len(seq))
        if pool is None or len(pool) == 0:
            st["no_donor"] += 1
            continue
        di = int(pool[rng.integers(len(pool))])
        dseq = donor_seqs[di]
        S = len(seq)
        # Virtual per-vertex alphabet: id i is vertex i of BOTH polylines, so
        # warp_path's monotone-DP correspondence, endpoint pins, arc remap and
        # movement-frame residual transfer run byte-for-byte unchanged and every
        # Phase-H exactness invariant carries over. Letter identity never enters.
        src_virtual = qwerty[dseq]                        # [S,2]
        dst_virtual = centers[seq]                        # [S,2]
        warped = warp_path(donor_feats[di], np.arange(S, dtype=np.int64),
                           src_virtual, dst_virtual)
        np.clip(warped, 0.0, 1.0, out=warped)             # the training loop's
        out_feats[len(out_words)] = warped                # post-noise clip range
        out_words.append(str(words_arr[k]))
        st["made"] += 1
        if progress and st["made"] % progress == 0:
            print(f"  {st['made']}/{n_rows} "
                  f"({st['made'] / (time.time() - t0):.0f}/s)", flush=True)
    return out_feats, out_words, st


def endpoint_stats(feats: np.ndarray, words: Sequence[str], letters: str,
                   centers: np.ndarray) -> Dict[str, float]:
    """PHASE_H §2.3 endpoint proximity, generalized off the a-z assumption."""
    idx = {c: i for i, c in enumerate(letters)}
    hit_s = hit_e = n = 0
    dsum_s = dsum_e = 0.0
    for f, w in zip(feats, words):
        if not w:
            continue
        n += 1
        for px, py, letter, first in ((f[0][0], f[1][0], w[0], True),
                                      (f[0][-1], f[1][-1], w[-1], False)):
            d = np.hypot(centers[:, 0] - px, centers[:, 1] - py)
            li = idx[letter]
            hit = int(d.argmin()) == li
            if first:
                dsum_s += float(d[li]); hit_s += hit
            else:
                dsum_e += float(d[li]); hit_e += hit
    return {"n": n, "start_hit": round(hit_s / max(n, 1), 4),
            "end_hit": round(hit_e / max(n, 1), 4),
            "start_d": round(dsum_s / max(n, 1), 4),
            "end_d": round(dsum_e / max(n, 1), 4)}


def wrong_geometry(letters: str, rng: np.random.Generator) -> np.ndarray:
    """The PHASE_I_B falsification control: a deliberately wrong geometry.

    Key slots are randomly permuted, so every key sits on a real key position
    but the WRONG one.  Endpoint proximity against it must collapse to
    near-chance (ru measured 0.008 start-hit against 0.917 for the true frame);
    a frame that scores well under this control is not being validated by the
    endpoint test at all.
    """
    _, qwerty = load_layout(HERE / "en_qwerty.json")
    n = len(letters)
    src = qwerty[rng.permutation(len(qwerty))]
    return np.array([src[i % len(qwerty)] for i in range(n)], np.float32)


def write_split(cache: Path, name: str, feats: np.ndarray, words: List[str],
                letters: str, prov: Dict[str, object]) -> Path:
    idx = {c: i for i, c in enumerate(letters)}
    tgt_flat = np.concatenate([np.array([idx[c] for c in w], np.int64)
                               for w in words])
    tgt_len = np.array([len(w) for w in words], np.int64)
    out = cache / name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, targets=tgt_flat,
                        target_lengths=tgt_len, words=np.array(words),
                        provenance=np.array(json.dumps(prov, sort_keys=True)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", required=True, help="script code (script_registry)")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--rows", type=int, default=1_000_000, help="training rows")
    ap.add_argument("--val-rows", type=int, default=5_000,
                    help="greedy-selection rows (disjoint draw, same donor side)")
    ap.add_argument("--holdout-rows", type=int, default=10_000,
                    help="final-read rows (disjoint draw AND disjoint donor half)")
    ap.add_argument("--donors", default="train_t3futo.npz,train_t3hws.npz",
                    help="comma-separated cache npz names for the donor pool")
    ap.add_argument("--cache", default="", help="cache dir name (default cache_<code>)")
    ap.add_argument("--splits", default="train,val,holdout")
    ap.add_argument("--validate-rows", type=int, default=2000)
    args = ap.parse_args()

    spec = SR.get(args.code)
    cache_en = resolve(args.workdir, Path("cache"))
    cache = resolve(args.workdir, Path(args.cache or f"cache_{spec.code}"))
    cache.mkdir(parents=True, exist_ok=True)

    letters, centers = load_layout(HERE / spec.layout_json)
    if "".join(letters) != spec.letters:
        raise SystemExit(f"{spec.layout_json}: letters {''.join(letters)!r} != "
                         f"registry {spec.letters!r}")
    _, qwerty = load_layout(HERE / "en_qwerty.json")

    lexicon, lex_st = spec.load_lexicon(T_OUT)
    print(f"[{spec.code}] lexicon {lex_st}", flush=True)
    if not lexicon:
        raise SystemExit(f"[{spec.code}] empty lexicon — nothing to synthesize")

    report: Dict[str, object] = {
        "script": spec.code, "layout": spec.layout_json,
        "letters": spec.letters, "lexicon": lex_st,
        "lexicon_tier": spec.lexicon.tier,
        "donor_holdout_frac": DONOR_HOLDOUT_FRAC,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    donor_paths = [cache_en / p.strip() for p in args.donors.split(",")]
    want = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Vertex-count coverage of the lexicon against each donor side.
    for side in sorted({"holdout" if s == "holdout" else "train" for s in want}):
        feats_d, seqs_d, by_count = build_donor_index(donor_paths, side)
        print(f"[{spec.code}] donor side {side}: {len(feats_d)} traces, "
              f"vertex counts {min(by_count)}..{max(by_count)}", flush=True)
        idx = {c: i for i, c in enumerate(spec.letters)}
        need: Dict[int, int] = defaultdict(int)
        for w, _ in lexicon:
            need[len(collapse(np.array([idx[c] for c in w])))] += 1
        uncovered = {k: v for k, v in need.items() if k not in by_count}
        report[f"donor_{side}"] = {
            "traces": int(len(feats_d)),
            "uncovered_vertex_counts": uncovered or None,
            "uncovered_words": int(sum(uncovered.values())),
        }
        print(f"[{spec.code}]   uncovered vertex counts: {uncovered or 'NONE'}",
              flush=True)

        for split in want:
            if ("holdout" if split == "holdout" else "train") != side:
                continue
            n = {"train": args.rows, "val": args.val_rows,
                 "holdout": args.holdout_rows}[split]
            rng = np.random.default_rng(SEEDS[split])
            feats, words, st = synthesize(n, rng, lexicon, spec.letters, feats_d,
                                          seqs_d, by_count, qwerty, centers)
            fname = {"train": "train_synth.npz", "val": "val.npz",
                     "holdout": "holdout.npz"}[split]
            prov = dict(generator="script_synth.py", script=spec.code,
                        split=split, seed=SEEDS[split], rows=n,
                        donor_side=side, donors=args.donors,
                        layout=spec.layout_json, stats=st,
                        lexicon=spec.lexicon.tier)
            out = write_split(cache, fname, feats, words, spec.letters, prov)
            report[f"gen_{split}"] = st
            print(f"[{spec.code}] {split}: {st} -> {out}", flush=True)

            if split in ("train", "holdout"):
                v = args.validate_rows
                ep = endpoint_stats(feats[:v], words[:v], spec.letters, centers)
                ctrl = endpoint_stats(
                    feats[:v], words[:v], spec.letters,
                    wrong_geometry(spec.letters, np.random.default_rng(4242)))
                report[f"endpoints_{split}"] = ep
                report[f"endpoints_{split}_wrong_geo_control"] = ctrl
                print(f"[{spec.code}] {split} endpoints {ep}", flush=True)
                print(f"[{spec.code}] {split} wrong-geo control {ctrl}", flush=True)

    (cache / "synth_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[{spec.code}] wrote {cache / 'synth_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
