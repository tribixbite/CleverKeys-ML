#!/usr/bin/env python3
"""Targeted English residual-transplant synthesis — pipeline-v2 element E2.

Same generator as `cyrillic_synth.py` (donor residual → target word's ideal
polyline through `layout_aug.warp_path`), pointed at English instead of
Cyrillic and aimed at the two places the English training mix is measurably
*starved* rather than merely under-emphasized (PIPELINE_V2_PROPOSAL.md §1.6,
§2.3-E2):

* ``--mode short`` — every 1–4-letter word of the shipped AOSP/FUTO lexicon,
  frequency-weighted. The ≤3-length stratum is 34 % of val but a far smaller
  share of the real train mix (PHASE_J.md §9); `--short-loss-weight` could
  only re-emphasize the rows that exist (PHASE_K.md §8.3), while this adds
  *word shapes that were never traced*.
* ``--mode tail`` — lexicon words with **fewer than 3 real traces** in the
  train mix, i.e. exactly the rows `build_tiers.py`'s ``MIN_WORD_FREQ = 3``
  dropped.

Why this is not the Cyrillic experiment again: en→en is a strictly easier
transplant (donor and target share the geometry AND the alphabet — only the
word changes), so the only structural requirement, matching collapsed-polyline
vertex counts, is satisfied from a 1.2 M-trace donor pool.

Guards, all motivated by the ru capacity-overfit lesson (PHASE_J.md §6.5 — a
ch-192 model *lost* 2.7 pt in-dict on synthetic-heavy data while greedy rose):

1. **Donors are TRAIN rows only.** val.npz/test.npz are never opened here; the
   default donor list is the sw2345 train mix. test-2400 stays sealed.
2. **Real-only selection.** These pools are training data exclusively — every
   checkpoint-selection and evaluation row in the campaign stays 100 % real.
3. **Capped fraction.** The proposal budgets ≈150 k rows per mode ⇒ ≈19 % of
   the 1.59 M-row v2 mix, under the 25 % cap.
4. **Validated before use**, against the *donor pool's own* endpoint statistics
   measured through the same code path (not a quoted constant), on three
   gates — see :func:`gate_report`:
   a. **displacement magnitude** — synthetic mean endpoint distance to the
      intended key within ``DIST_TOL`` of the real pool's;
   b. **wrong-geometry falsification** — the same synthetic traces scored
      against dvorak key centers must lose ≥ ``WRONG_GEOM_MIN_DROP`` start-hit;
      if they do not, the traces carry no geometry and the generator is broken;
   c. **precedent** — the nearest-key hit-rate gap must be no worse than the
      audited ru generator's (which had −0.21 start-hit vs its real corpus and
      still took Cyrillic 0 → ≈77.4, PHASE_I_DATA §6 / PHASE_J §6.9).
   The *decisive* gate is not any of these: it is the single-seed E2 on/off
   training ablation the proposal specifies (§2.3-E2).

Output schema is `prepare_data.py`'s, so the npz drops straight into
``train.py``/``train_v2.py --train-npz``.

Usage:
  python3 english_synth.py --mode short --rows 150000 --out synth_en_short.npz
  python3 english_synth.py --mode tail  --rows 150000 --out synth_en_tail.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyrillic_synth import build_donor_index, collapse  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from layout_aug import endpoint_stats, load_az_centers, warp_path  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import T_OUT, adjacent_repeats  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Gate (a): max excess of the synthetic mean endpoint-to-intended-key distance
#: over the real donor pool's, in normalized keyboard units (a key pitch on
#: en_qwerty is ~0.10 in x, so 0.02 is a fifth of a key).
DIST_TOL = 0.02
#: Gate (b): the wrong-geometry control must lose at least this much start-hit.
WRONG_GEOM_MIN_DROP = 0.30
#: Gate (c): nearest-key hit-rate gap ceiling. ``start_hit`` 0.21 IS the
#: audited ru precedent (synth 0.7095 vs real 0.917 — PHASE_I_DATA §6 — on the
#: generator that produced the campaign's proven synthesis win, and an en→en
#: transplant is strictly easier). ``end_hit`` 0.15 is NOT a precedent: the ru
#: generator's end-hit was slightly *better* than its real corpus (0.656 vs
#: 0.6465), so there is no measured ceiling on that axis and 0.15 is a chosen
#: tolerance of the same order as the start-hit one. Stated so the gate is not
#: read as more evidence-backed than it is.
RU_PRECEDENT_GAP = {"start_hit": 0.21, "end_hit": 0.15}


def load_en_lexicon(path: Path) -> List[Tuple[str, float]]:
    """(a-z word, AOSP freq) from a combined wordlist.

    Same surface normalization as `futo_decoder_eval.load_combined_vocab`
    (apostrophes/hyphens stripped rather than the word dropped), so this word
    set is exactly the beam's trie — the lexicon the app ships, not the val
    vocabulary.
    """
    out: Dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("word="):
                continue
            word, freq = None, 1.0
            for field in line.split(","):
                kv = field.split("=", 1)
                if len(kv) != 2:
                    continue
                if kv[0] == "word":
                    word = kv[1]
                elif kv[0] == "f":
                    try:
                        freq = float(kv[1])
                    except ValueError:
                        pass
            if not word:
                continue
            w = "".join(c for c in word.lower() if "a" <= c <= "z")
            if not w or len(w) + adjacent_repeats(w) > T_OUT:
                continue
            out[w] = max(out.get(w, 0.0), max(freq, 1.0))
    return sorted(out.items())


def real_word_counts(npz_paths: Sequence[Path]) -> Counter:
    """Per-word REAL trace counts over the train mix (pool repeats not counted
    twice — oversampling is a training-loop concern, not an information one)."""
    counts: Counter = Counter()
    for p in dict.fromkeys(npz_paths):
        with np.load(p, allow_pickle=True) as d:
            counts.update(str(w) for w in d["words"])
    return counts


def select_words(mode: str, lexicon: List[Tuple[str, float]],
                 counts: Counter, min_real: int, max_len: int
                 ) -> List[Tuple[str, float]]:
    """The E2 target word list for *mode*, frequency-weighted."""
    if mode == "short":
        sel = [(w, f) for w, f in lexicon if len(w) <= max_len]
    elif mode == "tail":
        sel = [(w, f) for w, f in lexicon if counts.get(w, 0) < min_real]
    else:
        raise SystemExit(f"unknown --mode {mode}")
    if not sel:
        raise SystemExit(f"--mode {mode}: empty word list")
    return sel


def synthesize_en(n_rows: int, rng: np.random.Generator,
                  words: List[Tuple[str, float]],
                  donor_feats: np.ndarray, donor_seqs: List[np.ndarray],
                  donor_by_count: Dict[int, np.ndarray],
                  qwerty: np.ndarray
                  ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Transplant donor residuals onto the target words' QWERTY polylines.

    Identical mechanism to `cyrillic_synth.synthesize`: donors are matched on
    collapsed-polyline vertex count and the correspondence runs through the
    per-vertex *virtual* alphabet, so `warp_path`'s monotone DP, endpoint pins,
    arc remap and movement-frame residual transfer are the audited code path.
    Here src and dst geometry are both canonical QWERTY — only the word (hence
    the polyline) changes.
    """
    words_arr = np.array([w for w, _ in words])
    weights = np.array([f for _, f in words], np.float64)
    weights /= weights.sum()
    seqs = [collapse(np.frombuffer(w.encode("ascii"), np.uint8).astype(np.int64) - 97)
            for w in words_arr]
    st = dict(drawn=0, no_donor=0, made=0)
    out_feats = np.empty((n_rows, 2, 64), np.float32)
    out_words: List[str] = []
    t0 = time.time()
    while len(out_words) < n_rows:
        k = int(rng.choice(len(words_arr), p=weights))
        st["drawn"] += 1
        seq = seqs[k]
        pool = donor_by_count.get(len(seq))
        if pool is None or len(pool) == 0:
            st["no_donor"] += 1
            continue
        di = int(pool[rng.integers(len(pool))])
        s = len(seq)
        warped = warp_path(donor_feats[di], np.arange(s, dtype=np.int64),
                           qwerty[donor_seqs[di]], qwerty[seq])
        np.clip(warped, 0.0, 1.0, out=warped)      # the training loop's range
        out_feats[len(out_words)] = warped
        out_words.append(str(words_arr[k]))
        st["made"] += 1
        if st["made"] % 50_000 == 0:
            print(f"  {st['made']}/{n_rows} "
                  f"({st['made'] / (time.time() - t0):.0f}/s)", flush=True)
    return out_feats, out_words, st


def real_endpoint_band(donor_paths: Sequence[Path], donor_feats: np.ndarray,
                       qwerty: np.ndarray, n: int,
                       rng: np.random.Generator) -> Dict[str, float]:
    """Endpoint statistics of the REAL donor rows, same code path as the synth.

    Measured rather than quoted: the published 0.895/0.769 band is a corpus
    average, while the pools differ materially (train_t3 0.948/0.833,
    tier_sw5q 0.871/0.692 — measured), and the honest comparator for a
    transplant is the pool the residuals were drawn from.
    """
    words: List[str] = []
    for p in donor_paths:
        with np.load(p, allow_pickle=True) as d:
            words.extend(str(w) for w in d["words"])
    if len(words) != len(donor_feats):
        raise SystemExit("donor word/feature count mismatch")
    idx = rng.choice(len(words), size=min(n, len(words)), replace=False)
    return endpoint_stats(donor_feats[idx], [words[i] for i in idx], qwerty)


def gate_report(synth: Dict[str, float], real: Dict[str, float],
                wrong: Dict[str, float]) -> Tuple[Dict[str, bool], Dict[str, float]]:
    """The three S0 acceptance gates -> (per-gate verdicts, measured deltas).

    The hit-rate gap is *reported* rather than required to vanish: the nearest-
    key hit is a direction-sensitive statistic (a residual of realistic
    magnitude transplanted into a different key neighbourhood crosses a Voronoi
    boundary more often), and the audited ru generator shows a large gap there
    is compatible with a large training win.
    """
    d = {
        "start_d_excess": synth["start_d"] - real["start_d"],
        "end_d_excess": synth["end_d"] - real["end_d"],
        "start_hit_gap": real["start_hit"] - synth["start_hit"],
        "end_hit_gap": real["end_hit"] - synth["end_hit"],
        "wrong_geom_start_drop": synth["start_hit"] - wrong["start_hit"],
    }
    gates = {
        "displacement_magnitude": (d["start_d_excess"] <= DIST_TOL
                                   and d["end_d_excess"] <= DIST_TOL),
        "wrong_geometry": d["wrong_geom_start_drop"] >= WRONG_GEOM_MIN_DROP,
        "ru_precedent": (d["start_hit_gap"] <= RU_PRECEDENT_GAP["start_hit"]
                         and d["end_hit_gap"] <= RU_PRECEDENT_GAP["end_hit"]),
    }
    return gates, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--mode", required=True, choices=("short", "tail"))
    ap.add_argument("--rows", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--donors",
                    default="train_t3.npz,train_t3hws.npz,tier_sw234.npz,"
                            "tier_sw5q.npz",
                    help="TRAIN caches only — the donor pool AND the real-trace "
                         "census for --mode tail")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    ap.add_argument("--max-len", type=int, default=4, dest="max_len",
                    help="--mode short: longest word admitted")
    ap.add_argument("--min-real", type=int, default=3, dest="min_real",
                    help="--mode tail: a word is 'tail' below this many real "
                         "train traces (build_tiers MIN_WORD_FREQ)")
    ap.add_argument("--out", default="", help="cache npz name (default: "
                                              "synth_en_<mode>.npz)")
    ap.add_argument("--validate-rows", type=int, default=5000,
                    dest="validate_rows")
    ap.add_argument("--wrong-geom", default="futo_dvorak.json",
                    dest="wrong_geom",
                    help="falsification control geometry (layouts/<name>)")
    args = ap.parse_args()

    cache = resolve(args.workdir, Path("cache"))
    rng = np.random.default_rng(args.seed)
    _, qwerty = load_layout(HERE / "en_qwerty.json")
    qwerty = np.asarray(qwerty, np.float32)[:26]

    lexicon = load_en_lexicon(resolve(args.workdir, args.vocab))
    donor_paths = [cache / p.strip() for p in args.donors.split(",")
                   if p.strip()]
    for p in donor_paths:
        if not p.exists():
            raise SystemExit(f"--donors: missing {p}")
        if p.name in ("val.npz", "test.npz"):
            raise SystemExit(f"refusing to donate from {p.name}: eval rows")
    counts = real_word_counts(donor_paths)
    sel = select_words(args.mode, lexicon, counts, args.min_real, args.max_len)
    covered = sum(1 for w, _ in sel if counts.get(w, 0) > 0)
    print(f"lexicon {len(lexicon)} words; --mode {args.mode} selects "
          f"{len(sel)} ({covered} of them have ≥1 real trace; "
          f"{len(sel) - covered} have none)")

    donor_feats, donor_seqs, by_count = build_donor_index(donor_paths)
    print(f"donor pool: {len(donor_feats)} real train traces; vertex counts "
          f"{min(by_count)}..{max(by_count)}")
    need: Dict[int, int] = defaultdict(int)
    for w, _ in sel:
        need[len(collapse(np.frombuffer(w.encode(), np.uint8).astype(np.int64)
                          - 97))] += 1
    uncovered = {k: v for k, v in sorted(need.items()) if k not in by_count}
    print(f"target vertex-count coverage: uncovered {uncovered or 'NONE'}")

    feats, words, st = synthesize_en(args.rows, rng, sel, donor_feats,
                                     donor_seqs, by_count, qwerty)
    ep_real = real_endpoint_band(donor_paths, donor_feats, qwerty,
                                 args.validate_rows, rng)
    del donor_feats

    tgt_flat = np.concatenate([np.frombuffer(w.encode(), np.uint8)
                               .astype(np.int64) - 97 for w in words])
    tgt_len = np.array([len(w) for w in words], np.int64)
    out = cache / (args.out or f"synth_en_{args.mode}.npz")
    prov = dict(generator="english_synth.py", mode=args.mode, seed=args.seed,
                rows=args.rows, donors=args.donors, vocab=args.vocab,
                max_len=args.max_len, min_real=args.min_real,
                n_target_words=len(sel), stats=st,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ── validation gates (measured BEFORE the file is written) ───────────────
    n = min(args.validate_rows, len(words))
    ep = endpoint_stats(feats[:n], words[:n], qwerty)
    wrong = np.asarray(load_az_centers(HERE / "layouts" / args.wrong_geom),
                       np.float32)
    ep_wrong = endpoint_stats(feats[:n], words[:n], wrong)
    gates, deltas = gate_report(ep, ep_real, ep_wrong)
    print(f"synth   endpoints (qwerty): {ep}")
    print(f"real    endpoints (donor pool, same code path): {ep_real}")
    print(f"control endpoints ({args.wrong_geom}): {ep_wrong}")
    print(f"deltas: {json.dumps({k: round(v, 4) for k, v in deltas.items()})}")
    for g, ok in gates.items():
        print(f"gate {g}: {'PASS' if ok else 'FAIL'}")
    prov["endpoints"] = ep
    prov["endpoints_real_donor_pool"] = ep_real
    prov["endpoints_wrong_geom"] = ep_wrong
    prov["gate_deltas"] = deltas
    prov["gates"] = gates

    np.savez_compressed(out, features=feats, targets=tgt_flat,
                        target_lengths=tgt_len, words=np.array(words),
                        provenance=np.array(json.dumps(prov, sort_keys=True)))
    print(f"synth: {st} -> {out}")
    (cache / f"synth_en_{args.mode}_stats.json").write_text(
        json.dumps(prov, indent=1))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
