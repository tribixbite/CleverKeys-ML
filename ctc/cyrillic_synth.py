#!/usr/bin/env python3
"""Residual-transplant generator: synthetic Cyrillic swipes from English motor noise.

The counterfactual this implements: **a script/layout with no swipe corpus at
all.**  Real human deviation (undershoot, corner-cutting, jitter — the
tangent/normal residual decomposition `layout_aug.warp_path` computes) is
sampled from ENGLISH traces and re-anchored onto the ideal polylines of
RUSSIAN words on the ЙЦУКЕН geometry.  Russia happens to have a real corpus
(the Yandex Cup release, `prepare_yandex.py`), which is exactly what makes the
counterfactual *testable*: train-on-synthetic / eval-on-real measures how far
transplantation gets a script for which nothing real exists.

Mechanism — `warp_path` reused verbatim through per-vertex virtual indices
--------------------------------------------------------------------------
`warp_path(feats, target, src_centers, dst_centers)` warps a path for one word
between two geometries of the SAME alphabet.  Cross-alphabet transplant needs
one generalization, and it is an input transformation, not new math: a donor
English trace whose ideal polyline has the same number of vertices as the
Russian word's polyline is given *virtual* letter ids `0..S`, with
`src_virtual[i] = qwerty[donor_seq[i]]` and `dst_virtual[i] =
jcuken[russian_seq[i]]`.  The monotone-DP correspondence, endpoint pins,
vertex-absolute arc remap and movement-frame residual transfer then run
unchanged — every Phase-H exactness invariant carries over because the code
path is literally the same.

Donor matching is on **polyline vertex count** (adjacent-repeat-collapsed
length) — the only structural requirement — with donors drawn uniformly from
the count-matched pool.  Letter identities do not constrain the match: the
correspondence is geometric.

Word sampling: the app's bundled ru CKDT-v2 dictionary (langpack-ru, 50 k
words), weighted by `freq = 255 - rank`.  Deliberately NOT the Yandex train
distribution — in the no-corpus counterfactual only a lexicon exists.

What this CAN and CANNOT validate
---------------------------------
CAN: endpoint-proximity stats vs the real Yandex corpus (§ `--validate`), and
the train-on-synth → eval-on-real accuracy measurement (`eval_cyrillic.py`).
CANNOT: per-script motor idiosyncrasies (Cyrillic swipers may cut corners
differently); nothing about scripts whose writing direction or key density
departs further than ЙЦУКЕН's from QWERTY's.  Stated, not smoothed over.

Outputs:
  cache_ru/train_synth.npz     [N,2,64] synthetic features + ru targets
                               (same schema as prepare_yandex.py caches)
  cache_ru/synth_stats.json    donor-pool coverage + validation numbers
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
from eval_altlayout import read_ckdt_v2  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from layout_aug import warp_path  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import T_OUT, adjacent_repeats  # noqa: E402
from prepare_yandex import RU_LETTERS_31, project_ru  # noqa: E402

HERE = Path(__file__).resolve().parent
RU_LANGPACK_BIN = Path("/home/will/git/swype/CleverKeys/scripts/dictionaries")
APP_RU_DICT = "langpack-ru.zip"          # dictionary.bin inside (CKDT v2)


def collapse(seq: np.ndarray) -> np.ndarray:
    """Adjacent-repeat collapse — same rule as layout_aug._polyline_letters."""
    if len(seq) <= 1:
        return seq
    keep = np.ones(len(seq), bool)
    keep[1:] = seq[1:] != seq[:-1]
    return seq[keep]


def load_ru_lexicon() -> List[Tuple[str, float]]:
    """(projected_word, weight) from the app's ru CKDT dictionary."""
    import io
    import zipfile
    with zipfile.ZipFile(RU_LANGPACK_BIN / APP_RU_DICT) as zf:
        blob = zf.read("dictionary.bin")
    tmp = Path("/tmp/ru_dictionary_ckdt.bin")
    tmp.write_bytes(blob)
    out: Dict[str, float] = {}
    for word, rank in read_ckdt_v2(tmp):
        p = project_ru(word)
        if p is None or len(p) < 2:
            continue
        if len(p) + adjacent_repeats(p) > T_OUT:
            continue
        w = float(max(1, 255 - rank))
        if p not in out or w > out[p]:
            out[p] = w
    return sorted(out.items())


def build_donor_index(npz_paths: List[Path], letters_az: str = "abcdefghijklmnopqrstuvwxyz"
                      ) -> Tuple[np.ndarray, List[np.ndarray], Dict[int, np.ndarray]]:
    """Load English donors -> (features [N,2,64], per-row az polyline seqs,
    vertex-count -> donor row indices)."""
    feats_all: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    by_count: Dict[int, List[int]] = defaultdict(list)
    row = 0
    for p in npz_paths:
        with np.load(p) as d:
            f = np.array(d["features"])
            words = [str(w) for w in d["words"]]
        feats_all.append(f)
        for w in words:
            t = np.frombuffer(w.encode("ascii"), np.uint8).astype(np.int64) - 97
            seq = collapse(t)
            seqs.append(seq)
            by_count[len(seq)].append(row)
            row += 1
    feats = np.concatenate(feats_all) if len(feats_all) > 1 else feats_all[0]
    assert len(feats) == row
    return feats, seqs, {k: np.asarray(v) for k, v in by_count.items()}


def synthesize(n_rows: int, rng: np.random.Generator,
               lexicon: List[Tuple[str, float]],
               donor_feats: np.ndarray, donor_seqs: List[np.ndarray],
               donor_by_count: Dict[int, np.ndarray],
               qwerty: np.ndarray, ru_centers: np.ndarray,
               ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Draw words, transplant donor residuals, return ([N,2,64], words, stats)."""
    ru_idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    words_arr = np.array([w for w, _ in lexicon])
    weights = np.array([f for _, f in lexicon], np.float64)
    weights /= weights.sum()
    # Pre-collapse every lexicon word once.
    ru_seqs = [collapse(np.array([ru_idx[c] for c in w], np.int64))
               for w in words_arr]
    st = dict(drawn=0, no_donor=0, made=0)
    out_feats = np.empty((n_rows, 2, 64), np.float32)
    out_words: List[str] = []
    t0 = time.time()
    while len(out_words) < n_rows:
        k = rng.choice(len(words_arr), p=weights)
        st["drawn"] += 1
        seq = ru_seqs[k]
        pool = donor_by_count.get(len(seq))
        if pool is None or len(pool) == 0:
            st["no_donor"] += 1
            continue
        di = int(pool[rng.integers(len(pool))])
        dseq = donor_seqs[di]
        S = len(seq)
        # Virtual per-vertex alphabet: id i is vertex i of both polylines.
        src_virtual = qwerty[dseq]                        # [S,2]
        dst_virtual = ru_centers[seq]                     # [S,2]
        warped = warp_path(donor_feats[di], np.arange(S, dtype=np.int64),
                           src_virtual, dst_virtual)
        np.clip(warped, 0.0, 1.0, out=warped)             # the training loop's
        out_feats[len(out_words)] = warped                # post-noise clip range
        out_words.append(str(words_arr[k]))
        st["made"] += 1
        if st["made"] % 100_000 == 0:
            print(f"  {st['made']}/{n_rows} ({st['made'] / (time.time() - t0):.0f}/s)",
                  flush=True)
    return out_feats, out_words, st


def ru_endpoint_stats(feats: np.ndarray, words: Sequence[str],
                      centers: np.ndarray) -> Dict[str, float]:
    """endpoint_stats generalized off the a-z assumption (ord-97 indexing)."""
    idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--donors", default="train_t3futo.npz,train_t3hws.npz",
                    help="comma-separated cache npz names for the donor pool")
    ap.add_argument("--layout", default="ru_jcuken_default.json")
    ap.add_argument("--out", default="train_synth.npz")
    ap.add_argument("--validate-real", type=Path, default=None,
                    help="canonical ru jsonl (prepare_yandex output) for the "
                         "endpoint-proximity comparison")
    ap.add_argument("--validate-rows", type=int, default=2000)
    args = ap.parse_args()

    cache = resolve(args.workdir, Path("cache"))
    cache_ru = resolve(args.workdir, Path("cache_ru"))
    rng = np.random.default_rng(args.seed)

    _, qwerty = load_layout(HERE / "en_qwerty.json")
    ru_letters, ru_centers = load_layout(HERE / "layouts" / args.layout)
    assert "".join(ru_letters) == RU_LETTERS_31

    lexicon = load_ru_lexicon()
    print(f"ru lexicon: {len(lexicon)} projectable words")

    donor_paths = [cache / p.strip() for p in args.donors.split(",")]
    donor_feats, donor_seqs, by_count = build_donor_index(donor_paths)
    print(f"donor pool: {len(donor_feats)} traces; vertex counts "
          f"{min(by_count)}..{max(by_count)}")

    # Coverage: which lexicon vertex-counts lack donors?
    need = defaultdict(int)
    ru_idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    for w, f in lexicon:
        need[len(collapse(np.array([ru_idx[c] for c in w])))] += 1
    uncovered = {k: v for k, v in need.items() if k not in by_count}
    print(f"lexicon vertex-count coverage: uncovered {uncovered or 'NONE'}")

    feats, words, st = synthesize(args.rows, rng, lexicon, donor_feats,
                                  donor_seqs, by_count, qwerty, ru_centers)
    idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    tgt_flat = np.concatenate([np.array([idx[c] for c in w], np.int64)
                               for w in words])
    tgt_len = np.array([len(w) for w in words], np.int64)
    out = cache_ru / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    prov = dict(generator="cyrillic_synth.py", seed=args.seed, rows=args.rows,
                donors=args.donors, layout=args.layout, stats=st,
                lexicon="app langpack-ru CKDT v2 (freq = 255 - rank)",
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    np.savez_compressed(out, features=feats, targets=tgt_flat,
                        target_lengths=tgt_len, words=np.array(words),
                        provenance=np.array(json.dumps(prov, sort_keys=True)))
    print(f"synth: {st} -> {out}")

    # ── validation: endpoint proximity, synthetic vs real ─────────────────────
    report: Dict[str, object] = {"generation": st}
    synth_ep = ru_endpoint_stats(feats[:args.validate_rows],
                                 words[:args.validate_rows], ru_centers)
    report["synth_endpoints"] = synth_ep
    print(f"synth endpoints: {synth_ep}")
    if args.validate_real and args.validate_real.exists():
        from futo_decoder_eval import featurize
        real_feats, real_words = [], []
        with open(args.validate_real, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                pts = o["points"]
                real_feats.append(featurize([p["x"] for p in pts],
                                            [p["y"] for p in pts],
                                            [p["t"] for p in pts]))
                real_words.append(o["word"])
                if len(real_words) >= args.validate_rows:
                    break
        real_ep = ru_endpoint_stats(np.stack(real_feats), real_words, ru_centers)
        report["real_endpoints"] = real_ep
        print(f"real  endpoints: {real_ep}")
    (cache_ru / "synth_stats.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
