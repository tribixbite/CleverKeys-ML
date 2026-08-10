#!/usr/bin/env python3
"""Convert the Yandex Cup 2023 "NeuroSwipe" corpus into the CTC training format.

Source (Phase I-B acquisition, 2026-08-09)
------------------------------------------
``disk.yandex.ru/d/IYiSpLob-zAxqg`` — the organizer's public distribution that
the 7th-place solution repo (``proshian/neural-swipe-typing``, vendored locally
at ``/home/will/git/neural-swipe-typing``) downloads in
``download_original_data.py``.  ``data.zip`` 1,745,670,429 B, sha256
``2e65d7a2…b37521`` (verified against the Disk API's published checksum).
Contents: ``train.jsonl`` (6,000,000 rows), ``valid.jsonl`` (10,000 rows,
targets in ``valid.ref``), ``test.jsonl`` (no targets), ``voc.txt`` (503,598
words).  **License: none stated** — competition data, publicly distributed by
the organizer; treat as research-use-only until clarified.  Nothing built from
it ships without an explicit owner decision.

Row schema: ``{"word": …, "curve": {"x": [px], "y": [px], "t": [ms],
"grid": {grid_name, width, height, keys: [{label, hitbox{x,y,w,h}}]}}}``.
Two grids: ``default`` (31 letter keys — no ё, no ъ; 93.8 % of train) and
``extra`` (33 letter keys; 6.2 %).  Targets never contain ё (the organizers
folded it); ъ occurs in ~0.06 % of words; ``-`` in ~0.3 %.

The mapping — same policy as the alt-layout eval (ALT_LAYOUT_EVAL.md §2–3)
--------------------------------------------------------------------------
* Coordinates: the **letter-area frame**, the same convention the canonical
  en corpus uses (DATA_TIERS.md §1): x over the full grid width (the letter
  rows tile it), y over the letter-key block's vertical extent, so the three
  ЙЦУКЕН rows land at cy 0.167/0.5/0.833 — exactly the (2r+1)/2R family every
  trained geometry (canonical and layout-alt synthetic) lives in.  A plain
  ``y/height`` would leave the letter block squashed into cy 0.138–0.60
  (the grid's 667 px include the action row), an sy≈0.46 compression that
  sits outside the measured affine-tolerance envelope
  (ALT_LAYOUT_EVAL.md §7.2).  ``t − t[0]``; path y clipped to [0,1] (swipes
  drifting onto the action row are pinned to the block edge, matching the
  training loop's post-noise clip).
* Alphabet: the 31 default-grid letters, **alphabetical** slot order
  (``абвгдежзийклмнопрстуфхцчшщыьэюя``).  Projection for targets AND lexicon:
  lowercase, strip ``-``/``'``, fold ё→е and ъ→ь (the keys a default-grid swipe
  physically visits).  NO unicode NFD — it would decompose й into и + breve.
* Geometry: ``layouts/ru_jcuken_default.json`` (and ``_extra``) generated from
  the corpus' own embedded grids in the en_qwerty.json schema, letters == key
  order, so ``train.py --layout`` accepts them unchanged.

Outputs (under ``--workdir``, default ``~/ctc-train``)
------------------------------------------------------
  ``<repo>/ctc/layouts/ru_jcuken_default.json``  (+ ``_extra``)  vendored geometry
  ``data/yandex_val10k.jsonl``      canonical rows from valid.jsonl + valid.ref
                                    (NOT used during training; final-eval only)
  ``cache_ru/train_yandex.npz``     systematic 1-in-k sample of default-grid
                                    train rows (``--train-limit``, default 1M)
  ``cache_ru/val.npz``              5,000 default-grid train rows disjoint from
                                    the training sample (greedy-selection val,
                                    so valid-10k stays untouched for reporting)
  ``cache_ru/prep_stats.json``      full drop accounting

train.py invocation this enables (committed code, untouched):
  ``--layout layouts/ru_jcuken_default.json --cache cache_ru
    --train-npz train_yandex.npz --beam-val-rows 0``  (greedy selection; the
  in-train beam validator is en-hardcoded via its vocab loader — the offline
  ``eval_cyrillic.py`` supplies the lexicon beam instead).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import featurize  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import T_OUT, adjacent_repeats  # noqa: E402

RU_LETTERS_31 = "абвгдежзийклмнопрстуфхцчшщыьэюя"
assert len(RU_LETTERS_31) == 31

#: Extra-grid alphabet order (33): adds ё after е and ъ after щ (alphabetical).
RU_LETTERS_33 = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
assert len(RU_LETTERS_33) == 33

LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"


def project_ru(word: str) -> Optional[str]:
    """Project a target/lexicon word onto the 31-letter default-grid alphabet.

    Returns ``None`` when the word has no spelling in that alphabet (latin
    letters, digits, …) — counted as untypeable, never mangled.
    """
    s = word.lower().replace("-", "").replace("'", "").replace("’", "")
    s = s.replace("ё", "е").replace("ъ", "ь")
    if not s:
        return None
    for c in s:
        if c not in RU_LETTERS_31:
            return None
    return s


def letter_block(grid: dict, letters: str) -> Tuple[float, float]:
    """(y_top, y_bottom) of the letter-key block in grid pixels."""
    tops, bots = [], []
    for k in grid["keys"]:
        lab = k.get("label")
        if lab is not None and lab in letters:
            hb = k["hitbox"]
            tops.append(hb["y"])
            bots.append(hb["y"] + hb["h"])
    return float(min(tops)), float(max(bots))


def grid_layout(grid: dict, letters: str) -> dict:
    """One embedded corpus grid -> en_qwerty.json-schema layout dict.

    Letter-area frame: x over the full grid width, y over the letter block, so
    the geometry lands in the canonical-corpus convention (see module docs).
    """
    w = float(grid["width"])
    y0, y1 = letter_block(grid, letters)
    bh = y1 - y0
    by_label = {}
    for k in grid["keys"]:
        lab = k.get("label")
        if lab is not None and lab in letters:
            hb = k["hitbox"]
            by_label[lab] = {
                "letter": lab,
                "cx": (hb["x"] + hb["w"] / 2.0) / w,
                "cy": (hb["y"] + hb["h"] / 2.0 - y0) / bh,
                "rx": hb["w"] / 2.0 / w,
                "ry": hb["h"] / 2.0 / bh,
            }
    missing = [c for c in letters if c not in by_label]
    if missing:
        raise SystemExit(f"grid {grid['grid_name']}: letters missing keys: {missing}")
    return {"name": f"ru_jcuken_{grid['grid_name']}",
            "letters": letters,
            "letter_block_px": [y0, y1],
            "keys": [by_label[c] for c in letters]}


def row_hash(word: str, xs: List[float], ys: List[float], ts: List[float]) -> bytes:
    """Exact-trace dedup key over the projected word + raw normalized floats."""
    h = hashlib.blake2b(digest_size=16)
    h.update(word.encode())
    h.update(np.asarray(xs, np.float64).tobytes())
    h.update(np.asarray(ys, np.float64).tobytes())
    h.update(np.asarray(ts, np.float64).tobytes())
    return h.digest()


def iter_corpus(path: Path, ref: Optional[Path] = None
                ) -> Iterable[Tuple[Optional[str], str, List[float], List[float],
                                    List[float], dict]]:
    """Yield ``(raw_word, grid_name, xs, ys, ts, grid)`` with normalized coords."""
    refs = ref.read_text(encoding="utf-8").split() if ref else None
    blocks: Dict[str, Tuple[float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            c = o["curve"]
            g = c["grid"]
            gname = g["grid_name"]
            if gname not in blocks:
                letters = RU_LETTERS_31 if gname == "default" else RU_LETTERS_33
                blocks[gname] = letter_block(g, letters)
            y0, y1 = blocks[gname]
            bh = y1 - y0
            w = float(g["width"])
            xs = [x / w for x in c["x"]]
            ys = [min(1.0, max(0.0, (y - y0) / bh)) for y in c["y"]]
            t0 = c["t"][0] if c["t"] else 0.0
            ts = [float(t - t0) for t in c["t"]]
            word = refs[i] if refs is not None else o.get("word")
            yield word, gname, xs, ys, ts, g


def keep_reason(word: Optional[str], xs: List[float], ts: List[float]
                ) -> Tuple[Optional[str], Optional[str]]:
    """-> (projected_word, None) if the row is trainable, else (None, reason)."""
    if word is None:
        return None, "no_target"
    p = project_ru(word)
    if p is None:
        return None, "unprojectable"
    if len(p) < 2:
        return None, "len1"
    if len(p) + adjacent_repeats(p) > T_OUT:
        return None, "ctc_infeasible"
    if len(xs) < 3:
        return None, "few_points"
    if ts[-1] <= 0:
        return None, "zero_duration"
    return p, None


def featurize_rows(rows: List[Tuple[str, List[float], List[float], List[float]]],
                   letters: str, out: Path, jobs: int,
                   provenance: Dict[str, object]) -> None:
    import multiprocessing as mp
    idx = {c: i for i, c in enumerate(letters)}
    feats: List[np.ndarray] = []
    if jobs > 1:
        pend = [(xs, ys, ts) for _, xs, ys, ts in rows]
        chunk = max(1, len(pend) // (jobs * 8))
        batches = [pend[i:i + chunk] for i in range(0, len(pend), chunk)]
        with mp.get_context("fork").Pool(jobs) as pool:
            for part in pool.imap(_feat_batch, batches):
                feats.extend(part)
    else:
        feats = [featurize(xs, ys, ts) for _, xs, ys, ts in rows]
    words = [w for w, _, _, _ in rows]
    tgt_flat = np.concatenate([np.array([idx[c] for c in w], np.int64)
                               for w in words])
    tgt_len = np.array([len(w) for w in words], np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=np.stack(feats).astype(np.float32),
                        targets=tgt_flat, target_lengths=tgt_len,
                        words=np.array(words),
                        provenance=np.array(json.dumps(provenance, sort_keys=True)))
    print(f"  -> {out}  rows {len(words)}")


def _feat_batch(batch):
    return [featurize(xs, ys, ts) for xs, ys, ts in batch]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--yandex-dir", type=Path,
                    default=Path.home() / "ctc-train" / "data" / "yandex_cup")
    ap.add_argument("--grid", default="default", choices=("default", "extra"))
    ap.add_argument("--train-limit", type=int, default=1_000_000)
    ap.add_argument("--val-rows", type=int, default=5_000,
                    help="training-time greedy-val rows carved from train "
                         "(valid-10k is never touched during training)")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--layouts-only", action="store_true")
    args = ap.parse_args()

    yd = args.yandex_dir
    cache = resolve(args.workdir, Path("cache_ru"))
    data_dir = resolve(args.workdir, Path("data"))

    # ── layouts, from the corpus' own grids (first occurrence of each) ─────────
    grids: Dict[str, dict] = {}
    for _, gname, _, _, _, g in iter_corpus(yd / "valid.jsonl"):
        if gname not in grids:
            grids[gname] = g
        if len(grids) == 2:
            break
    LAYOUT_DIR.mkdir(exist_ok=True)
    for gname, letters in (("default", RU_LETTERS_31), ("extra", RU_LETTERS_33)):
        lay = grid_layout(grids[gname], letters)
        p = LAYOUT_DIR / f"ru_jcuken_{gname}.json"
        p.write_text(json.dumps(lay, ensure_ascii=False, indent=1))
        print(f"layout {gname}: {len(lay['keys'])} keys -> {p}")
    if args.layouts_only:
        return 0

    letters = RU_LETTERS_31 if args.grid == "default" else RU_LETTERS_33
    st: Dict[str, int] = dict(train_in=0, off_grid=0, unprojectable=0, len1=0,
                              ctc_infeasible=0, few_points=0, zero_duration=0,
                              no_target=0, dup_val10k=0, dup_selfval=0,
                              dup_self=0, kept_pool=0)

    # ── valid-10k -> canonical jsonl (final-eval only) + its dedup hashes ─────
    t0 = time.time()
    val_rows: List[dict] = []
    val_hashes: Set[bytes] = set()
    v_st = dict(rows=0, off_grid=0, dropped=0)
    for word, gname, xs, ys, ts, _ in iter_corpus(yd / "valid.jsonl",
                                                  yd / "valid.ref"):
        v_st["rows"] += 1
        p, why = keep_reason(word, xs, ts)
        if p is None:
            v_st["dropped"] += 1
            continue
        # every valid row is a dedup source, but only same-grid rows are eval rows
        val_hashes.add(row_hash(p, xs, ys, ts))
        if gname != args.grid:
            v_st["off_grid"] += 1
            continue
        val_rows.append({"word": p, "grid": gname,
                         "points": [{"t": t, "x": x, "y": y}
                                    for t, x, y in zip(ts, xs, ys)]})
    val10k = data_dir / "yandex_val10k.jsonl"
    with open(val10k, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"valid-10k: {v_st} -> {len(val_rows)} eval rows ({args.grid}) "
          f"-> {val10k}  ({time.time() - t0:.0f}s)")

    # ── train pass: filter + systematic subsample + dedup ─────────────────────
    # Pass 1 estimates the keep-rate cheaply by sampling? Instead: single pass,
    # reservoir-free systematic thinning against a running keep counter with a
    # stride chosen from the known corpus size (6.0 M rows, ~92 % keep at
    # default grid) so the sample is spread over the whole collection period.
    est_kept = 6_000_000 * (0.92 if args.grid == "default" else 0.06)
    budget = args.train_limit + args.val_rows
    stride = max(1, int(est_kept / budget))
    print(f"train pass: stride {stride} (budget {budget})")
    picked: List[Tuple[str, List[float], List[float], List[float]]] = []
    seen: Set[bytes] = set()
    t0 = time.time()
    for word, gname, xs, ys, ts, _ in iter_corpus(yd / "train.jsonl"):
        st["train_in"] += 1
        if gname != args.grid:
            st["off_grid"] += 1
            continue
        p, why = keep_reason(word, xs, ts)
        if p is None:
            st[why] += 1
            continue
        st["kept_pool"] += 1
        if st["kept_pool"] % stride:
            continue
        h = row_hash(p, xs, ys, ts)
        if h in val_hashes:
            st["dup_val10k"] += 1
            continue
        if h in seen:
            st["dup_self"] += 1
            continue
        seen.add(h)
        picked.append((p, xs, ys, ts))
        if len(picked) >= budget:
            break
    print(f"train pass done: {json.dumps(st)}  picked {len(picked)} "
          f"({time.time() - t0:.0f}s)")

    # ── split: tail val_rows become the greedy-selection val ──────────────────
    sel_val = picked[-args.val_rows:]
    train = picked[:-args.val_rows]
    prov = dict(generator="prepare_yandex.py", grid=args.grid, stride=stride,
                letters=letters, stats=st, source=str(yd),
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    featurize_rows(train, letters, cache / "train_yandex.npz", args.jobs, prov)
    featurize_rows(sel_val, letters, cache / "val.npz", args.jobs,
                   dict(prov, role="greedy-selection val (train-derived)"))
    (cache / "prep_stats.json").write_text(json.dumps(
        dict(train=st, valid=v_st, n_train=len(train), n_sel_val=len(sel_val),
             n_val10k=len(val_rows), stride=stride), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
