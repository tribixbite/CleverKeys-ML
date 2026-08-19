#!/usr/bin/env python3
"""Decode evaluation for a Phase-O per-script model, through the exported ONNX.

Generalizes ``eval_cyrillic.py`` along the one axis Phase O needs and Phase I-B
did not: the probe may be a **synthesis holdout** (`cache_<code>/holdout.npz`,
features already in the cached ``[N,2,64]`` form) rather than a real corpus
jsonl.  Everything downstream is the same code path as every other decode in
this campaign — ``LexTrie``, ``futo_viterbi_beam``, ``greedy_ctc``, ``Tally`` —
with the a-z assumptions replaced at the two call sites they live in:

* the emission gather takes columns ``0..n_letters-1`` of the exported
  ``[32,65]`` head plus the blank at ``MAX_KEYS`` (emission column ``c`` is
  whatever key sits in slot ``c``);
* the word projection comes from :mod:`script_registry` and is applied
  identically to targets and lexicon.

**What a synthesis-holdout number is, stated once.**  The probe words are drawn
from the same lexicon the trie is built from, so **OOV is zero by construction**
and in-dict == all-rows.  That is *not* how a real corpus behaves (the real ru
probe carries 10.0 % OOV), so a synthesis in-dict figure is comparable to a real
in-dict figure but a synthesis all-rows figure is not comparable to anything.
More importantly the traces are generated, not swiped: the holdout measures
generalization to fresh samples of the generator, over a disjoint half of the
English donor pool and an independent word draw, and nothing more.  Russian is
the one script where the distance from that number to a real-swipe number is
measurable, and PHASE_O.md §2.1 measures it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_registry as SR  # noqa: E402
from eval_altlayout import OnnxEncoder  # noqa: E402
from futo_decoder_ceiling import futo_viterbi_beam  # noqa: E402
from futo_decoder_eval import (LexTrie, Tally, featurize, greedy_ctc,  # noqa: E402
                               len_stratum, load_layout, rank_of)
from model import MAX_KEYS  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402

HERE = Path(__file__).resolve().parent

#: E1 — the English-tuned preset every cross-script number starts from.
E1 = (1.05, 1.1, 0.2, 0.3734, 0.9882)
#: The app's CKDT-scale preset (`CtcScoringParams.tunedRuCkdt`): E1 with λ = 2.0.
#: λ is a property of the lexicon's frequency scale, not of the language
#: (PHASE_J §6.9), and every Phase-O lexicon is on the CKDT ``255 − rank``
#: scale — so this is the *prediction* each script's λ sweep tests.
CKDT = (1.05, 2.0, 0.2, 0.3734, 0.9882)


def build_trie(spec: SR.ScriptSpec) -> Tuple[LexTrie, Dict[str, int]]:
    """The script's lexicon as a ``LexTrie`` on the CKDT ``255 − rank`` scale."""
    lex, st = spec.load_lexicon()
    trie = LexTrie()
    for word, weight in lex:
        trie.insert(word, weight)
    st["distinct"] = trie.num_words
    return trie, st


def iter_npz(path: Path):
    """``(features [2,64], word)`` from a synthesis cache npz."""
    with np.load(path) as d:
        feats = d["features"]
        words = [str(w) for w in d["words"]]
    for f, w in zip(feats, words):
        yield np.asarray(f, np.float32), w


def iter_jsonl(path: Path):
    """``(features [2,64], word)`` from a canonical points jsonl (real corpus)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            pts = o["points"]
            yield featurize([p["x"] for p in pts], [p["y"] for p in pts],
                            [p["t"] for p in pts]), o["word"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code", required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--probe", default="holdout.npz",
                    help="cache npz name, or an absolute path to a .npz / .jsonl")
    ap.add_argument("--cache", default="", help="default cache_<code>")
    ap.add_argument("--preset", default="ckdt",
                    help="'e1', 'ckdt', or gamma,lambda,beta,gammaPrune,betaPrune")
    ap.add_argument("--lam", type=float, default=None,
                    help="override just lambda (the per-script sweep axis)")
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--rows", default="",
                    help="row slice 'lo:hi' over the probe, for tune/confirm "
                         "halves; applied before every other filter so the two "
                         "halves are disjoint by construction")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=1000)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None,
                    help="per-row JSONL {row, target, rank, greedy_hit} so two "
                         "models can be compared with a PAIRED test (McNemar) "
                         "instead of two independent point estimates")
    args = ap.parse_args()

    spec = SR.get(args.code)
    if args.preset == "e1":
        preset = list(E1)
    elif args.preset == "ckdt":
        preset = list(CKDT)
    else:
        preset = [float(v) for v in args.preset.split(",")]
    if args.lam is not None:
        preset[1] = args.lam
    gamma, lam_w, beta, gamma_prune, beta_prune = preset

    # ru's Phase-O synthesis lives in its own dir; cache_ru holds Phase I-B's
    # real-data caches (script_synth.py carries the same rule).
    default_cache = "cache_ru_phaseO" if spec.code == "ru" else f"cache_{spec.code}"
    cache = resolve(args.workdir, Path(args.cache or default_cache))
    probe = Path(args.probe)
    if not probe.is_absolute():
        probe = cache / probe

    letters, centers = load_layout(HERE / spec.layout_json)
    n_letters = len(letters)
    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:n_letters] = centers
    mask = np.zeros((MAX_KEYS,), bool)
    mask[:n_letters] = True

    trie, lex_st = build_trie(spec)
    print(f"[{spec.code}] lexicon {lex_st}", flush=True)

    enc = OnnxEncoder(args.onnx)
    g_tal, b_tal = Tally(), Tally()
    strat = {"<=3": Tally(), "4+": Tally()}
    oov = unproj = n_seen = 0
    row_lo, row_hi = 0, 1 << 62
    if args.rows:
        row_lo, row_hi = (int(v) for v in args.rows.split(":"))
    src = iter_jsonl(probe) if probe.suffix == ".jsonl" else iter_npz(probe)
    dump = None
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        dump = open(args.dump, "w", encoding="utf-8")
    t0 = time.time()
    for row_i, (feats, raw_word) in enumerate(src):
        if row_i < row_lo:
            continue
        if row_i >= row_hi:
            break
        if args.limit and n_seen >= args.limit:
            break
        n_seen += 1
        target = spec.project(raw_word)
        if target is None:
            unproj += 1
            continue
        if not trie.contains(target):
            oov += 1
            continue
        full = enc.forward(feats, keys, mask)                # [T', 65]
        lp = np.empty((full.shape[0], n_letters + 1), np.float32)
        lp[:, :n_letters] = full[:, :n_letters]
        lp[:, n_letters] = full[:, MAX_KEYS]
        greedy = greedy_ctc(lp, letters, n_letters)
        beam = futo_viterbi_beam(lp, letters, n_letters, trie, args.beam_width,
                                 args.top_k, gamma, lam_w, beta, gamma_prune,
                                 beta_prune)
        words = [w for w, _ in beam]
        g_tal.add(0 if greedy == target else -1)
        r = rank_of(target, words)
        b_tal.add(r)
        strat[len_stratum(target)].add(r)
        if dump is not None:
            dump.write(json.dumps({"row": row_i, "target": target, "rank": r,
                                   "greedy_hit": int(greedy == target)},
                                  ensure_ascii=False) + "\n")
        if args.progress and b_tal.n % args.progress == 0:
            print(f"  {b_tal.n} decoded  t1 {b_tal.t1 / b_tal.n * 100:.2f} "
                  f"greedy {g_tal.t1 / g_tal.n * 100:.2f} "
                  f"({b_tal.n / (time.time() - t0):.1f} tr/s)", flush=True)

    res = {
        "script": spec.code, "onnx": str(args.onnx), "probe": str(probe),
        "layout": spec.layout_json, "lexicon_tier": spec.lexicon.tier,
        "rows_slice": args.rows or "all", "preset": preset,
        "rows": n_seen, "decoded": b_tal.n, "unprojectable": unproj, "oov": oov,
        "greedy_t1": round(g_tal.t1 / max(g_tal.n, 1) * 100, 2),
        "indict_t1": round(b_tal.t1 / max(b_tal.n, 1) * 100, 2),
        "indict_t3": round(b_tal.t3 / max(b_tal.n, 1) * 100, 2),
        "indict_t5": round(b_tal.t5 / max(b_tal.n, 1) * 100, 2),
        "allrows_t1": round(b_tal.t1 / max(n_seen, 1) * 100, 2),
        "le3_n": strat["<=3"].n,
        "le3_t1": round(strat["<=3"].t1 / max(strat["<=3"].n, 1) * 100, 2),
        "ge4_n": strat["4+"].n,
        "ge4_t1": round(strat["4+"].t1 / max(strat["4+"].n, 1) * 100, 2),
        "seconds": round(time.time() - t0, 1),
    }
    if dump is not None:
        dump.close()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
