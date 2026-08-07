#!/usr/bin/env python3
"""Beam-eval one or more Phase-A arms on val-9918, sliced by source and by val_clean.

Uses the cached-emission fast path from ``sweep_scoring.py`` rather than
``eval_beam.py``'s per-row loop: emissions are computed once per arm through the
exported ONNX graph, then the vendored beam runs once over all 9,918 rows with
``gamma=beta=lambda=0`` and ``top_k=beam_width`` so every terminal-beam word comes
back with its raw CTC path score. Re-scoring those candidates analytically at the
published ``enc`` preset reproduces ``eval_beam.py`` exactly (verified on r2:
81.57 / 89.84 / 91.37 both ways) at ~7 s instead of ~160 s per arm.

Reported per arm:
  * FULL val top-1/3/5;
  * per-source top-1 (``cache/holdout_source_tags.json``: futo vs hws) — the two
    halves sit at a known ~0.064 systematic Y offset, so the aggregate hides them;
  * top-1 on the arm's own ``val_clean`` mask (rows with no contributor overlap
    with that arm's training pool) and on any other mask named by ``--also-masks``,
    which is what makes a cross-arm comparison fair.

**test-2400 is never decoded here.** The only accepted ``--test`` is the val split.

Usage:
  python eval_arms.py --arms phaseA-T0,phaseA-T1,phaseA-T2,phaseA-T2b
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_ceiling import (ENC_BETA, ENC_BETA_PRUNE, ENC_GAMMA,  # noqa: E402
                                  ENC_GAMMA_PRUNE, ENC_LAMBDA)
from futo_decoder_eval import load_combined_vocab, load_test  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
from sweep_scoring import (TraceCandidates, _init_worker, build_emissions,  # noqa: E402
                           collect, score_grid)

#: The published encoder-only preset every committed number is quoted at.
PRESET = (ENC_GAMMA, ENC_LAMBDA, ENC_BETA, ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)


def subset(traces: Sequence[TraceCandidates], keep: np.ndarray) -> List[TraceCandidates]:
    """Rows of *traces* selected by a boolean mask."""
    return [t for t, k in zip(traces, keep) if k]


def top(traces: Sequence[TraceCandidates]) -> Tuple[float, float, float]:
    """-> ``(t1, t3, t5)`` at :data:`PRESET`; ``(nan,)*3`` for an empty subset."""
    if not traces:
        return float("nan"), float("nan"), float("nan")
    g, lam, b = PRESET[0], PRESET[1], PRESET[2]
    return score_grid(list(traces), g, b, lam)


def load_masks(path: Path, n: int) -> Dict[str, np.ndarray]:
    """Read ``val_clean_masks.json`` -> ``{name: bool[n]}``, plus derived combos."""
    raw = json.loads(Path(path).read_text())
    out: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "clean" in v:
            m = np.array(v["clean"], bool)
            if len(m) != n:
                raise SystemExit(f"{path}: mask '{k}' has {len(m)} rows, val has {n}")
            out[k] = m
    if "T2" in out and "T2b" in out:
        # The unconfounded contrast: both arms are FUTO-only and session-excluded,
        # so their clean subsets are directly comparable to each other.
        out["T2_T2b_SHARED"] = out["T2"] & out["T2b"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--arms", required=True,
                    help="comma-separated run names under ckpt/ (e.g. phaseA-T0)")
    ap.add_argument("--onnx-name", default="ctc_swipe_encoder.onnx", dest="onnx_name")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    ap.add_argument("--test", default="data/val_hwsfuto.jsonl",
                    help="val split only; test-2400 is refused")
    ap.add_argument("--tags", default="cache/holdout_source_tags.json")
    ap.add_argument("--masks", default="cache/val_clean_masks.json")
    ap.add_argument("--also-masks", default="T2_T2b_SHARED", dest="also_masks",
                    help="comma-separated extra mask names to score every arm on")
    ap.add_argument("--own-mask", default="", dest="own_mask",
                    help="override the arm->mask name mapping (default: strip "
                         "the 'phaseA-' run-name prefix)")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="cache/phase_a_results.json")
    args = ap.parse_args()

    if "test" in Path(args.test).name:
        raise SystemExit(f"refusing to decode {args.test}: Phase A is val-only")

    rows = load_test(resolve(args.workdir, args.test))
    targets = [w.lower() for w, _, _, _ in rows]
    n = len(rows)
    tags = json.loads(resolve(args.workdir, args.tags).read_text())["val"]
    if len(tags) != n:
        raise SystemExit(f"source tags cover {len(tags)} rows, val has {n}")
    is_futo = np.array([t == "futo" for t in tags], bool)
    masks = load_masks(resolve(args.workdir, args.masks), n)
    extra = [m for m in args.also_masks.split(",") if m and m in masks]
    print(f"val rows {n}  futo {int(is_futo.sum())}  hws {int((~is_futo).sum())}")
    print(f"masks: " + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()))

    trie = load_combined_vocab(resolve(args.workdir, args.vocab))
    print(f"trie: {trie.num_words} words\n")

    results: Dict[str, object] = {"preset": list(PRESET), "n_val": n}
    for arm in args.arms.split(","):
        run = resolve(args.workdir, Path("ckpt") / arm)
        onnx = run / args.onnx_name
        if not onnx.exists():
            print(f"{arm}: MISSING {onnx} — run export_onnx.py first; skipped\n")
            continue
        t0 = time.time()
        emissions, letters = build_emissions(onnx, args.layout, rows,
                                             run / "eval_emissions.npz", False)
        ctx = mp.get_context("fork")
        pool = ctx.Pool(args.jobs, initializer=_init_worker,
                        initargs=(trie, letters, emissions, args.beam_width))
        try:
            traces = collect(pool, trie, targets, 0, n, PRESET[3], PRESET[4], args.jobs)
        finally:
            pool.close()
            pool.join()

        rec: Dict[str, object] = {}
        t1, t3, t5 = top(traces)
        rec["full"] = {"n": n, "t1": t1, "t3": t3, "t5": t5}
        print(f"=== {arm} ===")
        print(f"  {'FULL val':<24} n={n:<5} t1 {t1:5.2f}  t3 {t3:5.2f}  t5 {t5:5.2f}")
        for name, sel in (("futo", is_futo), ("hws", ~is_futo)):
            s = top(subset(traces, sel))
            rec[name] = {"n": int(sel.sum()), "t1": s[0], "t3": s[1], "t5": s[2]}
            print(f"  {'source ' + name:<24} n={int(sel.sum()):<5} t1 {s[0]:5.2f}"
                  f"  t3 {s[1]:5.2f}  t5 {s[2]:5.2f}")
        own = args.own_mask or arm.split("-", 1)[-1]
        for name in ([own] if own in masks else []) + extra:
            sel = masks[name]
            s = top(subset(traces, sel))
            label = f"clean[{name}]" + ("*" if name == own else "")
            rec[f"clean_{name}"] = {"n": int(sel.sum()), "t1": s[0], "t3": s[1], "t5": s[2]}
            print(f"  {label:<24} n={int(sel.sum()):<5} t1 {s[0]:5.2f}"
                  f"  t3 {s[1]:5.2f}  t5 {s[2]:5.2f}")
        if own not in masks:
            print(f"  (no own mask for '{own}' in {args.masks})")
        ck = run / "best.pt"
        if ck.exists():
            import torch
            meta = torch.load(ck, map_location="cpu", weights_only=True)
            rec["best_val_greedy"] = float(meta["best"])
            rec["best_epoch"] = int(meta["best_epoch"])
            rec["step"] = int(meta["step"])
            print(f"  best val_greedy {meta['best'] * 100:.2f}% @ epoch "
                  f"{meta['best_epoch']} (step {meta['step']})")
        print(f"  [{time.time() - t0:.1f}s]\n")
        results[arm] = rec

    p = resolve(args.workdir, args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
