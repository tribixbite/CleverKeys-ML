#!/usr/bin/env python3
"""Grid-sweep the trie-beam scoring params over cached emissions (guide §7 free win).

The published `encoder:honorable_sturgeon` preset (gamma 0.4056, lambda 0.0176,
beta 0.9866, gammaPrune 0.4234, betaPrune 1.0382) was tuned for FUTO's emissions.
Ours are a different model, so re-tuning on **val** is a legitimate free win. Test
is never touched here.

Two speedups make an exhaustive grid cheap:

1. **Emissions are computed once.** The ONNX encoder runs over the row range a
   single time and the sliced ``[32,27]`` contract view is cached to npz. Every
   subsequent sweep reads the cache — no NN re-runs.
2. **The beam runs once per prune setting, not once per grid point.** In
   ``futo_viterbi_beam`` the per-frame pruning key depends only on
   ``(gammaPrune, betaPrune)``; ``(gamma, beta, lambda)`` enter solely through the
   final scoring applied to the surviving beam,
   ``final = raw / max(L,1)**gamma + beta*L + lambda*log_freq``.
   So for a fixed prune setting we call the **unmodified vendored beam** once with
   ``gamma=beta=lambda=0`` and ``top_k=beam_width``, which makes it return every
   complete word in the terminal beam with its raw CTC path score, then re-score
   those candidates analytically for all grid points. Dedup-by-word is unaffected:
   for a fixed word, ``final`` is a strictly increasing function of ``raw``, so
   keeping the max-raw hypothesis is the same as keeping the max-final one.

Usage:
  python sweep_scoring.py --onnx ckpt/r2/ctc_swipe_encoder.onnx
  python sweep_scoring.py --eval-only --preset 0.4056,0.0176,0.9866,0.4234,1.0382
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_ceiling import (ENC_BETA, ENC_BETA_PRUNE, ENC_GAMMA,  # noqa: E402
                                  ENC_GAMMA_PRUNE, ENC_LAMBDA,
                                  futo_viterbi_beam, slice_emissions)
from futo_decoder_eval import (featurize, len_stratum,  # noqa: E402
                               load_layout, load_test)
import lexicon  # noqa: E402
from model import MAX_KEYS  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
import seal  # noqa: E402
from prepare_data import normalize_word  # noqa: E402

#: Coarse grid, centred on the published encoder-only preset.
GRID_GAMMA = (0.30, 0.3556, 0.4056, 0.4556, 0.51)
GRID_BETA = (0.89, 0.94, 0.9866, 1.03, 1.08)
GRID_LAMBDA = (0.009, 0.0176, 0.026)
#: Refinement pass: prune params at +-0.05 around the preset.
REFINE_STEP = 0.05

# Worker globals, populated in the parent and inherited through fork (no pickling
# of the 147 k-word trie or the emission cache).
_TRIE = None
_LETTERS: List[str] = []
_EMISSIONS: Optional[np.ndarray] = None
_BEAM_WIDTH = 100


def _init_worker(trie, letters: List[str], emissions: np.ndarray, beam_width: int) -> None:
    global _TRIE, _LETTERS, _EMISSIONS, _BEAM_WIDTH
    _TRIE, _LETTERS, _EMISSIONS, _BEAM_WIDTH = trie, letters, emissions, beam_width


def _raw_candidates(bounds: Tuple[int, int, float, float]) -> List[List[Tuple[str, float]]]:
    """Run the vendored beam over a row slice, returning raw-scored candidates.

    ``gamma=beta=lambda=0`` reduces the final score to the raw CTC path score, and
    ``top_k=beam_width`` disables truncation, so the result is the complete set of
    terminal-beam words with their raw scores.
    """
    lo, hi, gp, bp = bounds
    n_letters = len(_LETTERS)
    out = []
    for i in range(lo, hi):
        out.append(futo_viterbi_beam(_EMISSIONS[i], _LETTERS, n_letters, _TRIE,
                                     _BEAM_WIDTH, _BEAM_WIDTH,
                                     0.0, 0.0, 0.0, gp, bp))
    return out


def word_log_freq(trie, word: str) -> float:
    """AOSP-scale ``log_freq`` stored on the word's terminal trie node."""
    node = trie.root
    for ch in word:
        node = node.children.get(ch)
        if node is None:
            return 0.0
    return node.log_freq


class TraceCandidates:
    """Per-trace re-scoring arrays: raw path score, length and log-frequency."""

    __slots__ = ("raw", "length", "logfreq", "target_i", "stratum")

    def __init__(self, cands: Sequence[Tuple[str, float]], target: str, trie) -> None:
        self.raw = np.array([s for _, s in cands], np.float64)
        self.length = np.array([max(len(w), 1) for w, _ in cands], np.float64)
        self.logfreq = np.array([word_log_freq(trie, w) for w, _ in cands], np.float64)
        self.target_i = next((j for j, (w, _) in enumerate(cands) if w == target), -1)
        self.stratum = len_stratum(target)

    def rank(self, gamma: float, beta: float, lam: float) -> int:
        """Rank of the target under a scoring setting, or -1 if it never surfaced."""
        if self.target_i < 0:
            return -1
        final = self.raw / self.length ** gamma + beta * self.length + lam * self.logfreq
        return int((final > final[self.target_i]).sum())


def score_grid(traces: List[TraceCandidates], gamma: float, beta: float,
               lam: float) -> Tuple[float, float, float]:
    """-> ``(top1, top3, top5)`` percentages over *traces*."""
    n = len(traces)
    t1 = t3 = t5 = 0
    for tc in traces:
        r = tc.rank(gamma, beta, lam)
        if r < 0:
            continue
        t1 += r < 1
        t3 += r < 3
        t5 += r < 5
    return t1 / n * 100, t3 / n * 100, t5 / n * 100


def score_all(traces: List[TraceCandidates], gamma: float, beta: float,
              lam: float) -> Tuple[float, float, float, float, float]:
    """-> ``(t1, t3, t5, le3_t1, ge4_t1)`` percentages in ONE ranking pass.

    Phase J: the terminal condition is conjunctive over five val metrics, two of
    which are length strata, so a sweep that optimises aggregate t1 alone cannot
    express it. ``--objective minmargin`` needs all five per grid point.
    """
    n = len(traces)
    t1 = t3 = t5 = 0
    acc = {"<=3": [0, 0], "4+": [0, 0]}
    for tc in traces:
        r = tc.rank(gamma, beta, lam)
        a = acc[tc.stratum]
        a[0] += 1
        if r < 0:
            continue
        hit = r < 1
        t1 += hit
        t3 += r < 3
        t5 += r < 5
        a[1] += hit
    return (t1 / n * 100, t3 / n * 100, t5 / n * 100,
            acc["<=3"][1] / max(acc["<=3"][0], 1) * 100,
            acc["4+"][1] / max(acc["4+"][0], 1) * 100)


def objective_key(metrics: Tuple[float, ...], objective: str,
                  bars: Sequence[float]) -> Tuple[float, ...]:
    """Sort key (higher is better) for a grid point's metrics.

    ``t1``        — the historical key ``(t1, t3)``; every pre-Phase-J number
                    was tuned under it and it stays the default.
    ``minmargin`` — maximise the WORST margin over the five bars
                    ``(t1, t3, t5, le3, ge4)``, tie-broken by the mean margin.
                    This is a direct encoding of "clear every bar", so it will
                    trade aggregate t1 away to lift a lagging stratum.
    """
    if objective == "t1":
        return (metrics[0], metrics[1])
    margins = [m - b for m, b in zip(metrics, bars)]
    return (min(margins), sum(margins) / len(margins))


def strata(traces: List[TraceCandidates], gamma: float, beta: float,
           lam: float) -> Dict[str, Tuple[int, float]]:
    """Top-1 by length stratum."""
    acc: Dict[str, List[int]] = {"<=3": [0, 0], "4+": [0, 0]}
    for tc in traces:
        r = tc.rank(gamma, beta, lam)
        a = acc[tc.stratum]
        a[0] += 1
        a[1] += 0 <= r < 1
    return {k: (v[0], v[1] / max(v[0], 1) * 100) for k, v in acc.items()}


def build_emissions(onnx_path: Path, layout: Path, rows, cache: Path,
                    force: bool) -> Tuple[np.ndarray, List[str]]:
    """Compute (or load) the sliced ``[N,32,27]`` emission cache for *rows*."""
    letters, key_centers = load_layout(layout)
    n_letters = len(letters)
    if cache.exists() and not force:
        d = np.load(cache)
        if len(d["emissions"]) >= len(rows):
            print(f"[cache] {cache} ({len(d['emissions'])} rows)")
            return d["emissions"][: len(rows)], letters
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:n_letters] = key_centers
    mask = np.zeros((MAX_KEYS,), bool)
    mask[:n_letters] = True
    out = np.empty((len(rows), 32, n_letters + 1), np.float32)
    t0 = time.time()
    for i, (_, xs, ys, ts) in enumerate(rows):
        feats = featurize(xs, ys, ts)
        full = sess.run(["log_emissions"],
                        {"features": feats[None], "layout_keys": keys[None],
                         "layout_mask": mask[None]})[0][0]
        out[i] = slice_emissions(full, n_letters, MAX_KEYS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emissions=out)
    print(f"[cache] built {len(rows)} rows in {time.time() - t0:.1f}s -> {cache}")
    return out, letters


def collect(pool, trie, targets: List[str], lo: int, hi: int, gp: float, bp: float,
            jobs: int) -> List[TraceCandidates]:
    """Run the beam once over ``[lo,hi)`` at one prune setting -> re-scoring arrays."""
    step = max(1, (hi - lo + jobs - 1) // jobs)
    chunks = [(a, min(a + step, hi), gp, bp) for a in range(lo, hi, step)]
    traces: List[TraceCandidates] = []
    for chunk, res in zip(chunks, pool.map(_raw_candidates, chunks)):
        for off, cands in enumerate(res):
            traces.append(TraceCandidates(cands, targets[chunk[0] + off], trie))
    return traces


def report(tag: str, traces: List[TraceCandidates], p: Sequence[float]) -> Tuple[float, float, float]:
    """Print and return top-1/3/5 for one preset over one row set."""
    g, lam, b = p[0], p[1], p[2]
    t1, t3, t5 = score_grid(traces, g, b, lam)
    st = strata(traces, g, b, lam)
    print(f"  {tag:<34} n={len(traces):<5} t1 {t1:5.2f}  t3 {t3:5.2f}  t5 {t5:5.2f}"
          f"   (<=3 {st['<=3'][1]:5.2f} / 4+ {st['4+'][1]:5.2f})")
    return t1, t3, t5


def parse_rows(spec: str) -> Tuple[int, int]:
    lo, hi = spec.split(":")
    return int(lo), int(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--onnx", default="ckpt/r2/ctc_swipe_encoder.onnx")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    lexicon.add_argument(ap)
    ap.add_argument("--test", default="data/val_hwsfuto.jsonl")
    ap.add_argument("--cache", default="ckpt/r2/sweep_emissions.npz")
    ap.add_argument("--rebuild-cache", action="store_true", dest="rebuild_cache")
    ap.add_argument("--sweep-rows", default="0:2000", dest="sweep_rows")
    ap.add_argument("--holdout-rows", default="2000:4000", dest="holdout_rows")
    ap.add_argument("--full-rows", default="0:9918", dest="full_rows")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--eval-only", action="store_true", dest="eval_only",
                    help="skip the grid; just evaluate --preset (+ baseline)")
    ap.add_argument("--preset", default="",
                    help="gamma,lambda,beta,gammaPrune,betaPrune to evaluate")
    ap.add_argument("--out", default="ckpt/r2/sweep_scoring.json")
    ap.add_argument("--grid-gamma", default="", dest="grid_gamma",
                    help="comma-separated override for the coarse gamma grid")
    ap.add_argument("--grid-beta", default="", dest="grid_beta")
    ap.add_argument("--grid-lambda", default="", dest="grid_lambda")
    ap.add_argument("--grid-gamma-prune", default="", dest="grid_gamma_prune",
                    help="explicit pass-2 gammaPrune values (default: the preset "
                         "+-REFINE_STEP). The prune params are the beam's length "
                         "normalisation for SURVIVAL, so a mis-tuned gammaPrune "
                         "drops candidates before scoring ever sees them and caps "
                         "top-3/top-5 no matter what the final score does")
    ap.add_argument("--grid-beta-prune", default="", dest="grid_beta_prune")
    ap.add_argument("--skip-refine", action="store_true", dest="skip_refine",
                    help="coarse pass only (no prune-param refinement)")
    ap.add_argument("--blank-grid", default="", dest="blank_grid",
                    help="Phase J: comma-separated constant offsets added to the "
                         "blank column of the sliced log-emissions BEFORE the "
                         "beam runs (RESEARCH_SCAN.md §1.1 #5 — the zero-cost "
                         "test of the CTC-peakiness hypothesis). Unlike "
                         "(gamma,beta,lambda), a blank shift changes the beam's "
                         "path scores and pruning, so each offset runs its own "
                         "beam pass over --sweep-rows and --holdout-rows at the "
                         "--preset prune setting; scoring uses --preset. 0 must "
                         "be in the grid — it is the exact incumbent")
    ap.add_argument("--objective", default="t1", choices=("t1", "minmargin"),
                    help="what the grid maximises. 't1' = the historical "
                         "(t1,t3) key every pre-Phase-J preset was tuned under. "
                         "'minmargin' = the worst margin over --bars across "
                         "(t1,t3,t5,le3,ge4), i.e. a direct encoding of the "
                         "conjunctive 'clear every bar' condition")
    ap.add_argument("--bars", default="88.30,92.60,93.26,91.27,86.77",
                    help="t1,t3,t5,le3,ge4 bars for --objective minmargin "
                         "(default: the resbn192i 3-seed val bars)")
    ap.add_argument("--lambda-only", action="store_true", dest="lambda_only",
                    help="sweep ONLY lambda over --grid-lambda, holding gamma/beta/"
                         "gammaPrune/betaPrune at --preset. lambda is the "
                         "frequency-scale knob, so this is the correct (and only "
                         "legitimate) re-fit when the lexicon's log_freq scale "
                         "changes but the emissions do not — see lexicon.py / O3. "
                         "Selection is on --sweep-rows; --holdout-rows and "
                         "--full-rows are confirmation only.")
    seal.add_argument(ap)
    args = ap.parse_args()
    bars = [float(v) for v in args.bars.split(",")]
    if len(bars) != 5:
        raise SystemExit("--bars needs exactly 5 values: t1,t3,t5,le3,ge4")

    global GRID_GAMMA, GRID_BETA, GRID_LAMBDA
    if args.grid_gamma:
        GRID_GAMMA = tuple(float(v) for v in args.grid_gamma.split(","))
    if args.grid_beta:
        GRID_BETA = tuple(float(v) for v in args.grid_beta.split(","))
    if args.grid_lambda:
        GRID_LAMBDA = tuple(float(v) for v in args.grid_lambda.split(","))

    base = (ENC_GAMMA, ENC_LAMBDA, ENC_BETA, ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)
    sw_lo, sw_hi = parse_rows(args.sweep_rows)
    ho_lo, ho_hi = parse_rows(args.holdout_rows)
    fu_lo, fu_hi = parse_rows(args.full_rows)
    need = max(sw_hi, ho_hi, fu_hi)

    rows = load_test(resolve(args.workdir, args.test))[:need]
    seal.check(rows, args.unseal_test, what="sweep_scoring.py")
    # Phase N §15.1: a-z-normalize targets (not merely lowercase). Canonical
    # splits are pre-normalized so this is a no-op for them; on raw corpora
    # (FUTO official dev/test) the old form auto-missed every punctuated word
    # for every preset — the PHASE_N.md §11.2 erratum. Trie candidates are
    # normalized, so the target must be scored in the same alphabet.
    targets = [normalize_word(w) for w, _, _, _ in rows]
    print(f"rows: {len(rows)}  sweep {sw_lo}:{sw_hi}  holdout {ho_lo}:{ho_hi}  "
          f"full {fu_lo}:{fu_hi}")

    emissions, letters = build_emissions(resolve(args.workdir, args.onnx), args.layout,
                                         rows, resolve(args.workdir, args.cache),
                                         args.rebuild_cache)
    trie = lexicon.load_vocab(resolve(args.workdir, args.vocab), args.vocab_kind)
    print(f"trie: {trie.num_words} words  (kind '{args.vocab_kind}', {args.vocab})")

    ctx = mp.get_context("fork")

    if args.blank_grid:
        # Each offset needs its own worker pool: the shift enters the beam DP
        # itself (transition scores AND per-frame pruning), so the analytic
        # re-scoring shortcut does not apply along this axis.
        if not args.preset:
            raise SystemExit("--blank-grid needs --preset "
                             "gamma,lambda,beta,gammaPrune,betaPrune")
        p = tuple(float(v) for v in args.preset.split(","))
        if len(p) != 5:
            raise SystemExit("--preset needs 5 floats")
        offsets = [float(v) for v in args.blank_grid.split(",")]
        blank_col = len(letters)            # sliced view: blank at column K
        results = {"preset": list(p), "blank_grid": []}
        print(f"[blank-grid] offsets {offsets} at preset {p}")
        for off in offsets:
            em = emissions if off == 0.0 else emissions.copy()
            if off != 0.0:
                em[..., blank_col] += np.float32(off)
            pool = ctx.Pool(args.jobs, initializer=_init_worker,
                            initargs=(trie, letters, em, args.beam_width))
            try:
                rec = {"offset": off}
                for tag, lo, hi in (("sweep half", sw_lo, sw_hi),
                                    ("holdout half", ho_lo, ho_hi)):
                    tr = collect(pool, trie, targets, lo, hi, p[3], p[4], args.jobs)
                    rec[tag] = report(f"blank {off:+.2f} / {tag}", tr, p)
                    st = strata(tr, p[0], p[2], p[1])
                    rec[tag + " strata"] = {k: v[1] for k, v in st.items()}
                results["blank_grid"].append(rec)
            finally:
                pool.close()
                pool.join()
        out_path = resolve(args.workdir, args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {out_path}")
        return 0

    pool = ctx.Pool(args.jobs, initializer=_init_worker,
                    initargs=(trie, letters, emissions, args.beam_width))
    results: Dict[str, object] = {"baseline_preset": list(base)}

    try:
        if args.eval_only:
            presets = {"baseline (enc preset)": base}
            if args.preset:
                presets["tuned"] = tuple(float(v) for v in args.preset.split(","))
            for name, p in presets.items():
                for tag, lo, hi in (("sweep half", sw_lo, sw_hi),
                                    ("holdout half", ho_lo, ho_hi),
                                    ("FULL val", fu_lo, fu_hi)):
                    tr = collect(pool, trie, targets, lo, hi, p[3], p[4], args.jobs)
                    report(f"{name} / {tag}", tr, p)
            return 0

        if args.lambda_only:
            if not args.preset:
                raise SystemExit("--lambda-only needs --preset "
                                 "gamma,lambda,beta,gammaPrune,betaPrune")
            inc = tuple(float(v) for v in args.preset.split(","))
            if len(inc) != 5:
                raise SystemExit("--preset needs 5 floats")
            g, b, gp, bp = inc[0], inc[2], inc[3], inc[4]
            print(f"[lambda-only] gamma={g} beta={b} gammaPrune={gp} betaPrune={bp} "
                  f"fixed; sweeping lambda over {list(GRID_LAMBDA)}")
            # The beam depends on (gammaPrune, betaPrune) only, so ONE beam pass over
            # the selection rows serves every lambda in the grid (module docstring).
            t0 = time.time()
            sw_tr = collect(pool, trie, targets, sw_lo, sw_hi, gp, bp, args.jobs)
            print(f"[lambda-only] beam over {len(sw_tr)} selection rows in "
                  f"{time.time() - t0:.1f}s")
            scored = []
            for lam in GRID_LAMBDA:
                t1, t3, t5 = score_grid(sw_tr, g, b, lam)
                st = strata(sw_tr, g, b, lam)
                mark = "  <- incumbent" if lam == inc[1] else ""
                print(f"   lambda {lam:8.4f}   t1 {t1:5.2f}  t3 {t3:5.2f}  t5 {t5:5.2f}"
                      f"   (<=3 {st['<=3'][1]:5.2f} / 4+ {st['4+'][1]:5.2f}){mark}")
                scored.append(((t1, t3), lam, (t1, t3, t5)))
            scored.sort(key=lambda r: r[0], reverse=True)
            best_lam = scored[0][1]
            tuned = (g, best_lam, b, gp, bp)
            print(f"\n[lambda-only] selection winner lambda={best_lam} "
                  f"(t1 {scored[0][2][0]:.2f} on rows {sw_lo}:{sw_hi})")
            results["lambda_grid"] = [{"lambda": lam, "t1": m[0], "t3": m[1],
                                       "t5": m[2]} for _, lam, m in scored]
            results["incumbent_preset"] = list(inc)
            results["tuned_preset"] = list(tuned)
            conf: Dict[str, object] = {}
            for tag, lo, hi in (("selection rows", sw_lo, sw_hi),
                                ("CONFIRM holdout", ho_lo, ho_hi),
                                ("FULL val", fu_lo, fu_hi)):
                tr = (sw_tr if (lo, hi) == (sw_lo, sw_hi)
                      else collect(pool, trie, targets, lo, hi, gp, bp, args.jobs))
                im = report(f"incumbent lambda={inc[1]} / {tag}", tr, inc)
                tm = report(f"tuned     lambda={best_lam} / {tag}", tr, tuned)
                st = strata(tr, g, b, best_lam)
                print(f"  {'delta':<34} {'':>11} t1 {tm[0] - im[0]:+5.2f}  "
                      f"t3 {tm[1] - im[1]:+5.2f}  t5 {tm[2] - im[2]:+5.2f}")
                conf[tag] = {"incumbent": im, "tuned": tm,
                             "tuned_strata": {k: v for k, v in st.items()}}
            results["confirmation"] = conf
            out_path = resolve(args.workdir, args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=1))
            print(f"\nwrote {out_path}")
            return 0

        # ── pass 1: coarse (gamma, beta, lambda) at the preset prune setting ────
        t0 = time.time()
        sweep_tr = collect(pool, trie, targets, sw_lo, sw_hi, base[3], base[4], args.jobs)
        print(f"[pass1] beam over {len(sweep_tr)} rows in {time.time() - t0:.1f}s "
              f"({args.jobs} procs); grid {len(GRID_GAMMA)}x{len(GRID_BETA)}x"
              f"{len(GRID_LAMBDA)} = {len(GRID_GAMMA) * len(GRID_BETA) * len(GRID_LAMBDA)}")
        scored = []
        for g, b, lam in itertools.product(GRID_GAMMA, GRID_BETA, GRID_LAMBDA):
            m = score_all(sweep_tr, g, b, lam)
            scored.append((objective_key(m, args.objective, bars),
                           (g, lam, b, base[3], base[4]), m))
        scored.sort(key=lambda r: r[0], reverse=True)
        print("[pass1] top 5:")
        for key, p, m in scored[:5]:
            print(f"   gamma {p[0]:.4f} lambda {p[1]:.4f} beta {p[2]:.4f}"
                  f"   t1 {m[0]:5.2f}  t3 {m[1]:5.2f}  t5 {m[2]:5.2f}"
                  f"  <=3 {m[3]:5.2f}  4+ {m[4]:5.2f}"
                  f"  minmargin {min(v - b for v, b in zip(m, bars)):+5.2f}")
        w1 = scored[0][1]
        results["pass1_winner"] = list(w1)
        results["pass1_grid"] = [{"preset": list(p), "t1": m[0], "t3": m[1],
                                  "t5": m[2], "le3": m[3], "ge4": m[4]}
                                 for _, p, m in scored]

        # ── pass 2: prune params +-REFINE_STEP, local (gamma, beta, lambda) grid ──
        gps = ([float(v) for v in args.grid_gamma_prune.split(",")]
               if args.grid_gamma_prune
               else [round(base[3] + d * REFINE_STEP, 4) for d in (-1, 0, 1)])
        bps = ([float(v) for v in args.grid_beta_prune.split(",")]
               if args.grid_beta_prune
               else [round(base[4] + d * REFINE_STEP, 4) for d in (-1, 0, 1)])
        loc_g = sorted({round(w1[0] + d, 4) for d in (-0.05, -0.025, 0.0, 0.025, 0.05)})
        loc_b = sorted({round(w1[2] + d, 4) for d in (-0.05, -0.025, 0.0, 0.025, 0.05)})
        best = (scored[0][0], w1, scored[0][2])
        if args.skip_refine:
            gps, bps = [base[3]], [base[4]]
        print(f"[pass2] {len(gps) * len(bps)} prune settings x "
              f"{len(loc_g) * len(loc_b) * len(GRID_LAMBDA)} local points")
        for gp, bp in itertools.product(gps, bps):
            tr = collect(pool, trie, targets, sw_lo, sw_hi, gp, bp, args.jobs)
            loc = []
            for g, b, lam in itertools.product(loc_g, loc_b, GRID_LAMBDA):
                m = score_all(tr, g, b, lam)
                loc.append((objective_key(m, args.objective, bars),
                            (g, lam, b, gp, bp), m))
            loc.sort(key=lambda r: r[0], reverse=True)
            print(f"   gp {gp:.4f} bp {bp:.4f}  best t1 {loc[0][2][0]:5.2f} "
                  f"(gamma {loc[0][1][0]:.4f} beta {loc[0][1][2]:.4f} "
                  f"lambda {loc[0][1][1]:.4f})")
            if loc[0][0] > best[0]:
                best = loc[0]
        tuned = best[1]
        print(f"\nTUNED PRESET  gamma={tuned[0]} lambda={tuned[1]} beta={tuned[2]} "
              f"gammaPrune={tuned[3]} betaPrune={tuned[4]}   "
              f"sweep-half t1 {best[2][0]:.2f}")
        results["tuned_preset"] = list(tuned)

        # ── confirmation: sweep half / untouched holdout half / full val ────────
        print("\nconfirmation (same rows, both presets):")
        conf: Dict[str, object] = {}
        for tag, lo, hi in (("sweep half", sw_lo, sw_hi),
                            ("holdout half", ho_lo, ho_hi),
                            ("FULL val", fu_lo, fu_hi)):
            b_tr = collect(pool, trie, targets, lo, hi, base[3], base[4], args.jobs)
            bm = report(f"baseline / {tag}", b_tr, base)
            t_tr = (b_tr if (tuned[3], tuned[4]) == (base[3], base[4])
                    else collect(pool, trie, targets, lo, hi, tuned[3], tuned[4], args.jobs))
            tm = report(f"tuned    / {tag}", t_tr, tuned)
            print(f"  {'delta':<34} {'':>11} t1 {tm[0] - bm[0]:+5.2f}  "
                  f"t3 {tm[1] - bm[1]:+5.2f}  t5 {tm[2] - bm[2]:+5.2f}")
            conf[tag] = {"baseline": bm, "tuned": tm}
        results["confirmation"] = conf
    finally:
        pool.close()
        pool.join()

    out_path = resolve(args.workdir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
