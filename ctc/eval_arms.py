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
from model import MAX_KEYS  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
import seal  # noqa: E402
from sweep_scoring import (TraceCandidates, _init_worker, build_emissions,  # noqa: E402
                           collect, score_grid, strata)

#: The published encoder-only preset every committed number is quoted at.
#: ``--preset`` rebinds this module-global so every downstream helper (``top``,
#: the beam's prune setting) sees one consistent scoring setting; Phase E's E1
#: arm re-tunes it per model, and a report that mixes presets is meaningless.
PRESET = (ENC_GAMMA, ENC_LAMBDA, ENC_BETA, ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)


def parse_preset(spec: str) -> Tuple[float, float, float, float, float]:
    """``"gamma,lambda,beta,gammaPrune,betaPrune"`` -> the 5-tuple :data:`PRESET` holds."""
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 5:
        raise SystemExit(f"--preset needs 5 comma-separated floats, got {len(parts)}: "
                         "gamma,lambda,beta,gammaPrune,betaPrune")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def subset(traces: Sequence[TraceCandidates], keep: np.ndarray) -> List[TraceCandidates]:
    """Rows of *traces* selected by a boolean mask."""
    return [t for t, k in zip(traces, keep) if k]


def top(traces: Sequence[TraceCandidates]) -> Tuple[float, float, float]:
    """-> ``(t1, t3, t5)`` at :data:`PRESET`; ``(nan,)*3`` for an empty subset."""
    if not traces:
        return float("nan"), float("nan"), float("nan")
    g, lam, b = PRESET[0], PRESET[1], PRESET[2]
    return score_grid(list(traces), g, b, lam)


def onnx_latency(onnx_path: Path, layout: Path, runs: int = 100,
                 warmup: int = 20) -> Tuple[float, float]:
    """Single-call CPU latency of the exported graph -> ``(mean_ms, p90_ms)``.

    One session, one thread, batch 1, fixed shapes — the on-device shape of the
    call the IME makes once per swipe. Threads are pinned to 1 because a phone
    will not hand the encoder the whole machine, and an unpinned ORT session
    silently uses every core and reports a number the device can never reach.
    """
    import onnxruntime as ort
    from futo_decoder_eval import load_layout as _ll
    letters, centers = _ll(layout)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts,
                                providers=["CPUExecutionProvider"])
    keys = np.zeros((1, 64, 2), np.float32)
    keys[0, :len(letters)] = centers
    mask = np.zeros((1, 64), bool)
    mask[0, :len(letters)] = True
    feed = {"features": np.random.rand(1, 2, 64).astype(np.float32),
            "layout_keys": keys, "layout_mask": mask}
    for _ in range(warmup):
        sess.run(["log_emissions"], feed)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(["log_emissions"], feed)
        ts.append((time.perf_counter() - t) * 1000.0)
    a = np.array(ts)
    return float(a.mean()), float(np.percentile(a, 90))


def build_emissions_refined(base_ckpt: Path, head_ckpt: Path, layout: Path, rows,
                            cache: Path, force: bool) -> Tuple[np.ndarray, List[str]]:
    """Sliced ``[N,32,27]`` emissions **after** the phase-2 refinement head.

    The head replaces the encoder's emissions before the beam, so the whole eval
    path downstream is unchanged — only the array fed into it differs. Both
    modules run in torch here rather than through ONNX: the head's exported graph
    would have to be chained onto the encoder's three outputs, and the encoder's
    own torch/ONNX parity is asserted at export time (<2e-5 on this view), so the
    torch path is the same quantity with one less moving part.

    :param base_ckpt: ``ckpt/<arm>/best.pt`` — the frozen encoder the head was
        trained on. ``train_refine.py`` records the base's sha256, which is
        checked here so a head can never be evaluated on the wrong encoder.
    """
    import torch
    from futo_decoder_eval import featurize as _featurize, load_layout as _ll
    from model import CtcRefineHead, build_refine_input, encoder_from_checkpoint
    from paths import sha256_file

    letters, key_centers = _ll(layout)
    n_letters = len(letters)
    if cache.exists() and not force:
        with np.load(cache) as d:
            if len(d["emissions"]) >= len(rows):
                print(f"[cache] {cache} ({len(d['emissions'])} rows)")
                return np.array(d["emissions"][: len(rows)]), letters

    hck = torch.load(head_ckpt, map_location="cpu", weights_only=True)
    want = hck.get("base_sha256", "")
    got = sha256_file(base_ckpt)
    if want and want != got:
        raise SystemExit(f"{head_ckpt}: head was trained on base {want[:12]}, but "
                         f"{base_ckpt} is {got[:12]}")
    bck = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    base = encoder_from_checkpoint(bck)
    base.load_state_dict(bck["model"])
    head = CtcRefineHead(num_letters=int(hck.get("num_letters", n_letters)),
                         hidden=int(hck["hidden"]))
    head.load_state_dict(hck["head"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, head = base.to(device).eval(), head.to(device).eval()

    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:n_letters] = key_centers
    mask = np.zeros((MAX_KEYS,), bool)
    mask[:n_letters] = True
    kt = torch.from_numpy(keys)[None].to(device)
    mt = torch.from_numpy(mask)[None].to(device)

    t0 = time.time()
    feats = np.stack([_featurize(xs, ys, ts) for _, xs, ys, ts in rows])  # [N,2,64]
    out = np.empty((len(rows), 32, n_letters + 1), np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), 512):
            f = torch.from_numpy(feats[s:s + 512]).to(device)
            b = f.shape[0]
            log_e, coeff, lam = base(f, kt.expand(b, -1, -1), mt.expand(b, -1))
            ref = head(build_refine_input(log_e, coeff, lam, n_letters))
            out[s:s + b] = ref.cpu().numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emissions=out)
    print(f"[cache] built {len(rows)} refined rows in {time.time() - t0:.1f}s "
          f"-> {cache}")
    return out, letters


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
    ap.add_argument("--latency-runs", type=int, default=0, dest="latency_runs",
                    help="also time the exported graph: single-thread batch-1 CPU "
                         "mean/p90 ms over N runs (0 = skip)")
    ap.add_argument("--rebuild-cache", action="store_true", dest="rebuild_cache",
                    help="recompute ckpt/<arm>/eval_emissions.npz; required after an "
                         "arm is retrained, since the cache is keyed only by run dir")
    ap.add_argument("--preset", default="",
                    help="override the published enc preset with "
                         "'gamma,lambda,beta,gammaPrune,betaPrune' (Phase E E1)")
    ap.add_argument("--refine-ckpt", default="", dest="refine_ckpt",
                    help="run name under ckpt/ holding a train_refine.py best.pt; "
                         "its refined [32,27] log-probs REPLACE the encoder's "
                         "emissions before the beam (Phase E E2). Emissions are then "
                         "computed in torch from ckpt/<arm>/best.pt, not from ONNX.")
    ap.add_argument("--emissions-name", default="", dest="emissions_name",
                    help="basename of the per-arm emission cache (default "
                         "eval_emissions.npz, or eval_emissions_<refine>.npz when "
                         "--refine-ckpt is given)")
    ap.add_argument("--out", default="cache/phase_a_results.json")
    seal.add_argument(ap)
    args = ap.parse_args()

    if "test" in Path(args.test).name:
        raise SystemExit(f"refusing to decode {args.test}: Phase A is val-only")

    global PRESET
    if args.preset:
        PRESET = parse_preset(args.preset)
        print(f"scoring preset OVERRIDDEN: gamma {PRESET[0]} lambda {PRESET[1]} "
              f"beta {PRESET[2]} gammaPrune {PRESET[3]} betaPrune {PRESET[4]}")

    rows = load_test(resolve(args.workdir, args.test))
    seal.check(rows, args.unseal_test, what="eval_arms.py")
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

    refine_run = (resolve(args.workdir, Path("ckpt") / args.refine_ckpt)
                  if args.refine_ckpt else None)
    em_name = args.emissions_name or (
        f"eval_emissions_{args.refine_ckpt}.npz" if refine_run else "eval_emissions.npz")

    results: Dict[str, object] = {"preset": list(PRESET), "n_val": n,
                                  "refine_ckpt": args.refine_ckpt}
    for arm in args.arms.split(","):
        run = resolve(args.workdir, Path("ckpt") / arm)
        onnx = run / args.onnx_name
        if refine_run is None and not onnx.exists():
            print(f"{arm}: MISSING {onnx} — run export_onnx.py first; skipped\n")
            continue
        t0 = time.time()
        if refine_run is not None:
            emissions, letters = build_emissions_refined(
                run / "best.pt", refine_run / "best.pt", args.layout, rows,
                run / em_name, args.rebuild_cache)
        else:
            emissions, letters = build_emissions(onnx, args.layout, rows,
                                                 run / em_name,
                                                 args.rebuild_cache)
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
        # The FUTO baselines are published per length stratum (<=3 / 4+) and the
        # two strata move in opposite directions under several of the levers this
        # campaign has tried, so an aggregate alone cannot be compared to them.
        st = strata(list(traces), PRESET[0], PRESET[2], PRESET[1])
        rec["strata"] = {k: {"n": v[0], "t1": v[1]} for k, v in st.items()}
        print(f"=== {arm} ===")
        print(f"  {'FULL val':<24} n={n:<5} t1 {t1:5.2f}  t3 {t3:5.2f}  t5 {t5:5.2f}"
              f"   (<=3 {st['<=3'][1]:5.2f} n={st['<=3'][0]} / "
              f"4+ {st['4+'][1]:5.2f} n={st['4+'][0]})")
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
            # A FUTO-only tier leaves every HWS val row "clean", so an undivided
            # clean number is mostly a cross-corpus transfer score. Split it.
            for sname, ssel in (("futo", is_futo), ("hws", ~is_futo)):
                sub = sel & ssel
                if not sub.any():
                    continue
                ss = top(subset(traces, sub))
                rec[f"clean_{name}_{sname}"] = {"n": int(sub.sum()), "t1": ss[0],
                                                "t3": ss[1], "t5": ss[2]}
                print(f"    {'|- ' + sname:<22} n={int(sub.sum()):<5} t1 {ss[0]:5.2f}"
                      f"  t3 {ss[1]:5.2f}  t5 {ss[2]:5.2f}")
        if own not in masks:
            print(f"  (no own mask for '{own}' in {args.masks})")
        if args.latency_runs and onnx.exists():
            ms, p90 = onnx_latency(onnx, args.layout, args.latency_runs)
            rec["latency_ms_mean"] = ms
            rec["latency_ms_p90"] = p90
            print(f"  {'latency (1 thread)':<24} mean {ms:6.2f} ms   p90 {p90:6.2f} ms"
                  f"   ({args.latency_runs} runs)")
        ck = run / "best.pt"
        if ck.exists():
            import torch
            meta = torch.load(ck, map_location="cpu", weights_only=True)
            # Pre-Phase-D runs select on val_greedy (a fraction); Phase D selects
            # on beam top-1 over a 2,000-row val prefix (a percent). Report both
            # the metric name and the checkpoint's own greedy/beam values so a
            # mixed table cannot be misread.
            sel = str(meta.get("select_metric", "val_greedy"))
            rec["select_metric"] = sel
            rec["best_select"] = float(meta["best"])
            rec["best_epoch"] = int(meta["best_epoch"])
            rec["step"] = int(meta["step"])
            rec["ckpt_val_greedy"] = float(meta.get("val_greedy", float("nan")))
            rec["ckpt_beam2000_t1"] = float(meta.get("val_beam_t1", float("nan")))
            shown = (meta["best"] * 100 if sel == "val_greedy" else meta["best"])
            print(f"  selected on {sel} = {shown:.2f} @ epoch {meta['best_epoch']} "
                  f"(step {meta['step']}); ckpt greedy "
                  f"{rec['ckpt_val_greedy'] * 100:.2f}%  beam2000 t1 "
                  f"{rec['ckpt_beam2000_t1']:.2f}")
        print(f"  [{time.time() - t0:.1f}s]\n")
        results[arm] = rec

    p = resolve(args.workdir, args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
