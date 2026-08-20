#!/usr/bin/env python3
"""Phase Q gate battery — SYNTH v3 arms against v2, on Phase P's instruments.

Every instrument is imported from ``synth_gap_audit.py`` unchanged (5-fold
word-disjoint CV, final-epoch statistics, mandatory real-vs-real floor arm,
exact within-pair permutation null, the 17+6 metric batteries), so a v3 number
and a v2 number differ only in the generator that produced the arm.  The arm to
beat is **v2 C+B′+S5** — read live from ``matched_v2.npz`` with the same fold
seeds, never quoted from a document.

Pre-registered bars (PHASE_Q.md §2):

* G1  endpoints within 0.05 of v2's (or closer to real); wrong-geo < 0.05.
* G2  train-draw length mix within 0.03 of the wordfreq token mass (regression
  check — S0 is v2's code verbatim; needs ``--train-cache``).
* G3  the committed ``G3_BARS`` + minima/segment ±0.10 — same absolute bars.
* G4  beat v2's point estimates on GBM₁₇ AND MLP-speed; GBM₂₃ reported, never
  gated; UCL₉₅ ≤ 0.60 recorded as the standing open shortfall.
* GQ-D  PRDC (k = 5, coords space, MIT ``prdc``): recall ≥ control − 0.05.

The research twin's arm (``*_RESEARCH_ONLY``) runs through the same battery for
CALIBRATION ONLY — it is expected near the floor, gates nothing, and its file
must already carry the seal (`synth_v3.assert_sealed` was applied at write
time).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import load_layout  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_yandex import RU_LETTERS_31  # noqa: E402
from script_synth import endpoint_stats, wrong_geometry  # noqa: E402
from synth_gap_audit import (  # noqa: E402
    G3_BARS, GATE_VIEWS, METRIC_KEYS_17, METRIC_KEYS_EXT, gap_closure,
    gate_gbm_kfold, gate_mlp_kfold, ks_stat, metric_matrix,
    permutation_null_gbm, real_pairs, trace_metrics)

HERE = Path(__file__).resolve().parent

#: PHASE_P §2.2's registered ru wordfreq token mass (the G2 target; the v3 S0
#: draw is v2's code verbatim, so this is a regression check, bar 0.03).
RU_WORDFREQ_MASS = {"le3": 0.268, "4to6": 0.376, "ge7": 0.356, "mean_len": 5.74}


def _pass(ok: bool) -> str:
    return "PASS" if ok else "MISS"


def length_mix(words: Sequence[str]) -> Dict[str, float]:
    L = np.array([len(w) for w in words])
    return {"le3": float((L <= 3).mean()), "4to6": float(((L >= 4) & (L <= 6)).mean()),
            "ge7": float((L >= 7).mean()), "mean_len": float(L.mean())}


def prdc_arm(real: np.ndarray, synth: np.ndarray, seed: int = 1234,
             k: int = 5) -> Dict[str, Dict[str, float]]:
    """PRDC on flattened coords with the mandatory real-vs-real control.

    ``prdc``'s own self-test on two identical Gaussians returns precision 0.804,
    so the uncontrolled number is meaningless (SYNTH_V2_RESEARCH_AUDIT §3.3) —
    the control is computed every run, on a disjoint real half-split of the
    same size as the arm comparison.
    """
    from prdc import compute_prdc
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(real))
    half = len(real) // 2
    a, b = real[perm[:half]], real[perm[half:2 * half]]
    Xa = a.reshape(len(a), -1).astype(np.float64)
    Xb = b.reshape(len(b), -1).astype(np.float64)
    ctrl = compute_prdc(real_features=Xa, fake_features=Xb, nearest_k=k)
    sub = perm[:half]                       # same real half against the arm
    arm = compute_prdc(real_features=real[sub].reshape(half, -1).astype(np.float64),
                       fake_features=synth[perm[half:2 * half]].reshape(
                           half, -1).astype(np.float64), nearest_k=k)
    return {"control_real_vs_real": {m: float(v) for m, v in ctrl.items()},
            "arm_real_vs_synth": {m: float(v) for m, v in arm.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--arms", required=True,
                    help="comma-separated name=npz pairs of v3 matched arms, "
                         "e.g. v3_ship=synth_gap/matched_v3_ship.npz")
    ap.add_argument("--primary", default="v3_ship",
                    help="the arm the pre-registered bars are read on")
    ap.add_argument("--train-cache", default="",
                    help="v3 training cache dir for the G2 length-mix check")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--out", default="synth_gap/gates_v3.json")
    args = ap.parse_args()

    out_dir = resolve(args.workdir, Path("synth_gap"))
    with np.load(out_dir / "matched_v2.npz", allow_pickle=False) as d:
        real = np.array(d["real"])
        words = [str(w) for w in d["words"]]
        v1 = np.array(d["v1_repro"])
        v2 = np.array(d["C_B_S5"])
    arms: Dict[str, np.ndarray] = {}
    for spec in args.arms.split(","):
        name, _, path = spec.partition("=")
        with np.load(resolve(args.workdir, Path(path)), allow_pickle=False) as d:
            aw = [str(w) for w in d["words"]]
            assert aw == words, f"{name}: word stream differs from matched_v2"
            arms[name.strip()] = np.array(d["features"])
    assert args.primary in arms, f"--primary {args.primary} not in {list(arms)}"
    letters, centers = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    assert "".join(letters) == RU_LETTERS_31
    rep: Dict[str, object] = {"n_pairs": len(words), "arms": list(arms),
                              "primary": args.primary, "folds": args.folds,
                              "seed": args.seed}
    prim = args.primary

    # ── G1 endpoints ────────────────────────────────────────────────────────
    ctrl_geo = wrong_geometry(RU_LETTERS_31, np.random.default_rng(4242))
    g1: Dict[str, object] = {
        "real": endpoint_stats(real, words, RU_LETTERS_31, centers),
        "v2": endpoint_stats(v2, words, RU_LETTERS_31, centers)}
    for nm, F in arms.items():
        g1[nm] = endpoint_stats(F, words, RU_LETTERS_31, centers)
        g1[nm + "_wrong_geo"] = endpoint_stats(F, words, RU_LETTERS_31, ctrl_geo)
    rs, v2s = g1["real"]["start_hit"], g1["v2"]["start_hit"]
    ps = g1[prim]["start_hit"]
    g1_ok = (g1[prim + "_wrong_geo"]["start_hit"] < 0.05 and
             (abs(ps - v2s) < 0.05 or abs(ps - rs) < abs(v2s - rs)))
    rep["G1"] = {"stats": g1, "pass": g1_ok}
    print(f"\nG1 endpoints  real {rs:.4f}/{g1['real']['end_hit']:.4f}  "
          f"v2 {v2s:.4f}/{g1['v2']['end_hit']:.4f}  "
          f"{prim} {ps:.4f}/{g1[prim]['end_hit']:.4f}  wrong-geo "
          f"{g1[prim + '_wrong_geo']['start_hit']:.4f}   {_pass(g1_ok)}")

    # ── G2 length mix of the v3 training draw (regression check on S0) ─────
    if args.train_cache:
        with np.load(resolve(args.workdir,
                             Path(args.train_cache) / "train_synth.npz"),
                     allow_pickle=False) as d:
            mix = length_mix([str(w) for w in d["words"]])
        dev = max(abs(mix[k] - RU_WORDFREQ_MASS[k])
                  for k in ("le3", "4to6", "ge7"))
        g2_ok = dev <= 0.03
        rep["G2"] = {"mix": mix, "target": RU_WORDFREQ_MASS,
                     "max_dev": dev, "pass": g2_ok}
        print(f"\nG2 length mix  {mix}  vs wordfreq mass  max dev {dev:.3f}   "
              f"{_pass(g2_ok)}")

    # ── G3 kinematics, word-matched KS vs real ──────────────────────────────
    mr = trace_metrics(real, words, centers)
    agg_min = {"real": float(mr["minima"].mean() / mr["n_segments"].mean())}
    ks: Dict[str, Dict[str, float]] = {}
    for nm, F in {"v2": v2, **arms}.items():
        ms = trace_metrics(F, words, centers)
        ks[nm] = {k: ks_stat(mr[k], ms[k]) for k in METRIC_KEYS_EXT}
        agg_min[nm] = float(ms["minima"].mean() / ms["n_segments"].mean())
    shown = ("step_cv", "step_max", "sharp_turns", "turn_mean", "ac1",
             "spec_centroid", "ldlj", "minima_per_seg", "sc_slope", "sc_r2")
    print(f"\nG3 kinematic parity — KS vs real, n={len(words)}")
    print(f"{'arm':<10}" + "".join(f"{k[:9]:>11}" for k in shown) + f"{'min/seg':>9}")
    print(f"{'REAL':<10}" + "".join(f"{'-':>11}" for _ in shown)
          + f"{agg_min['real']:>9.2f}")
    for nm in ks:
        print(f"{nm:<10}" + "".join(f"{ks[nm][k]:>11.3f}" for k in shown)
              + f"{agg_min[nm]:>9.2f}")
    g3_rows = [(k, ks[prim][k], bar, ks[prim][k] < bar) for k, bar in G3_BARS]
    min_ok = abs(agg_min[prim] - agg_min["real"]) <= 0.10
    g3_ok = all(r[3] for r in g3_rows) and min_ok
    rep["G3"] = {"ks": ks, "minima_per_segment": agg_min,
                 "bars": [{"metric": k, "value": v, "bar": b, "pass": p}
                          for k, v, b, p in g3_rows],
                 "minima_pass": min_ok, "pass": g3_ok}
    for k, v, b, p in g3_rows:
        print(f"  {k:<14} {v:.3f} < {b:.2f}   {_pass(p)}")
    print(f"  {'minima/seg':<14} {agg_min[prim]:.2f} vs real "
          f"{agg_min['real']:.2f} (±0.10)   {_pass(min_ok)}")

    # ── G4 discriminability — Phase P's instruments, v2 read live ──────────
    print(f"\nG4 discriminability — {args.folds}-fold word-disjoint, "
          f"final-epoch")
    i17 = [METRIC_KEYS_EXT.index(k) for k in METRIC_KEYS_17]
    Xr = metric_matrix(real, words, centers)
    ia, ib, ws = real_pairs(real, words)
    floor_mlp = gate_mlp_kfold(real[ia], real[ib], ws, args.folds, args.seed)
    floor_gbm = gate_gbm_kfold(Xr[ia][:, i17], Xr[ib][:, i17], ws, args.folds,
                               args.seed, importances=False, keys=METRIC_KEYS_17)
    fl_speed, fl_gbm = floor_mlp["speed"]["mean"], floor_gbm["mean"]
    valid = 0.48 <= fl_speed <= 0.52 and 0.48 <= fl_gbm <= 0.52
    print(f"  floor arm ({len(ws)} real-vs-real pairs): MLP speed "
          f"{fl_speed:.4f}, GBM17 {fl_gbm:.4f} — "
          f"{'OK' if valid else 'VOID'}")
    g4: Dict[str, object] = {"floor": {"n_pairs": len(ws), "mlp": floor_mlp,
                                       "gbm17": floor_gbm, "valid": valid}}
    for nm, F in {"v1": v1, "v2": v2, **arms}.items():
        Xa = metric_matrix(F, words, centers)
        mlp = gate_mlp_kfold(F, real, words, args.folds, args.seed)
        gbm = gate_gbm_kfold(Xa[:, i17], Xr[:, i17], words, args.folds,
                             args.seed, keys=METRIC_KEYS_17)
        gbm_ext = gate_gbm_kfold(Xa, Xr, words, args.folds, args.seed)
        g4[nm] = {"mlp": mlp, "gbm17": gbm, "gbm_ext": gbm_ext}
        print(f"  {nm:<10} MLP speed {mlp['speed']['mean']:.4f} "
              f"(UCL {mlp['speed']['ucl95']:.4f})  coords "
              f"{mlp['coords']['mean']:.4f}  angles {mlp['angles']['mean']:.4f}"
              f"  GBM17 {gbm['mean']:.4f} (UCL {gbm['ucl95']:.4f})"
              f"  GBM23 {gbm_ext['mean']:.4f}  "
              + ", ".join(f"{k}={v}" for k, v in gbm_ext["top_features"][:3]),
              flush=True)
    if args.permutations > 0:
        Xp = metric_matrix(arms[prim], words, centers)
        null = permutation_null_gbm(Xp[:, i17], Xr[:, i17], words, args.folds,
                                    args.seed, args.permutations)
        g4["permutation_null_gbm"] = null
        print(f"  within-pair permutation null (GBM17, {null['n_perm']}): mean "
              f"{null['null_mean']:.4f}, p95 {null['null_p95']:.4f}, "
              f"max {null['null_max']:.4f}")
    bars = {
        "gbm17_lt_v2": (g4[prim]["gbm17"]["mean"], g4["v2"]["gbm17"]["mean"]),
        "mlp_speed_lt_v2": (g4[prim]["mlp"]["speed"]["mean"],
                            g4["v2"]["mlp"]["speed"]["mean"])}
    g4_ok = valid and all(a < b for a, b in bars.values())
    closure = {nm: {"mlp_speed": gap_closure(g4["v1"]["mlp"]["speed"]["mean"],
                                             g4[nm]["mlp"]["speed"]["mean"]),
                    "gbm17": gap_closure(g4["v1"]["gbm17"]["mean"],
                                         g4[nm]["gbm17"]["mean"])}
               for nm in (["v2"] + list(arms))}
    g4["bars"] = {k: {"arm": a, "v2": b, "pass": a < b}
                  for k, (a, b) in bars.items()}
    g4["closure_vs_v1"] = closure
    g4["ucl95_standard"] = {
        "bar": 0.60,
        "mlp_speed_ucl": g4[prim]["mlp"]["speed"]["ucl95"],
        "gbm17_ucl": g4[prim]["gbm17"]["ucl95"],
        "met": bool(max(g4[prim]["mlp"]["speed"]["ucl95"],
                        g4[prim]["gbm17"]["ucl95"]) <= 0.60)}
    g4["pass"] = g4_ok
    rep["G4"] = g4
    for k, (a, b) in bars.items():
        print(f"  {k:<18} {a:.4f} vs v2 {b:.4f}   {_pass(a < b)}")
    print(f"  UCL95 standard <= 0.60: "
          f"{'MET' if g4['ucl95_standard']['met'] else 'OPEN SHORTFALL (recorded)'}")

    # ── GQ-D diversity guard: PRDC with the mandatory control ──────────────
    prdc = prdc_arm(real, arms[prim], seed=args.seed)
    rec_c = prdc["control_real_vs_real"]["recall"]
    rec_a = prdc["arm_real_vs_synth"]["recall"]
    gqd_ok = rec_a >= rec_c - 0.05
    prdc["pass"] = gqd_ok
    rep["GQD_prdc"] = prdc
    print(f"\nGQ-D PRDC (k=5, coords): arm P/R/D/C "
          f"{prdc['arm_real_vs_synth']['precision']:.3f}/"
          f"{rec_a:.3f}/{prdc['arm_real_vs_synth']['density']:.3f}/"
          f"{prdc['arm_real_vs_synth']['coverage']:.3f}  control recall "
          f"{rec_c:.3f}  bar recall >= {rec_c - 0.05:.3f}   {_pass(gqd_ok)}")

    rep["verdict"] = {"G1": g1_ok, "G3": g3_ok, "G4": g4_ok, "GQD": gqd_ok}
    if "G2" in rep:
        rep["verdict"]["G2"] = rep["G2"]["pass"]  # type: ignore[index]
    out = resolve(args.workdir, Path(args.out))
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"\nverdict {rep['verdict']}\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
