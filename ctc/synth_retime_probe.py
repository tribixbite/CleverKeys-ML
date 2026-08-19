#!/usr/bin/env python3
"""Re-timing probe — the measurement harness behind `SYNTH_V2_RESEARCH_AUDIT.md`.

`synth_gap_audit.py` measured that v1 synthesis is 90.4 % separable from real
traces on the speed profile alone.  `SYNTH_V2_DESIGN.md` proposed four fixes and
pre-registered gates for them.  This script **builds the proposed fixes and
measures them** on the same 9,416 word-matched Yandex pairs, so the design's
expectations can be checked before any generator rewrite or retrain.

Everything is CPU except the gate MLP (seconds on GPU).  No model training.

Stages
------
``--stage law``
    Within-trace regression of per-segment TIME share on per-segment ideal
    LENGTH share:  ``T_k ∝ L_k^alpha``.  alpha = 1 is constant mean speed across
    segments; alpha < 1 is isochrony (longer segments traversed faster).  Fit on
    real English donors (HWS/FUTO, MIT — the licence-clean footing) and reported
    for real Russian as a TRANSFER CHECK only, never as a fitting footing.

``--stage variants``
    Reproduce the v1 matched synth bit-exactly from the committed
    ``synth_gap/matched.npz`` words, then build the re-timing / donor-selection
    variants and score the §1.1 metric battery (KS vs the real partner rows).

``--stage classifier``
    The §1.4 gate on every variant, plus the two control arms the design never
    measured: a word-matched **real-vs-real** floor and a **label-shuffled H0**
    arm that prices the max-over-epochs statistic's optimism.

``--stage gbm``
    A stronger gate than the MLP — gradient boosting on the 17 interpretable
    trace metrics — with permutation importances that localise which metric
    still betrays synthesis.  For a GATE the instrument must be the strongest
    available: a weak classifier that fails to separate proves nothing.

Variants
--------
``v1_repro``      the committed v1 mechanism (asserted bit-identical)
``C_only``        fix C: geometry-matched donor draw (k=16 reservoir, min
                  Σ|log(L_dst/L_src)| over segments)
``B_global``      fix B exactly as SYNTH_V2_DESIGN §3.1 S4 specifies it —
                  resample the warped polyline at the donor's GLOBAL cumulative
                  arc-length fractions
``B_seg_a{1,05}`` amended fix B — vertex-aligned per-segment re-timing with
                  cross-segment reallocation exponent alpha (see `retime_segment`)
``C_B_*``         C composed with the corresponding B
``*_bw``          oracle-duration acquisition emulation (see `bandwidth_emulate`):
                  NOT a shippable mechanism, it reads the real partner's
                  duration; it bounds how much of the residual turn/jaggedness
                  gap is the §1.5 sampling artefact rather than motor behaviour
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cyrillic_synth import build_donor_index, collapse  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from layout_aug import retime_global, retime_segment, warp_path  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_yandex import RU_LETTERS_31  # noqa: E402
from synth_gap_audit import feature_views, ks_stat, trace_metrics  # noqa: E402

#: The 17 metrics `synth_gap_audit.trace_metrics` returns, in report order.
METRIC_KEYS = ("step_cv", "step_max", "sharp_turns", "turn_mean", "turn_total",
               "straightness", "key_cover", "dwell_frac", "start_dwell",
               "end_dwell", "start_d", "end_d", "path_len", "step_mean",
               "dup_frac", "dwell_run_max", "speed_asym")

# ── warp_path, instrumented ─────────────────────────────────────────────────────

def warp_debug(feats: np.ndarray, letters: np.ndarray, src_centers: np.ndarray,
               dst_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`layout_aug.warp_path` plus its monotone-DP per-point source segment index.

    The correspondence re-timing needs is computed inside the production warp;
    Phase P exposed it there (``return_assign=True``) instead of keeping the
    audit's duplicate of the maths, so there is one implementation to trust.
    """
    return warp_path(feats, letters, src_centers, dst_centers, return_assign=True)


# ── re-timing mechanisms ────────────────────────────────────────────────────────

def bandwidth_emulate(path: np.ndarray, dur_ms: float) -> np.ndarray:
    """Oracle acquisition emulation — `layout_aug.resample_bandwidth` at the real
    partner's duration.

    NOT a shippable mechanism (it reads the validator); it bounds how much of the
    residual turn/jaggedness gap is the §1.5 sampling artefact rather than motor
    behaviour.  The generator's S5 stage calls the same function with a duration
    drawn from a model fit on MIT English only.
    """
    from layout_aug import resample_bandwidth
    return resample_bandwidth(path, dur_ms)


# ── stage: the time-allocation law ──────────────────────────────────────────────

def seg_shares(feats: np.ndarray, seqs: Sequence[np.ndarray],
               centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment (log time share, log ideal-length share, trace id)."""
    lt, ll, tid = [], [], []
    for r in range(len(feats)):
        seq = seqs[r]
        if len(seq) < 3:                      # need >=2 segments for a within-fit
            continue
        pts = centers[seq].astype(np.float64)
        _, j = warp_debug(feats[r], np.arange(len(seq)), pts, pts)
        L = np.hypot(*np.diff(pts, axis=0).T)
        n_k = np.bincount(j, minlength=len(L)).astype(np.float64)
        keep = (n_k > 0) & (L > 1e-6)
        if keep.sum() < 2:
            continue
        lt.append(np.log(n_k[keep] / n_k[keep].sum()))
        ll.append(np.log(L[keep] / L[keep].sum()))
        tid.append(np.full(int(keep.sum()), r))
    return np.concatenate(lt), np.concatenate(ll), np.concatenate(tid)


def within_slope(y: np.ndarray, x: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    """Within-group OLS slope (group means removed) and its R²."""
    order = np.argsort(g, kind="mergesort")
    y, x, g = y[order], x[order], g[order]
    _, start, cnt = np.unique(g, return_index=True, return_counts=True)
    yc = y - np.repeat(np.add.reduceat(y, start) / cnt, cnt)
    xc = x - np.repeat(np.add.reduceat(x, start) / cnt, cnt)
    b = float((xc * yc).sum() / max((xc * xc).sum(), 1e-12))
    r2 = 1.0 - float(((yc - b * xc) ** 2).sum()) / max(float((yc ** 2).sum()), 1e-12)
    return b, r2


def stage_law(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    res: Dict[str, object] = {}
    _, qwerty = load_layout(HERE / "en_qwerty.json")
    cache = resolve(args.workdir, Path("cache"))
    for name in args.donors.split(","):
        with np.load(cache / name.strip()) as d:
            f = np.array(d["features"][: args.rows])
            ws = [str(w) for w in d["words"][: args.rows]]
        seqs = [collapse(np.frombuffer(w.encode(), np.uint8).astype(np.int64) - 97)
                for w in ws]
        b, r2 = within_slope(*seg_shares(f, seqs, qwerty))
        res[name.strip()] = {"alpha": b, "r2": r2}
        print(f"{name.strip():<20} alpha={b:.3f}  R2={r2:.3f}")
    with np.load(out_dir / "matched.npz", allow_pickle=False) as d:
        real, synth = np.array(d["real"]), np.array(d["synth"])
        words = [str(w) for w in d["words"]]
    ru_idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    _, ru_c = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    seqs = [collapse(np.array([ru_idx[c] for c in w], np.int64)) for w in words]
    for label, F in (("ru_real (check only)", real[: args.rows]),
                     ("ru_v1synth", synth[: args.rows])):
        b, r2 = within_slope(*seg_shares(F, seqs, ru_c))
        res[label] = {"alpha": b, "r2": r2}
        print(f"{label:<20} alpha={b:.3f}  R2={r2:.3f}")
    (out_dir / "law.json").write_text(json.dumps(res, indent=1))
    return 0


# ── stage: variants ─────────────────────────────────────────────────────────────

def build_variants(args: argparse.Namespace):
    out_dir = resolve(args.workdir, Path("synth_gap"))
    with np.load(out_dir / "matched.npz", allow_pickle=False) as d:
        real, v1 = np.array(d["real"]), np.array(d["synth"])
        words = [str(w) for w in d["words"]]
        dur = np.array(d["raw_duration_ms"])
    n = len(words) if args.rows <= 0 else min(args.rows, len(words))
    cache = resolve(args.workdir, Path("cache"))
    donor_feats, donor_seqs, by_count = build_donor_index(
        [cache / p.strip() for p in args.donors.split(",")])
    _, qwerty = load_layout(HERE / "en_qwerty.json")
    _, ru_c = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    ru_idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    seqs = [collapse(np.array([ru_idx[c] for c in w], np.int64)) for w in words]

    # v1's donor draw, reproduced bit-exactly (same seed and call order as
    # `synth_gap_audit.synth_matched`).
    rng = np.random.default_rng(20260819)
    picks = [int(by_count[len(s)][rng.integers(len(by_count[len(s)]))]) for s in seqs]

    # Fix C: geometry-matched draw on an independent stream.
    rng_c = np.random.default_rng(20260819)
    picks_c: List[int] = []
    for seq in seqs:
        pool = by_count[len(seq)]
        cand = pool[rng_c.integers(len(pool), size=min(args.k_cand, len(pool)))]
        Ld = np.hypot(*np.diff(ru_c[seq].astype(np.float64), axis=0).T)
        best, bestcost = int(cand[0]), np.inf
        for ci in cand:
            Ls = np.hypot(*np.diff(qwerty[donor_seqs[int(ci)]].astype(np.float64),
                                   axis=0).T)
            cost = float(np.abs(np.log(np.maximum(Ld, 1e-6) /
                                       np.maximum(Ls, 1e-6))).sum())
            if cost < bestcost:
                best, bestcost = int(ci), cost
        picks_c.append(best)

    names = ("v1_repro", "B_global", "B_seg_a1", "B_seg_a05", "C_only",
             "C_B_global", "C_B_seg_a1", "C_B_seg_a05")
    var = {k: np.zeros((n, 2, 64), np.float32) for k in names}
    ratios: Dict[str, List[float]] = {"v1": [], "C": []}
    for i in range(n):
        seq = seqs[i]
        S = len(seq)
        dst = ru_c[seq]
        for tag, di in (("v1", picks[i]), ("C", picks_c[i])):
            src = qwerty[donor_seqs[di]]
            warped, j = warp_debug(donor_feats[di], np.arange(S), src, dst)
            np.clip(warped, 0.0, 1.0, out=warped)
            if S >= 2:
                ratios[tag].extend(list(
                    np.hypot(*np.diff(dst.astype(np.float64), axis=0).T) /
                    np.maximum(np.hypot(*np.diff(src.astype(np.float64), axis=0).T),
                               1e-6)))
            pre = "" if tag == "v1" else "C_"
            if S < 2:                            # no polyline: re-timing undefined
                for k in names:
                    if k.startswith("C_") == (tag == "C") and k != "v1_repro":
                        var[k][i] = warped
                if tag == "v1":
                    var["v1_repro"][i] = warped
                continue
            if tag == "v1":
                var["v1_repro"][i] = warped
            else:
                var["C_only"][i] = warped
            var[pre + "B_global"][i] = np.clip(
                retime_global(warped, donor_feats[di]), 0, 1)
            var[pre + "B_seg_a1"][i] = np.clip(
                retime_segment(warped, donor_feats[di], j, 1.0), 0, 1)
            var[pre + "B_seg_a05"][i] = np.clip(
                retime_segment(warped, donor_feats[di], j, 0.5), 0, 1)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    for base in ("C_B_seg_a05", "C_B_global"):
        var[base + "_bw"] = np.stack([bandwidth_emulate(var[base][i], float(dur[i]))
                                      for i in range(n)])
    return real[:n], v1[:n], var, words[:n], ratios


def stage_variants(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    real, v1, var, words, ratios = build_variants(args)
    delta = float(np.abs(var["v1_repro"] - v1).max())
    print(f"v1 reproduction max|delta| vs matched.npz: {delta:.2e}")
    assert delta == 0.0, "v1 mechanism not reproduced — the probe is invalid"
    for tag, r in ratios.items():
        q = np.percentile(np.array(r), [5, 25, 50, 75, 95])
        print(f"per-segment dst/src length ratio [{tag}]: "
              + "  ".join(f"p{p}={v:.2f}" for p, v in zip([5, 25, 50, 75, 95], q)))
    _, ru_c = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    mr = trace_metrics(real, words, ru_c)
    rows: Dict[str, Dict[str, float]] = {}
    print(f"\n{'variant':<16}" + "".join(f"{k[:9]:>10}" for k in METRIC_KEYS[:8]))
    for name in var:
        ms = trace_metrics(var[name], words, ru_c)
        rows[name] = {k: ks_stat(mr[k], ms[k]) for k in METRIC_KEYS}
        rows[name + "__mean"] = {k: float(ms[k].mean()) for k in METRIC_KEYS}
        print(f"{name:<16}" + "".join(f"{rows[name][k]:>10.3f}"
                                      for k in METRIC_KEYS[:8]))
    print(f"\n{'variant':<16}" + "".join(f"{k[:9]:>10}" for k in METRIC_KEYS[8:]))
    for name in var:
        print(f"{name:<16}" + "".join(f"{rows[name][k]:>10.3f}"
                                      for k in METRIC_KEYS[8:]))
    np.savez_compressed(out_dir / "variants.npz", real=real,
                        words=np.array(words), **var)
    (out_dir / "variants_ks.json").write_text(json.dumps(
        {"ks": rows, "real_mean": {k: float(mr[k].mean()) for k in METRIC_KEYS},
         "ratio_p95": {t: float(np.percentile(np.array(r), 95))
                       for t, r in ratios.items()}}, indent=1))
    print(f"\n-> {out_dir / 'variants.npz'}")
    return 0


# ── stage: gates ────────────────────────────────────────────────────────────────

def train_mlp2(x_tr, y_tr, x_te, y_te, seed=0, epochs=40) -> Tuple[float, float]:
    """The committed §1.4 MLP -> (max-over-epochs acc, FINAL-epoch acc).

    The committed harness reports the max, which selects on the test set; the
    final-epoch number is the honest one and both are recorded so the optimism
    can be priced against the label-shuffled H0 arm.
    """
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, device=dev)
    yt = torch.tensor(y_tr, device=dev)
    xv = torch.tensor((x_te - mu) / sd, device=dev)
    yv = torch.tensor(y_te, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(x_tr.shape[1], 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best = acc = 0.0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(xv).argmax(1) == yv).float().mean())
        best = max(best, acc)
    return best, acc


def _word_disjoint_masks(words: Sequence[str], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(words))
    test = set(np.array(uniq)[rng.permutation(len(uniq))[:len(uniq) // 5]])
    te = np.array([w in test for w in words])
    return ~te, te


def run_gate(a: np.ndarray, b: np.ndarray, words: Sequence[str], seed: int,
             views=("speed", "coords", "angles")) -> Dict[str, object]:
    tr, te = _word_disjoint_masks(words, seed)
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a), np.int64), np.ones(len(b), np.int64)])
    m_tr, m_te = np.concatenate([tr, tr]), np.concatenate([te, te])
    v = feature_views(x)
    out: Dict[str, object] = {"n_test_pairs": int(te.sum())}
    for name in views:
        mx, fin = train_mlp2(v[name][m_tr], y[m_tr], v[name][m_te], y[m_te], seed=seed)
        out[name] = {"max": mx, "final": fin}
    return out


def _real_pairs(real: np.ndarray, words: Sequence[str], seed: int = 7):
    """Word-matched real-vs-real pairs (words with >= 2 real traces)."""
    by: Dict[str, List[int]] = defaultdict(list)
    for i, w in enumerate(words):
        by[w].append(i)
    rng = np.random.default_rng(seed)
    ia, ib, ws = [], [], []
    for w, idxs in by.items():
        if len(idxs) >= 2:
            p = rng.permutation(idxs)
            ia.append(p[0]); ib.append(p[1]); ws.append(w)
    return np.array(ia), np.array(ib), ws


def stage_classifier(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    d = np.load(out_dir / "variants.npz", allow_pickle=False)
    real = np.array(d["real"])
    words = [str(w) for w in d["words"]]
    res: Dict[str, object] = {}
    for nm in [k for k in d.files if k not in ("real", "words")]:
        r = run_gate(real, np.array(d[nm]), words, args.seed)
        res[nm] = r
        print(f"{nm:<16} speed {r['speed']['max']:.4f}/{r['speed']['final']:.4f}  "
              f"coords {r['coords']['max']:.4f}/{r['coords']['final']:.4f}  "
              f"angles {r['angles']['max']:.4f}/{r['angles']['final']:.4f}"
              "   (max/final)", flush=True)
    ia, ib, ws = _real_pairs(real, words)
    print(f"\ncontrol arms on {len(ws)} word-matched REAL pairs")
    res["real_vs_real"] = run_gate(real[ia], real[ib], ws, args.seed)
    pool = np.concatenate([real[ia], real[ib]])
    pool = pool[np.random.default_rng(11).permutation(len(pool))]
    res["null_shuffled"] = run_gate(pool[:len(ws)], pool[len(ws):], ws, args.seed)
    for k in ("real_vs_real", "null_shuffled"):
        r = res[k]
        print(f"{k:<16} speed {r['speed']['max']:.4f}/{r['speed']['final']:.4f}  "
              f"coords {r['coords']['max']:.4f}/{r['coords']['final']:.4f}  "
              f"angles {r['angles']['max']:.4f}/{r['angles']['final']:.4f}")
    (out_dir / "variants_clf.json").write_text(json.dumps(res, indent=1))
    return 0


def stage_gbm(args: argparse.Namespace) -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    out_dir = resolve(args.workdir, Path("synth_gap"))
    d = np.load(out_dir / "variants.npz", allow_pickle=False)
    real = np.array(d["real"])
    words = [str(w) for w in d["words"]]
    _, ru_c = load_layout(HERE / "layouts" / "ru_jcuken_default.json")

    def mfeat(F):
        m = trace_metrics(F, words, ru_c)
        return np.stack([m[k] for k in METRIC_KEYS], 1)

    Xr = mfeat(real)

    def run(Xa, Xb, ws):
        tr, te = _word_disjoint_masks(ws, args.seed)
        X = np.concatenate([Xa, Xb])
        y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
        mtr, mte = np.concatenate([tr, tr]), np.concatenate([te, te])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             random_state=args.seed)
        clf.fit(X[mtr], y[mtr])
        acc = float(clf.score(X[mte], y[mte]))
        pi = permutation_importance(clf, X[mte], y[mte], n_repeats=3,
                                    random_state=args.seed, n_jobs=4)
        top = [(METRIC_KEYS[i], round(float(pi.importances_mean[i]), 4))
               for i in np.argsort(-pi.importances_mean)[:5]]
        return acc, top

    res = {}
    for nm in [k for k in d.files if k not in ("real", "words")]:
        acc, top = run(Xr, mfeat(np.array(d[nm])), words)
        res[nm] = {"acc": acc, "top_features": top}
        print(f"{nm:<16} GBM {acc:.4f}   " +
              ", ".join(f"{k}={v}" for k, v in top), flush=True)
    ia, ib, ws = _real_pairs(real, words)
    acc, top = run(Xr[ia], Xr[ib], ws)
    res["real_vs_real"] = {"acc": acc, "n_pairs": len(ws), "top_features": top}
    print(f"{'real_vs_real':<16} GBM {acc:.4f} ({len(ws)} pairs)")
    (out_dir / "variants_gbm.json").write_text(json.dumps(res, indent=1))
    return 0


# ── stage: speed–curvature coupling ─────────────────────────────────────────────

def speed_curvature(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trace (slope, R²) of log speed on log curvature.

    Menger curvature (parameterization-invariant, so time-uniform samples are
    fine): ``κ = 4·Area/(|ab|·|bc|·|ca|)``; speed at the middle sample is the
    mean of its two adjacent step lengths.  The two-thirds power law predicts
    slope −1/3; what matters here is not the absolute value (discrete-curvature
    estimators are biased, per Schaal & Sternad 2001) but the comparison between
    real and synthetic populations computed identically.
    """
    P = feats.transpose(0, 2, 1).astype(np.float64)
    st = np.diff(P, axis=1)
    d = np.hypot(st[..., 0], st[..., 1])
    a, b, c = P[:, :-2], P[:, 1:-1], P[:, 2:]
    area = 0.5 * np.abs((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) -
                        (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    l1 = np.hypot(*(b - a).transpose(2, 0, 1))
    l2 = np.hypot(*(c - b).transpose(2, 0, 1))
    l3 = np.hypot(*(c - a).transpose(2, 0, 1))
    kap = 4.0 * area / np.maximum(l1 * l2 * l3, 1e-12)
    v = 0.5 * (d[:, :-1] + d[:, 1:])
    slope = np.full(len(P), np.nan)
    r2 = np.full(len(P), np.nan)
    for i in range(len(P)):
        m = (kap[i] > 1e-3) & (v[i] > 1e-9)
        if m.sum() < 8:
            continue
        x, y = np.log(kap[i][m]), np.log(v[i][m])
        coef = np.polyfit(x, y, 1)
        slope[i] = coef[0]
        r2[i] = 1.0 - (y - np.polyval(coef, x)).var() / max(y.var(), 1e-12)
    ok = ~np.isnan(slope)
    return slope[ok], r2[ok]


def stage_coupling(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    d = np.load(out_dir / "variants.npz", allow_pickle=False)
    rs, rr = speed_curvature(np.array(d["real"]))
    res = {"real": {"slope": float(rs.mean()), "r2": float(rr.mean())}}
    print(f"{'variant':<17}{'slope':>8}{'KS':>7}{'R2':>8}{'KS':>7}")
    print(f"{'REAL':<17}{rs.mean():>8.3f}{'-':>7}{rr.mean():>8.3f}{'-':>7}")
    for nm in [k for k in d.files if k not in ("real", "words")]:
        s, r = speed_curvature(np.array(d[nm]))
        res[nm] = {"slope": float(s.mean()), "ks_slope": ks_stat(rs, s),
                   "r2": float(r.mean()), "ks_r2": ks_stat(rr, r)}
        print(f"{nm:<17}{s.mean():>8.3f}{res[nm]['ks_slope']:>7.3f}"
              f"{r.mean():>8.3f}{res[nm]['ks_r2']:>7.3f}")
    (out_dir / "variants_coupling.json").write_text(json.dumps(res, indent=1))
    return 0


# ── stage: en→en control ────────────────────────────────────────────────────────

def stage_enen(args: argparse.Namespace) -> int:
    """Transplant within English: prices the A1 donor-population hypothesis.

    One half of HWS is treated as "real"; the donor for each row is a
    vertex-count-matched trace from the DISJOINT other half, transplanted onto
    the SAME word on the SAME QWERTY geometry.  Every difference is generator
    error with no cross-script and no cross-population component.
    """
    import synth_gap_audit as sga
    out_dir = resolve(args.workdir, Path("synth_gap"))
    _, qw = load_layout(HERE / "en_qwerty.json")
    cache = resolve(args.workdir, Path("cache"))
    with np.load(cache / args.enen_corpus) as d:
        F = np.array(d["features"])
        W = [str(w) for w in d["words"]]
    rng = np.random.default_rng(4242)
    perm = rng.permutation(len(W))
    real_side, donor_side = perm[: len(W) // 2], perm[len(W) // 2:]
    seq = {i: collapse(np.frombuffer(W[i].encode(), np.uint8).astype(np.int64) - 97)
           for i in range(len(W))}
    by: Dict[int, List[int]] = defaultdict(list)
    for i in donor_side:
        by[len(seq[i])].append(int(i))
    pools = {k: np.array(v) for k, v in by.items()}

    real, v1, retimed, words = [], [], [], []
    for i in real_side[: args.rows or len(real_side)]:
        S = len(seq[i])
        pool = pools.get(S)
        if S < 2 or pool is None or len(pool) == 0:
            continue
        di = int(pool[rng.integers(len(pool))])
        warped, j = warp_debug(F[di], np.arange(S), qw[seq[di]], qw[seq[i]])
        np.clip(warped, 0.0, 1.0, out=warped)
        real.append(F[i])
        words.append(W[i])
        v1.append(warped)
        retimed.append(np.clip(retime_segment(warped, F[di], j, 0.5), 0, 1))
        if len(words) >= 9416:
            break
    real, v1, retimed = np.stack(real), np.stack(v1), np.stack(retimed)
    print(f"en->en matched pairs: {len(words)}")

    # trace_metrics indexes letters through the module-level alphabet.
    saved, sga.RU_LETTERS_31 = sga.RU_LETTERS_31, "abcdefghijklmnopqrstuvwxyz"
    try:
        mr = trace_metrics(real, words, qw)
        res: Dict[str, object] = {"n": len(words)}
        for nm, X in (("en_v1", v1), ("en_B_seg_a05", retimed)):
            ms = trace_metrics(X, words, qw)
            res[nm + "_ks"] = {k: ks_stat(mr[k], ms[k]) for k in METRIC_KEYS}
            print(f"{nm:<14} " + "  ".join(
                f"{k}={res[nm + '_ks'][k]:.3f}" for k in
                ("step_cv", "step_max", "sharp_turns", "turn_mean")))
    finally:
        sga.RU_LETTERS_31 = saved
    for nm, X in (("en_v1", v1), ("en_B_seg_a05", retimed)):
        g = run_gate(real, X, words, args.seed)
        res[nm + "_gate"] = g
        print(f"{nm:<14} gate speed {g['speed']['max']:.4f}/{g['speed']['final']:.4f}"
              f"  coords {g['coords']['max']:.4f}/{g['coords']['final']:.4f}"
              f"  angles {g['angles']['max']:.4f}/{g['angles']['final']:.4f}")
    (out_dir / "enen.json").write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["law", "variants", "classifier", "gbm", "coupling",
                             "enen"])
    ap.add_argument("--enen-corpus", default="train_t3hws.npz",
                    help="single-corpus donor+real source for --stage enen")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--donors", default="train_t3futo.npz,train_t3hws.npz")
    ap.add_argument("--rows", type=int, default=0, help="0 = all matched rows")
    ap.add_argument("--k-cand", type=int, default=16, help="fix-C reservoir size")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    return {"law": stage_law, "variants": stage_variants,
            "classifier": stage_classifier, "gbm": stage_gbm,
            "coupling": stage_coupling, "enen": stage_enen}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
