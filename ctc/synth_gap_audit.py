#!/usr/bin/env python3
"""Quality-gap audit of the v1 residual-transplant generator against real data.

Russian is the gold validator: the Yandex Cup valid-10k (eval-only footing per
YANDEX_LICENSE_RESEARCH.md) provides real ЙЦУКЕН swipes, and `cyrillic_synth.py`
can synthesize traces for the SAME words on the SAME layout — so every
difference between the two sets is generator error, not word/layout confound.

Three stages (run in order; each writes under ``<workdir>/synth_gap/``):

``--stage data``
    Featurize the real valid-10k (default grid, `keep_reason` filter — the
    established 9,416-row footing), record RAW-trace facts the features drop
    (duration, point count, inter-sample dt), then synthesize one word-matched
    trace per real row with the v1 mechanism (full donor pool, uniform
    count-matched donor draw — `cyrillic_synth.py` verbatim semantics).

``--stage metrics``
    Per-trace kinematic/geometric metrics on both sets + KS statistics, the
    word-length-mix table (v1 train draw vs real corpus), and raw-trace stats.

``--stage classifier``
    The discriminability probe: a 2-layer MLP real-vs-synth classifier on
    word-matched, word-DISJOINT splits (the classifier must generalize across
    words, so it can only use style, not lexical shape). Feature-set ablations
    localize WHAT betrays synthesis. Also the unmatched arm (v1 training draw
    vs real corpus) which additionally sees the word/length-mix gap.

Accuracy 0.5 = indistinguishable; the gap above 0.5 IS the quality metric.

Nothing here touches Phase O's files: outputs live in ``synth_gap/`` only,
and the real-val featurization is written there, NOT to ``cache_ru/val.npz``
(regenerating that file is Phase O's registered todo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyrillic_synth import build_donor_index, collapse  # noqa: E402
from futo_decoder_eval import featurize, load_layout  # noqa: E402
from layout_aug import DWELL_RADIUS, warp_path  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_yandex import RU_LETTERS_31, iter_corpus, keep_reason  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Step length below which two consecutive resampled points count as dwell.
#: The 64 samples are time-uniform, so a step is (local speed)·(duration/63);
#: 0.004 units ≈ 1/25 of a key pitch — effectively stationary.
DWELL_STEP = 0.004

#: Moving-average window for the stroke-count proxy's speed profile.
SPEED_SMOOTH = 9

#: The original 17-metric battery, in report order.  Frozen: it is what
#: `SYNTH_V2_DESIGN.md` §1.1 and the research audit's tables quote, and
#: `synth_retime_probe.py` scores its variants on exactly this set.
METRIC_KEYS_17 = ("step_cv", "step_max", "sharp_turns", "turn_mean", "turn_total",
                  "straightness", "key_cover", "dwell_frac", "start_dwell",
                  "end_dwell", "start_d", "end_d", "path_len", "step_mean",
                  "dup_frac", "dwell_run_max", "speed_asym")

#: Phase-P additions (SYNTH_V2_RESEARCH_AUDIT §3.2).  All six exist because the
#: 17 above are marginal or aggregate and none of them looks at the *temporal*
#: structure of the speed sequence — which is exactly what an MLP over 63 step
#: lengths exploits, and exactly what fix B repairs and fix B breaks:
#:
#: * ``ac1`` — lag-1 speed autocorrelation.  KS 0.618 on v1, the largest
#:   single-statistic gap found anywhere in the campaign.
#: * ``spec_centroid`` — the frequency-domain form of the same defect (0.566).
#: * ``ldlj`` — log dimensionless jerk (0.518); gets *worse* under speed-only
#:   re-timing, so it is not redundant with ``step_cv``.
#: * ``minima_per_seg`` — the stroke-count proxy; real ≈ 0.77 per segment.
#: * ``sc_slope`` / ``sc_r2`` — the speed–curvature coupling.  Nearly blind to
#:   v1's defect (KS 0.156) and wide awake to the one re-timing *creates*
#:   (0.566 in the design's global form).  Mandatory in the battery BEFORE any
#:   re-timing lands, or fix B Goodharts every other metric here.
METRIC_KEYS_NEW = ("ac1", "spec_centroid", "ldlj", "minima_per_seg",
                   "sc_slope", "sc_r2")

METRIC_KEYS_EXT = METRIC_KEYS_17 + METRIC_KEYS_NEW


# ── stage 1: data ───────────────────────────────────────────────────────────────

def load_real_valid(valid_jsonl: Path, ref: Optional[Path] = None
                    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Real valid-10k -> ([N,2,64] features, words, raw-trace fact arrays).

    Default-grid rows passing `keep_reason` — the campaign's 9,416-row footing.
    """
    feats: List[np.ndarray] = []
    words: List[str] = []
    n_points: List[int] = []
    duration: List[float] = []
    dt_median: List[float] = []
    dt_max: List[float] = []
    for word, gname, xs, ys, ts, _grid in iter_corpus(valid_jsonl, ref):
        if gname != "default":
            continue
        p, _reason = keep_reason(word, xs, ts)
        if p is None:
            continue
        feats.append(featurize(xs, ys, ts))
        words.append(p)
        n_points.append(len(xs))
        duration.append(ts[-1])
        dts = np.diff(np.asarray(ts, np.float64))
        dt_median.append(float(np.median(dts)) if len(dts) else 0.0)
        dt_max.append(float(dts.max()) if len(dts) else 0.0)
    raw = {"n_points": np.asarray(n_points, np.int64),
           "duration_ms": np.asarray(duration, np.float64),
           "dt_median_ms": np.asarray(dt_median, np.float64),
           "dt_max_ms": np.asarray(dt_max, np.float64)}
    return np.stack(feats), words, raw


def synth_matched(words: Sequence[str], donor_feats: np.ndarray,
                  donor_seqs: List[np.ndarray],
                  donor_by_count: Dict[int, np.ndarray],
                  qwerty: np.ndarray, ru_centers: np.ndarray,
                  seed: int = 20260819) -> Tuple[np.ndarray, np.ndarray]:
    """One v1-mechanism synthetic trace per word -> ([N,2,64], ok mask).

    Identical donor policy to `cyrillic_synth.synthesize`: uniform draw from
    the vertex-count-matched pool, full donor set, i.i.d. per trace.
    """
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    out = np.zeros((len(words), 2, 64), np.float32)
    ok = np.zeros(len(words), bool)
    for i, w in enumerate(words):
        seq = collapse(np.array([idx[c] for c in w], np.int64))
        pool = donor_by_count.get(len(seq))
        if pool is None or len(pool) == 0:
            continue
        di = int(pool[rng.integers(len(pool))])
        S = len(seq)
        warped = warp_path(donor_feats[di], np.arange(S, dtype=np.int64),
                           qwerty[donor_seqs[di]], ru_centers[seq])
        np.clip(warped, 0.0, 1.0, out=warped)
        out[i] = warped
        ok[i] = True
    return out, ok


def stage_data(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = resolve(args.workdir, Path("data/yandex_cup/valid.jsonl"))
    ref = resolve(args.workdir, Path("data/yandex_cup/valid.ref"))
    t0 = time.time()
    real_feats, words, raw = load_real_valid(valid, ref if ref.exists() else None)
    print(f"real: {len(words)} rows ({time.time() - t0:.0f}s)")

    cache = resolve(args.workdir, Path("cache"))
    donor_paths = [cache / p.strip() for p in args.donors.split(",")]
    donor_feats, donor_seqs, by_count = build_donor_index(donor_paths)
    print(f"donors: {len(donor_feats)} traces")

    _, qwerty = load_layout(HERE / "en_qwerty.json")
    ru_letters, ru_centers = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    assert "".join(ru_letters) == RU_LETTERS_31

    synth_feats, ok = synth_matched(words, donor_feats, donor_seqs, by_count,
                                    qwerty, ru_centers)
    print(f"synth matched: {int(ok.sum())}/{len(ok)} rows")
    np.savez_compressed(
        out_dir / "matched.npz", real=real_feats[ok], synth=synth_feats[ok],
        words=np.array([w for w, k in zip(words, ok) if k]),
        **{f"raw_{k}": v[ok] for k, v in raw.items()})
    print(f"-> {out_dir / 'matched.npz'}")
    return 0


# ── stage 2: metrics ────────────────────────────────────────────────────────────

def speed_curvature(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trace (slope, R²) of log speed on log curvature; NaN where undefined.

    Menger curvature (parameterization-invariant, so time-uniform samples are
    fine): ``κ = 4·Area/(|ab|·|bc|·|ca|)``; speed at the middle sample is the
    mean of its two adjacent step lengths.  The two-thirds power law predicts
    slope −1/3.  What matters is not the absolute value — discrete-curvature
    estimators are biased, per Schaal & Sternad 2001, and a 4th-order
    Butterworth manufactures β = 0.327 out of a β = 0 trajectory — but the
    comparison between real and synthetic populations computed identically.
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
    return slope, r2


def _log_dimensionless_jerk(d: np.ndarray) -> float:
    """LDLJ of a speed profile, the ISC-licensed reference formula.

    ``-log( (T³/v_peak²) · Σ (d²v/dt²)² · dt )`` with the profile's own duration
    normalized to 1 (our samples are time-uniform and absolute duration is not
    in the features).  Exactly invariant to that normalization *and* to a global
    speed scale, which is why it survives our two normalizations unchanged.
    Note the SPARC paper's Eq. 2 prints a duration exponent of 5, which is
    dimensionally wrong; the reference implementation uses 3, and so does this.
    """
    n = len(d)
    if n < 4:
        return float("nan")
    peak = float(np.max(np.abs(d)))
    if peak <= 1e-12:
        return float("nan")
    dt = 1.0 / n
    jerk = np.diff(d, 2) / (dt * dt)
    tot = float((jerk ** 2).sum()) * dt
    if tot <= 0:
        return float("nan")
    return float(-np.log((1.0 ** 3 / peak ** 2) * tot))


def trace_metrics(feats: np.ndarray, words: Sequence[str],
                  centers: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-trace kinematic/geometric metrics on [N,2,64] features.

    The 64 samples are time-uniform, so step length is speed in units of
    (board units per 1/63 of trace duration) — shape comparisons are valid
    even though absolute duration is not in the features.
    """
    idx = {c: i for i, c in enumerate(RU_LETTERS_31)}
    N = feats.shape[0]
    m: Dict[str, List[float]] = {k: [] for k in (
        "path_len", "step_mean", "step_cv", "step_max", "dwell_frac",
        "dwell_run_max", "start_dwell", "end_dwell", "start_d", "end_d",
        "turn_mean", "turn_total", "sharp_turns", "straightness",
        "speed_asym", "dup_frac", "key_cover",
        "ac1", "spec_centroid", "ldlj", "minima_per_seg",
        "n_segments", "minima")}
    for n in range(N):
        P = feats[n].T.astype(np.float64)                    # [64,2]
        w = words[n]
        seq = collapse(np.array([idx[c] for c in w], np.int64))
        kpts = centers[seq].astype(np.float64)
        steps = np.diff(P, axis=0)                           # [63,2]
        d = np.hypot(steps[:, 0], steps[:, 1])
        plen = float(d.sum())
        m["path_len"].append(plen)
        m["step_mean"].append(float(d.mean()))
        m["step_cv"].append(float(d.std() / max(d.mean(), 1e-9)))
        m["step_max"].append(float(d.max()))
        dw = d < DWELL_STEP
        m["dwell_frac"].append(float(dw.mean()))
        run = best = 0
        for b in dw:
            run = run + 1 if b else 0
            best = max(best, run)
        m["dwell_run_max"].append(float(best))
        # Leading/trailing samples within DWELL_RADIUS of the first/last key.
        d0 = np.hypot(*(P - kpts[0]).T)
        d1 = np.hypot(*(P - kpts[-1]).T)
        k0 = 0
        while k0 < 64 and d0[k0] <= DWELL_RADIUS:
            k0 += 1
        k1 = 0
        while k1 < 64 and d1[63 - k1] <= DWELL_RADIUS:
            k1 += 1
        m["start_dwell"].append(float(k0))
        m["end_dwell"].append(float(k1))
        m["start_d"].append(float(d0[0]))
        m["end_d"].append(float(d1[-1]))
        nz = d > 1e-9
        u = steps[nz] / d[nz][:, None]
        if len(u) >= 2:
            cross = u[:-1, 0] * u[1:, 1] - u[:-1, 1] * u[1:, 0]
            dot = (u[:-1] * u[1:]).sum(1)
            ang = np.abs(np.arctan2(cross, np.clip(dot, -1, 1)))
            m["turn_mean"].append(float(ang.mean()))
            m["turn_total"].append(float(ang.sum()))
            m["sharp_turns"].append(float((ang > np.pi / 3).sum()))
        else:
            m["turn_mean"].append(0.0)
            m["turn_total"].append(0.0)
            m["sharp_turns"].append(0.0)
        ideal = float(np.hypot(*np.diff(kpts, axis=0).T).sum()) if len(kpts) > 1 else 0.0
        m["straightness"].append(plen / max(ideal, 1e-9) if ideal > 0 else 1.0)
        m["speed_asym"].append(float(d[:16].mean() / max(d[-16:].mean(), 1e-9)))
        m["dup_frac"].append(float((d < 1e-9).mean()))
        # Fraction of the word's distinct keys the path passes within a key
        # half-width of (transit fidelity beyond the endpoints).
        cov = 0
        for kp in kpts:
            if np.hypot(*(P - kp).T).min() <= DWELL_RADIUS:
                cov += 1
        m["key_cover"].append(cov / len(kpts))

        # ── Phase-P: the temporal structure of the speed sequence ──────────
        dc = d - d.mean()
        den = float((dc * dc).sum())
        m["ac1"].append(float((dc[:-1] * dc[1:]).sum() / den) if den > 1e-18
                        else 0.0)
        mag = np.abs(np.fft.rfft(dc))
        fr = np.fft.rfftfreq(len(d), d=1.0 / len(d))
        m["spec_centroid"].append(float((mag * fr).sum() / max(mag.sum(), 1e-18)))
        m["ldlj"].append(_log_dimensionless_jerk(d))
        # Local minima of the SMOOTHED speed profile = the stroke-count proxy.
        # The 9-sample moving average is not cosmetic: on the raw 63-step
        # sequence every sampling wiggle counts as a stroke (real traces read
        # 8.99 "strokes" for a 3.7-segment word), and at window 9 the real
        # corpus lands on 2.87 strokes per trace — the 2.88 the research audit
        # measured with its own estimator, so the pre-registered
        # 0.77-per-segment bar keeps its meaning.
        sm = np.convolve(d, np.ones(SPEED_SMOOTH) / SPEED_SMOOTH, mode="same")
        mins = int(((sm[1:-1] < sm[:-2]) & (sm[1:-1] <= sm[2:])).sum())
        m["minima_per_seg"].append(mins / max(len(kpts) - 1, 1))
        m["n_segments"].append(float(max(len(kpts) - 1, 1)))
        m["minima"].append(float(mins))
    out = {k: np.asarray(v, np.float64) for k, v in m.items()}
    out["sc_slope"], out["sc_r2"] = speed_curvature(feats)
    return out


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no scipy dependency).

    NaNs are dropped per side: the speed–curvature fit is undefined for traces
    with fewer than 8 usable curvature samples, and dropping those rows from
    both marginals is the same convention `synth_retime_probe.stage_coupling`
    used to produce the audit's coupling table.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    allv.sort(kind="mergesort")
    cdf_a = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def stage_metrics(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    with np.load(out_dir / "matched.npz", allow_pickle=False) as d:
        real, synth = np.array(d["real"]), np.array(d["synth"])
        words = [str(w) for w in d["words"]]
        raw = {k[4:]: np.array(d[k]) for k in d.files if k.startswith("raw_")}
    _, centers = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    mr = trace_metrics(real, words, centers)
    ms = trace_metrics(synth, words, centers)

    report: Dict[str, object] = {"n": len(words)}
    print(f"\nword-matched metric comparison, n={len(words)} "
          f"(real Yandex valid vs v1 synth, same words, ru_jcuken_default)")
    hdr = (f"{'metric':<14}{'real mean':>10}{'synth mean':>11}{'real p50':>10}"
           f"{'synth p50':>10}{'KS':>7}")
    print(hdr)
    print("-" * len(hdr))
    tbl = {}
    for k in mr:
        row = {"real_mean": float(mr[k].mean()), "synth_mean": float(ms[k].mean()),
               "real_p50": float(np.median(mr[k])),
               "synth_p50": float(np.median(ms[k])),
               "ks": ks_stat(mr[k], ms[k])}
        tbl[k] = row
        print(f"{k:<14}{row['real_mean']:>10.4f}{row['synth_mean']:>11.4f}"
              f"{row['real_p50']:>10.4f}{row['synth_p50']:>10.4f}{row['ks']:>7.3f}")
    report["metrics"] = tbl

    # Raw-trace facts (real only — the generator has no time axis at all).
    print("\nraw real-trace facts (features drop these; the generator never had them)")
    rawtbl = {}
    for k, v in raw.items():
        q = np.percentile(v, [1, 25, 50, 75, 99])
        rawtbl[k] = {"p1": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                     "p75": float(q[3]), "p99": float(q[4])}
        print(f"  {k:<14} p1 {q[0]:.1f}  p25 {q[1]:.1f}  p50 {q[2]:.1f}  "
              f"p75 {q[3]:.1f}  p99 {q[4]:.1f}")
    report["raw_real"] = rawtbl

    # Word-length mix: the v1 TRAINING draw (CKDT 255-rank weights) vs real usage.
    synth_train = resolve(args.workdir, Path("cache_ru_synth/train_synth.npz"))
    if synth_train.exists():
        with np.load(synth_train) as d:
            tw = [str(w) for w in d["words"][:200_000]]
        mix = {}
        for name, ws in (("real_valid", words), ("v1_train_draw", tw)):
            L = np.array([len(w) for w in ws])
            mix[name] = {"le3": float((L <= 3).mean()),
                         "4to6": float(((L >= 4) & (L <= 6)).mean()),
                         "ge7": float((L >= 7).mean()),
                         "mean_len": float(L.mean())}
        report["length_mix"] = mix
        print("\nword-length mix (fraction of rows)")
        print(f"{'set':<14}{'<=3':>8}{'4-6':>8}{'>=7':>8}{'mean len':>10}")
        for name, r in mix.items():
            print(f"{name:<14}{r['le3']:>8.3f}{r['4to6']:>8.3f}{r['ge7']:>8.3f}"
                  f"{r['mean_len']:>10.2f}")

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=1))
    print(f"\n-> {out_dir / 'metrics.json'}")
    return 0


# ── stage 3: classifier ─────────────────────────────────────────────────────────

def feature_views(feats: np.ndarray) -> Dict[str, np.ndarray]:
    """Ablation views of [N,2,64] for the discriminability probe."""
    P = feats.transpose(0, 2, 1).astype(np.float32)          # [N,64,2]
    steps = np.diff(P, axis=1)                               # [N,63,2]
    d = np.hypot(steps[..., 0], steps[..., 1])               # [N,63]
    nz = np.maximum(d, 1e-9)[..., None]
    u = steps / nz
    cross = u[:, :-1, 0] * u[:, 1:, 1] - u[:, :-1, 1] * u[:, 1:, 0]
    dot = (u[:, :-1] * u[:, 1:]).sum(-1)
    ang = np.arctan2(cross, np.clip(dot, -1, 1)).astype(np.float32)  # [N,62]
    return {
        "coords": feats.reshape(len(feats), -1),             # 128
        "speed": d.astype(np.float32),                       # 63
        "angles": ang,                                       # 62
        "endpoints": np.concatenate([P[:, :4].reshape(len(P), -1),
                                     P[:, -4:].reshape(len(P), -1)], 1),  # 16
        "speed+angles": np.concatenate([d.astype(np.float32), ang], 1),   # 125
    }


def train_mlp(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray,
              y_te: np.ndarray, seed: int = 0, epochs: int = 40) -> float:
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
    n = len(xt)
    best = 0.0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(xt[b]), yt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(xv).argmax(1) == yv).float().mean())
        best = max(best, acc)
    return best


def stage_classifier(args: argparse.Namespace) -> int:
    out_dir = resolve(args.workdir, Path("synth_gap"))
    with np.load(out_dir / "matched.npz", allow_pickle=False) as d:
        real, synth = np.array(d["real"]), np.array(d["synth"])
        words = [str(w) for w in d["words"]]
    # Word-disjoint split: the classifier must transfer across words, so it can
    # only use style, never lexical shape (both classes share the word multiset).
    rng = np.random.default_rng(args.seed)
    uniq = sorted(set(words))
    test_words = set(np.array(uniq)[rng.permutation(len(uniq))[:len(uniq) // 5]])
    te = np.array([w in test_words for w in words])
    tr = ~te

    x_all = np.concatenate([real, synth])
    y_all = np.concatenate([np.zeros(len(real), np.int64),
                            np.ones(len(synth), np.int64)])
    m_tr = np.concatenate([tr, tr])
    m_te = np.concatenate([te, te])

    report: Dict[str, object] = {"n_pairs": len(words),
                                 "n_test_pairs": int(te.sum())}
    print(f"real-vs-synth MLP, word-matched + word-disjoint split "
          f"({int(tr.sum())} train / {int(te.sum())} test pairs)")
    views = feature_views(x_all)
    for name, X in views.items():
        acc = train_mlp(X[m_tr], y_all[m_tr], X[m_te], y_all[m_te],
                        seed=args.seed)
        report[f"acc_{name}"] = acc
        print(f"  {name:<14} dim {X.shape[1]:>4}  test acc {acc:.4f}")

    # Unmatched arm: the v1 TRAINING draw vs the real corpus — adds the
    # word/length-mix gap to whatever the matched arm sees.
    synth_train = resolve(args.workdir, Path("cache_ru_synth/train_synth.npz"))
    if synth_train.exists():
        with np.load(synth_train) as d:
            sf = np.array(d["features"][:len(real)])
        x2 = np.concatenate([real, sf]).reshape(len(real) + len(sf), -1)
        y2 = np.concatenate([np.zeros(len(real), np.int64),
                             np.ones(len(sf), np.int64)])
        perm = rng.permutation(len(x2))
        cut = int(0.8 * len(x2))
        acc = train_mlp(x2[perm[:cut]], y2[perm[:cut]],
                        x2[perm[cut:]], y2[perm[cut:]], seed=args.seed)
        report["acc_unmatched_coords"] = acc
        print(f"  unmatched (v1 train draw vs real), coords: test acc {acc:.4f}")

    (out_dir / "classifier.json").write_text(json.dumps(report, indent=1))
    print(f"-> {out_dir / 'classifier.json'}")
    return 0


# ── Phase P: the amended gate battery ───────────────────────────────────────────
#
# Seven defects of the committed classifier gate were measured by
# SYNTH_V2_RESEARCH_AUDIT §3.1 and every one of them is repaired here:
#
#  1. it reported max-over-epochs test accuracy — a selection on the test set
#     worth +1.2 pts under a true H0.  -> final-epoch accuracy only.
#  2. it had no floor arm.  -> a word-matched real-vs-real arm runs on every
#     gate call and the run is VOID if it lands outside [0.48, 0.52].
#  3. the textbook null is wrong for matched pairs (13-22 % too wide).
#     -> an exact within-pair permutation null.
#  4. a single arbitrary word split (accuracy sd 0.016-0.029, 3x binomial).
#     -> K-fold word-disjoint CV, every pair tested once, with a one-sided
#     95 % upper confidence limit from the fold spread.
#  5. an 80/20 split where balanced is power-optimal.  -> K-fold dominates both.
#  6. the MLP is too weak, which INVERTS the gate's logic: a C2ST accuracy is a
#     lower bound on (1+TV)/2, so a low number from a weak critic certifies
#     nothing (Arora et al. ICML 2017 Cor. 3.2 — and more data cannot fix it).
#     -> the bar is the MAX over a pre-registered classifier x view family, and
#     the strongest member (HistGBM on the metric battery) is reported by name.
#  7. `acc_unmatched_coords` split by random ROW across classes with different
#     word distributions and read 0.732 from word memorisation alone.
#     -> word-disjoint, and relabelled for what it now measures.

#: One-sided 95 % t quantiles by degrees of freedom (K-1), for the UCL.
_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833}

#: The pre-registered classifier x view family the G4 bar maximises over.
GATE_VIEWS = ("speed", "coords", "angles")


def ucl95(folds: Sequence[float]) -> Dict[str, float]:
    """Fold mean and its one-sided 95 % upper confidence limit."""
    a = np.asarray(folds, np.float64)
    k = len(a)
    sd = float(a.std(ddof=1)) if k > 1 else 0.0
    t = _T95.get(k - 1, 1.645)
    return {"mean": float(a.mean()), "sd": sd,
            "ucl95": float(a.mean() + t * sd / max(np.sqrt(k), 1.0)), "folds": k}


def kfold_word_folds(words: Sequence[str], k: int, seed: int
                     ) -> List[np.ndarray]:
    """K word-disjoint test masks over the pairs; every pair is tested once."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(words)))
    assign = {w: int(i % k) for i, w in
              enumerate(uniq[rng.permutation(len(uniq))])}
    fold_of = np.array([assign[w] for w in words])
    return [fold_of == f for f in range(k)]


def train_mlp_final(x_tr, y_tr, x_te, y_te, seed: int = 0,
                    epochs: int = 40) -> float:
    """The committed §1.4 architecture, reporting the FINAL-epoch accuracy.

    Same net, same optimiser, same schedule as :func:`train_mlp` — only the
    reported statistic changes, because the max over 40 epochs is a selection on
    the very set the gate is read from.
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
    acc = 0.0
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
    return acc


def gate_mlp_kfold(a: np.ndarray, b: np.ndarray, words: Sequence[str],
                   k: int, seed: int, views: Sequence[str] = GATE_VIEWS
                   ) -> Dict[str, Dict[str, float]]:
    """K-fold word-disjoint MLP gate over the pre-registered feature views."""
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a), np.int64), np.ones(len(b), np.int64)])
    v = feature_views(x)
    folds = kfold_word_folds(words, k, seed)
    out: Dict[str, Dict[str, float]] = {}
    for name in views:
        accs = []
        for f, te in enumerate(folds):
            m_te = np.concatenate([te, te])
            m_tr = ~m_te
            accs.append(train_mlp_final(v[name][m_tr], y[m_tr],
                                        v[name][m_te], y[m_te], seed=seed + f))
        out[name] = ucl95(accs)
    return out


def metric_matrix(feats: np.ndarray, words: Sequence[str], centers: np.ndarray,
                  keys: Sequence[str] = METRIC_KEYS_EXT) -> np.ndarray:
    """``[N, len(keys)]`` interpretable-metric design matrix (NaN allowed)."""
    m = trace_metrics(feats, words, centers)
    return np.stack([m[k] for k in keys], 1)


def gate_gbm_kfold(Xa: np.ndarray, Xb: np.ndarray, words: Sequence[str],
                   k: int, seed: int, importances: bool = True,
                   keys: Sequence[str] = METRIC_KEYS_EXT) -> Dict[str, object]:
    """K-fold word-disjoint HistGBM gate on the interpretable metric battery.

    The strongest practical critic in the family, which is what a gate needs:
    requiring a *weak* classifier to fail proves nothing about the distributions.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
    folds = kfold_word_folds(words, k, seed)
    accs: List[float] = []
    imp = np.zeros(X.shape[1])
    for f, te in enumerate(folds):
        m_te = np.concatenate([te, te])
        m_tr = ~m_te
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             random_state=seed + f)
        clf.fit(X[m_tr], y[m_tr])
        accs.append(float(clf.score(X[m_te], y[m_te])))
        if importances and f == 0:
            pi = permutation_importance(clf, X[m_te], y[m_te], n_repeats=3,
                                        random_state=seed, n_jobs=4)
            imp = pi.importances_mean
    res: Dict[str, object] = dict(ucl95(accs))
    if importances:
        res["top_features"] = [(keys[i], round(float(imp[i]), 4))
                               for i in np.argsort(-imp)[:5]]
    return res


def permutation_null_gbm(Xa: np.ndarray, Xb: np.ndarray, words: Sequence[str],
                         k: int, seed: int, n_perm: int) -> Dict[str, object]:
    """Exact within-pair permutation null for the GBM gate's fold-mean.

    Matched pairs are dependent — the per-pair variance under H0 is 0.077 where
    an independent-rows model implies 0.125, so the textbook ``N(½, 1/4n)`` is
    13–22 % too wide and therefore too easy to pass.  Swapping the two labels of
    a pair with probability ½ and rerunning the whole procedure is the null the
    data actually has.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.concatenate([Xa, Xb])
    n = len(Xa)
    folds = kfold_word_folds(words, k, seed)
    rng = np.random.default_rng(seed + 99)
    stats: List[float] = []
    for p in range(n_perm):
        swap = rng.random(n) < 0.5
        y = np.concatenate([swap.astype(int), (~swap).astype(int)])
        accs = []
        for f, te in enumerate(folds):
            m_te = np.concatenate([te, te])
            m_tr = ~m_te
            clf = HistGradientBoostingClassifier(max_iter=300,
                                                 learning_rate=0.08,
                                                 random_state=seed + f)
            clf.fit(X[m_tr], y[m_tr])
            accs.append(float(clf.score(X[m_te], y[m_te])))
        stats.append(float(np.mean(accs)))
    s = np.array(stats)
    return {"n_perm": n_perm, "null_mean": float(s.mean()),
            "null_p95": float(np.percentile(s, 95)), "null_max": float(s.max())}


def real_pairs(real: np.ndarray, words: Sequence[str], seed: int = 7,
               dense: bool = True):
    """Word-matched real-vs-real pairs — the empirical floor of the instrument.

    *dense* pairs off ⌊v/2⌋ pairs for every word with v ≥ 2 real traces (the
    better-powered arm, 3,194 pairs on the ru probe); otherwise one pair per
    such word (923).
    """
    by: Dict[str, List[int]] = {}
    for i, w in enumerate(words):
        by.setdefault(w, []).append(i)
    rng = np.random.default_rng(seed)
    ia: List[int] = []
    ib: List[int] = []
    ws: List[str] = []
    for w, idxs in by.items():
        if len(idxs) < 2:
            continue
        p = rng.permutation(idxs)
        take = len(p) // 2 if dense else 1
        for t in range(take):
            ia.append(int(p[2 * t]))
            ib.append(int(p[2 * t + 1]))
            ws.append(w)
    return np.array(ia), np.array(ib), ws


def gap_closure(acc_v1: float, acc_v2: float, floor: float = 0.50) -> float:
    """``(acc_v1 − acc_v2) / (acc_v1 − floor)`` — the progress meter of §3.4.

    The floor is measured (real-vs-real is 0.50), not assumed, and the
    instrument must be named alongside the number: A+B+C closes 39 % of the MLP
    speed-view gap and only 15 % of the GBM gap on the same data.
    """
    den = acc_v1 - floor
    return float((acc_v1 - acc_v2) / den) if den > 1e-9 else float("nan")


def stage_v2(args: argparse.Namespace) -> int:
    """Build the word-matched v2 arms with the **shipped** generator.

    Words come from the real Yandex partners, so fix A (the word draw) is out of
    scope here by construction — this stage measures the *mechanism* stages C,
    B′ and S5.  Fix A is gated separately by G2 on the training draw, which is
    the only place a draw policy can be measured at all.
    """
    import script_synth as SS
    out_dir = resolve(args.workdir, Path("synth_gap"))
    with np.load(out_dir / "matched.npz", allow_pickle=False) as d:
        real, v1 = np.array(d["real"]), np.array(d["synth"])
        words = [str(w) for w in d["words"]]
        raw_dur = np.array(d["raw_duration_ms"])
    n = len(words) if args.rows <= 0 else min(args.rows, len(words))
    real, v1, words, raw_dur = real[:n], v1[:n], words[:n], raw_dur[:n]

    cache = resolve(args.workdir, Path("cache"))
    _, qwerty = load_layout(HERE / "en_qwerty.json")
    letters, centers = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    assert "".join(letters) == RU_LETTERS_31
    bank = SS.build_donor_index([cache / p.strip() for p in args.donors.split(",")],
                                "all", qwerty=qwerty)
    bank.dur_exponent, dur_fit = SS.fit_duration_law(bank)
    print(f"donor bank: {len(bank)} traces; S5 duration law T ∝ L^"
          f"{bank.dur_exponent:.3f} (R² {dur_fit['r2']:.3f})", flush=True)

    arms = {"v1_repro": "", "C": "c", "C_B": "c,b", "C_B_S5": "c,b,s5",
            "B": "b", "S5": "s5"}
    var: Dict[str, np.ndarray] = {}
    durs: Dict[str, np.ndarray] = {}
    for name, mask in arms.items():
        rng = np.random.default_rng(20260819)   # v1's donor-draw seed, verbatim
        f, ok, st, rows = SS.synthesize_matched(
            words, rng, RU_LETTERS_31, bank, qwerty, centers,
            SS.GenOpts.parse(mask))
        assert ok.all(), f"{name}: {int((~ok).sum())} rows had no donor"
        var[name] = f
        durs[name] = rows["drawn_duration_ms"]
        print(f"  {name:<9} mask {mask or 'v1':<9} {st}", flush=True)
    delta = float(np.abs(var["v1_repro"] - v1).max())
    print(f"v1 reproduction max|Δ| vs matched.npz: {delta:.2e}")
    assert delta == 0.0, ("the shipped generator's v1 path is not the mechanism "
                          "matched.npz was built with — the gate is invalid")
    d5 = durs["C_B_S5"]
    m = d5 > 0
    print(f"S5 drawn durations vs the real partners' (n={int(m.sum())}): "
          f"synth p25/p50/p75 {np.percentile(d5[m], [25, 50, 75]).round(0)} vs "
          f"real {np.percentile(raw_dur[m], [25, 50, 75]).round(0)}  "
          f"KS {ks_stat(d5[m], raw_dur[m]):.3f}")
    np.savez_compressed(out_dir / "matched_v2.npz", real=real,
                        words=np.array(words), raw_duration_ms=raw_dur,
                        drawn_duration_ms=d5, **var)
    (out_dir / "matched_v2_meta.json").write_text(json.dumps(
        {"n": n, "arms": arms, "duration_law": dur_fit,
         "s5_duration_ks_vs_real": ks_stat(d5[m], raw_dur[m])}, indent=1))
    print(f"-> {out_dir / 'matched_v2.npz'}")
    return 0


#: G3's amended bars (SYNTH_V2_RESEARCH_AUDIT §4.3), as ``(metric, ceiling)``.
#: `sharp_turns` is 0.32 rather than the design's 0.25 because 0.25 is
#: unreachable even with an oracle that reads the validator's own durations
#: (0.288); the coupling bars are NEW because re-timing *creates* that defect.
G3_BARS = (("step_cv", 0.15), ("step_max", 0.12), ("sharp_turns", 0.32),
           ("ac1", 0.12), ("sc_slope", 0.35), ("sc_r2", 0.40))

#: G4's amended bars: gap-closure fractions against the measured 0.50 floor,
#: with the instrument named.  UCL95 <= 0.60 is recorded as an OPEN SHORTFALL,
#: not a pass condition — an English-donor transplant cannot be made
#: statistically indistinguishable from real Russian swipes, and ~0.15 of the
#: residual separability is donor-population mismatch no generator change can
#: touch (the en->en control prices it).
G4_BARS = {"mlp_speed_closure": 0.35, "gbm_closure": 0.20, "enen_closure": 0.65}
G4_STANDARD_UCL = 0.60


def _pass(ok: bool) -> str:
    return "PASS" if ok else "MISS"


def stage_gates(args: argparse.Namespace) -> int:
    """The amended G1–G4 battery on ru, against the pre-registered bars."""
    out_dir = resolve(args.workdir, Path("synth_gap"))
    d = np.load(out_dir / "matched_v2.npz", allow_pickle=False)
    real = np.array(d["real"])
    words = [str(w) for w in d["words"]]
    arms = [k for k in d.files
            if k not in ("real", "words", "raw_duration_ms", "drawn_duration_ms")]
    letters, centers = load_layout(HERE / "layouts" / "ru_jcuken_default.json")
    rep: Dict[str, object] = {"n_pairs": len(words), "arms": arms,
                              "folds": args.folds, "seed": args.seed}

    # ── G1 endpoint band + wrong-geometry falsification control ────────────
    import script_synth as SS
    ctrl_geo = SS.wrong_geometry(RU_LETTERS_31, np.random.default_rng(4242))
    g1: Dict[str, object] = {}
    for nm in ["real"] + arms:
        F = real if nm == "real" else np.array(d[nm])
        g1[nm] = SS.endpoint_stats(F, words, RU_LETTERS_31, centers)
        if nm != "real":
            g1[nm + "_wrong_geo"] = SS.endpoint_stats(F, words, RU_LETTERS_31,
                                                      ctrl_geo)
    v2ep = g1["C_B_S5"]
    g1_ok = (g1["C_B_S5_wrong_geo"]["start_hit"] < 0.05 and
             abs(v2ep["start_hit"] - g1["v1_repro"]["start_hit"]) < 0.05)
    rep["G1"] = {"stats": g1, "pass": g1_ok}
    print(f"\nG1 endpoints  v1 {g1['v1_repro']['start_hit']:.4f}/"
          f"{g1['v1_repro']['end_hit']:.4f}  v2 {v2ep['start_hit']:.4f}/"
          f"{v2ep['end_hit']:.4f}  real {g1['real']['start_hit']:.4f}/"
          f"{g1['real']['end_hit']:.4f}  wrong-geo "
          f"{g1['C_B_S5_wrong_geo']['start_hit']:.4f}   {_pass(g1_ok)}")

    # ── G2 length mix of the TRAINING draw vs the wordfreq token mass ──────
    g2 = length_mix_gate(args, words)
    rep["G2"] = g2
    print(f"\nG2 length mix (bar: within ±3 pts of the wordfreq mass per bucket)")
    print(f"{'set':<22}{'<=3':>8}{'4-6':>8}{'>=7':>8}{'mean len':>10}")
    for k in ("wordfreq_mass", "v2_train_draw", "v1_train_draw", "real_valid"):
        if k in g2["mix"]:
            r = g2["mix"][k]
            print(f"{k:<22}{r['le3']:>8.3f}{r['4to6']:>8.3f}{r['ge7']:>8.3f}"
                  f"{r['mean_len']:>10.2f}")
    print(f"  max bucket deviation from the wordfreq mass: "
          f"{g2['max_dev']:.3f}   {_pass(g2['pass'])}")
    print(f"  register residual, recorded not closed: wordfreq ≤3 "
          f"{g2['mix']['wordfreq_mass']['le3']:.3f} vs real usage "
          f"{g2['mix']['real_valid']['le3']:.3f} "
          f"({g2['register_residual_pts']:+.1f} pts)")

    # ── G3 kinematic parity, word-matched ──────────────────────────────────
    mr = trace_metrics(real, words, centers)
    ks: Dict[str, Dict[str, float]] = {}
    means: Dict[str, Dict[str, float]] = {"real": {k: float(np.nanmean(mr[k]))
                                                   for k in METRIC_KEYS_EXT}}
    agg_min = {"real": float(mr["minima"].mean() / mr["n_segments"].mean())}
    for nm in arms:
        ms = trace_metrics(np.array(d[nm]), words, centers)
        ks[nm] = {k: ks_stat(mr[k], ms[k]) for k in METRIC_KEYS_EXT}
        means[nm] = {k: float(np.nanmean(ms[k])) for k in METRIC_KEYS_EXT}
        agg_min[nm] = float(ms["minima"].mean() / ms["n_segments"].mean())
    shown = ("step_cv", "step_max", "sharp_turns", "turn_mean", "ac1",
             "spec_centroid", "ldlj", "minima_per_seg", "sc_slope", "sc_r2")
    print(f"\nG3 kinematic parity — KS vs real, n={len(words)}")
    print(f"{'arm':<10}" + "".join(f"{k[:9]:>11}" for k in shown) + f"{'min/seg':>9}")
    print(f"{'REAL':<10}" + "".join(f"{'-':>11}" for _ in shown)
          + f"{agg_min['real']:>9.2f}")
    for nm in arms:
        print(f"{nm:<10}" + "".join(f"{ks[nm][k]:>11.3f}" for k in shown)
              + f"{agg_min[nm]:>9.2f}")
    g3_rows = [(k, ks["C_B_S5"][k], bar, ks["C_B_S5"][k] < bar)
               for k, bar in G3_BARS]
    min_ok = abs(agg_min["C_B_S5"] - agg_min["real"]) <= 0.10
    g3_ok = all(r[3] for r in g3_rows) and min_ok
    rep["G3"] = {"ks": ks, "means": means, "minima_per_segment": agg_min,
                 "bars": [{"metric": k, "value": v, "bar": b, "pass": p}
                          for k, v, b, p in g3_rows],
                 "minima_pass": min_ok, "pass": g3_ok}
    for k, v, b, p in g3_rows:
        print(f"  {k:<14} {v:.3f} < {b:.2f}   {_pass(p)}")
    print(f"  {'minima/seg':<14} {agg_min['C_B_S5']:.2f} vs real "
          f"{agg_min['real']:.2f} (±0.10)   {_pass(min_ok)}")

    # ── G4 discriminability, both instruments, with the floor + null arms ──
    print(f"\nG4 discriminability — {args.folds}-fold word-disjoint, "
          f"final-epoch, max over the pre-registered family")
    # Two GBM instruments.  The bar was pre-registered against the audit's
    # 17-metric gate (§3.1b), so that is the one the PASS/MISS is read from; the
    # extended battery is a strictly stronger critic and is reported next to it,
    # because a gate must never be *weakened* by the reporting choice.
    i17 = [METRIC_KEYS_EXT.index(k) for k in METRIC_KEYS_17]
    Xr = metric_matrix(real, words, centers)
    ia, ib, ws = real_pairs(real, words)
    floor_mlp = gate_mlp_kfold(real[ia], real[ib], ws, args.folds, args.seed)
    floor_gbm = gate_gbm_kfold(Xr[ia][:, i17], Xr[ib][:, i17], ws, args.folds,
                               args.seed, importances=False, keys=METRIC_KEYS_17)
    floor_gbm_ext = gate_gbm_kfold(Xr[ia], Xr[ib], ws, args.folds, args.seed,
                                   importances=False)
    fl_speed, fl_gbm = floor_mlp["speed"]["mean"], floor_gbm["mean"]
    valid = 0.48 <= fl_speed <= 0.52 and 0.48 <= fl_gbm <= 0.52
    print(f"  floor arm ({len(ws)} real-vs-real pairs): MLP speed "
          f"{fl_speed:.4f}, GBM {fl_gbm:.4f} — validity "
          f"{'OK' if valid else 'VOID'}")
    g4: Dict[str, object] = {"floor": {"n_pairs": len(ws), "mlp": floor_mlp,
                                       "gbm17": floor_gbm,
                                       "gbm_ext": floor_gbm_ext,
                                       "valid": valid}}
    Xarm: Dict[str, np.ndarray] = {}
    for nm in arms:
        F = np.array(d[nm])
        Xarm[nm] = metric_matrix(F, words, centers)
        mlp = gate_mlp_kfold(F, real, words, args.folds, args.seed)
        gbm = gate_gbm_kfold(Xarm[nm][:, i17], Xr[:, i17], words, args.folds,
                             args.seed, keys=METRIC_KEYS_17)
        gbm_ext = gate_gbm_kfold(Xarm[nm], Xr, words, args.folds, args.seed)
        g4[nm] = {"mlp": mlp, "gbm17": gbm, "gbm_ext": gbm_ext,
                  "family_max": max([mlp[v]["mean"] for v in GATE_VIEWS]
                                    + [gbm["mean"], gbm_ext["mean"]])}
        print(f"  {nm:<10} MLP speed {mlp['speed']['mean']:.4f} "
              f"(UCL {mlp['speed']['ucl95']:.4f})  coords "
              f"{mlp['coords']['mean']:.4f}  angles {mlp['angles']['mean']:.4f}"
              f"  GBM17 {gbm['mean']:.4f} (UCL {gbm['ucl95']:.4f})"
              f"  GBM23 {gbm_ext['mean']:.4f}  "
              + ", ".join(f"{k}={v}" for k, v in gbm_ext["top_features"][:3]),
              flush=True)
    if args.permutations > 0:
        null = permutation_null_gbm(Xarm["C_B_S5"][:, i17], Xr[:, i17], words,
                                    args.folds, args.seed, args.permutations)
        g4["permutation_null_gbm"] = null
        print(f"  within-pair permutation null (GBM, {null['n_perm']} draws): "
              f"mean {null['null_mean']:.4f}, p95 {null['null_p95']:.4f}, "
              f"max {null['null_max']:.4f}")
    closure = {
        "mlp_speed_closure": gap_closure(g4["v1_repro"]["mlp"]["speed"]["mean"],
                                         g4["C_B_S5"]["mlp"]["speed"]["mean"]),
        "gbm_closure": gap_closure(g4["v1_repro"]["gbm17"]["mean"],
                                   g4["C_B_S5"]["gbm17"]["mean"]),
        "gbm_ext_closure": gap_closure(g4["v1_repro"]["gbm_ext"]["mean"],
                                       g4["C_B_S5"]["gbm_ext"]["mean"]),
    }
    enen = out_dir / "enen.json"
    if enen.exists():
        e = json.loads(enen.read_text())
        if "en_v2_gate" in e and "en_v1_gate" in e:
            closure["enen_closure"] = gap_closure(
                e["en_v1_gate"]["speed"]["final"], e["en_v2_gate"]["speed"]["final"])
    g4["closure"] = closure
    g4["ucl95_standard"] = {
        "bar": G4_STANDARD_UCL,
        "mlp_speed_ucl": g4["C_B_S5"]["mlp"]["speed"]["ucl95"],
        "gbm_ucl": g4["C_B_S5"]["gbm17"]["ucl95"],
        "gbm_ext_ucl": g4["C_B_S5"]["gbm_ext"]["ucl95"],
        "met": bool(max(g4["C_B_S5"]["mlp"]["speed"]["ucl95"],
                        g4["C_B_S5"]["gbm_ext"]["ucl95"]) <= G4_STANDARD_UCL)}
    # `gbm_ext_closure` is reported, never gated: its instrument was not the
    # pre-registered one and inventing a bar for it after seeing the number
    # would be exactly the selection this battery exists to prevent.
    g4_ok = valid and all(closure.get(k, -1.0) >= b for k, b in G4_BARS.items()
                          if k in closure)
    g4["pass"] = g4_ok
    rep["G4"] = g4
    for k, b in G4_BARS.items():
        if k in closure:
            print(f"  gap-closure {k:<18} {closure[k]:>6.1%} >= {b:.0%}   "
                  f"{_pass(closure[k] >= b)}")
        else:
            print(f"  gap-closure {k:<18}    n/a (arm not measured)")
    print(f"  gap-closure {'gbm_ext_closure':<18} "
          f"{closure['gbm_ext_closure']:>6.1%}  (reported, NOT a registered bar "
          f"— stronger critic than the one the 20 % was set on)")
    print(f"  UCL95 standard <= {G4_STANDARD_UCL}: MLP speed "
          f"{g4['ucl95_standard']['mlp_speed_ucl']:.4f}, GBM "
          f"{g4['ucl95_standard']['gbm_ucl']:.4f} — "
          f"{'MET' if g4['ucl95_standard']['met'] else 'OPEN SHORTFALL (recorded)'}")

    rep["verdict"] = {"G1": g1_ok, "G2": g2["pass"], "G3": g3_ok, "G4": g4_ok}
    (out_dir / "gates.json").write_text(json.dumps(rep, indent=1,
                                                   ensure_ascii=False))
    print(f"\nverdict {rep['verdict']}\n-> {out_dir / 'gates.json'}")
    return 0


def length_mix_gate(args: argparse.Namespace,
                    real_words: Sequence[str]) -> Dict[str, object]:
    """G2: the training draw's length mix against the wordfreq token mass.

    The design's G2 demanded both "within ±5 pts of the wordfreq mass" and
    "ru ≤3 in 30–40 %", and wordfreq's own ru ≤3 mass is 26.8 % — the two halves
    are mutually unsatisfiable.  The amended gate keeps the first half at ±3 pts
    and records the 26.8-vs-35.6 residual as a **register** difference (wordfreq
    blends written and subtitle text; swipe input is mobile chat), not a
    generator defect.  Closing it by tuning against the Yandex mix would consume
    the sole validator.
    """
    import script_registry as SR
    import script_synth as SS
    spec = SR.get("ru")
    lex, _ = spec.load_lexicon(SS.T_OUT)
    mass, _ = SS.token_mass(spec, "ru", [w for w, _ in lex])
    L = np.array([len(w) for w, _ in lex])

    def buckets(weights: np.ndarray) -> Dict[str, float]:
        p = np.asarray(weights, np.float64)
        p = p / p.sum()
        return {"le3": float(p[L <= 3].sum()),
                "4to6": float(p[(L >= 4) & (L <= 6)].sum()),
                "ge7": float(p[L >= 7].sum()), "mean_len": float((p * L).sum())}

    def observed(words: Sequence[str]) -> Dict[str, float]:
        a = np.array([len(w) for w in words])
        return {"le3": float((a <= 3).mean()),
                "4to6": float(((a >= 4) & (a <= 6)).mean()),
                "ge7": float((a >= 7).mean()), "mean_len": float(a.mean())}

    mix = {"wordfreq_mass": buckets(np.array([mass.get(w, 0.0) for w, _ in lex])),
           "ckdt_255_rank": buckets(np.array([f for _, f in lex])),
           "real_valid": observed(real_words)}
    for tag, cache in (("v2_train_draw", args.v2_cache),
                       ("v1_train_draw", args.v1_cache)):
        p = resolve(args.workdir, Path(cache) / "train_synth.npz")
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                mix[tag] = observed([str(w) for w in z["words"][:200_000]])
    dev = float("nan")
    if "v2_train_draw" in mix:
        dev = max(abs(mix["v2_train_draw"][k] - mix["wordfreq_mass"][k])
                  for k in ("le3", "4to6", "ge7"))
    return {"mix": mix, "max_dev": dev, "bar": 0.03,
            "pass": bool(dev == dev and dev <= 0.03),
            "register_residual_pts": 100.0 * (mix["real_valid"]["le3"]
                                              - mix["wordfreq_mass"]["le3"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["data", "metrics", "classifier", "v2", "gates"])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--donors", default="train_t3futo.npz,train_t3hws.npz")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rows", type=int, default=0, help="0 = all matched rows")
    ap.add_argument("--folds", type=int, default=5,
                    help="word-disjoint CV folds for the G4 gate")
    ap.add_argument("--permutations", type=int, default=20,
                    help="within-pair label permutations for the G4 null")
    ap.add_argument("--v2-cache", default="cache_ru_v2",
                    help="the generated v2 cache dir, for the G2 length mix")
    ap.add_argument("--v1-cache", default="cache_ru_phaseO",
                    help="the v1 cache dir the G2 table compares against")
    args = ap.parse_args()
    return {"data": stage_data, "metrics": stage_metrics,
            "classifier": stage_classifier, "v2": stage_v2,
            "gates": stage_gates}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
