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
        "speed_asym", "dup_frac", "key_cover")}
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
    return {k: np.asarray(v) for k, v in m.items()}


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no scipy dependency)."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["data", "metrics", "classifier"])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--donors", default="train_t3futo.npz,train_t3hws.npz")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    return {"data": stage_data, "metrics": stage_metrics,
            "classifier": stage_classifier}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
