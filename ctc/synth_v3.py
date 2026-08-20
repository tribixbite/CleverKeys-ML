#!/usr/bin/env python3
"""SYNTH v3 — conditional flow matching over the residual field (Phase Q).

The generator is a small conditional rectified-flow model over the campaign's
``[2,64]`` time-uniform trace representation, expressed as a residual field
from the **arc-uniform ideal-polyline reference** and conditioned on the target
word's polyline geometry alone.  No donors at generation time: where v2
transplants one English human's residuals onto the target polyline
(``script_synth.py`` S1–S5), v3 draws the residual field from a learned
conditional density p(residual | geometry).  Timing needs no separate stage —
the trace is time-uniform and the reference arc-uniform, so dwell, corner
deceleration, tempo shape and the 60 Hz acquisition signature all live in the
residual field the model fits.

Design provenance: SYNTH_V2_RESEARCH_AUDIT.md §2.6 pre-registered the shape
("conditional flow matching over the 64x2 residual field ... not an
MDN-over-offsets and not a full DDPM"); PHASE_Q.md §1 carries the full
rationale and every pre-registered gate.  What v3 keeps from v2 verbatim: the
S0/fix-A wordfreq draw (``script_synth.token_mass``), the npz schema, the split
seeds, and the whole Phase-P gate harness.

THE LICENSE SPLIT (PHASE_Q.md §0, YANDEX_LICENSE_RESEARCH.md §8.1 binds)
------------------------------------------------------------------------
Two twin generators, identical in code and hyperparameters:

* **SHIPPING TRACK** — trained ONLY on MIT data (FUTO t3 + HWS).  Outputs may
  feed shipped decoders.
* **RESEARCH TRACK (sealed)** — trained on the Yandex ru sample
  (``--research-yandex``), legal under the ст. 1335.1 научные carve-out for
  MEASUREMENT ONLY.  Weights, samples, and every decoder trained on them are
  permanently unshippable benchmark artifacts.  This module ENFORCES the seal
  mechanically: a research checkpoint refuses to write outside a
  ``research_only/`` directory, every filename carries ``RESEARCH_ONLY``, and
  every provenance blob is stamped with the license string.  There is no
  laundering path — a shipping v3 retrains from MIT data.

CLI
---
::

  # shipping-track generator
  python3 ctc/synth_v3.py train-gen \
      --bank train_t3futo.npz,train_t3hws.npz --layout en_qwerty.json \
      --out ckpt/synthq_gen_ship/gen.pt

  # sealed research twin
  python3 ctc/synth_v3.py train-gen \
      --bank cache_ru/train_yandex.npz --layout layouts/ru_jcuken_default.json \
      --research-yandex \
      --out research_only/synthq_gen_yx_RESEARCH_ONLY/gen_RESEARCH_ONLY.pt

  # training caches (same schema/splits/seeds as script_synth v2)
  python3 ctc/synth_v3.py sample-cache --gen <gen.pt> --code ru \
      --cache cache_ru_v3 --rows 1000000

  # word-matched arm for the gate battery (same 9,416 Yandex partner words)
  python3 ctc/synth_v3.py matched --gen <gen.pt> \
      --words-npz synth_gap/matched_v2.npz --out synth_gap/matched_v3_ship.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_registry as SR  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from paths import DEFAULT_WORKDIR, resolve  # noqa: E402
from prepare_data import T_OUT  # noqa: E402
from layout_aug import resample_bandwidth  # noqa: E402
from script_synth import (  # noqa: E402
    SEEDS, collapse, endpoint_stats, token_mass, wrong_geometry, write_split)

HERE = Path(__file__).resolve().parent

#: Number of geometry-conditioning channels (PHASE_Q.md §1.2).
COND_CH = 9
#: Trace channels (x, y).
TRACE_CH = 2
#: Samples per trace — the campaign's fixed representation.
N_SAMPLES = 64
#: Pre-registered sampler noise seed base; split offsets train/val/holdout.
NOISE_SEED = 20260820
NOISE_OFFSET = {"train": 0, "val": 1, "holdout": 2, "matched": 3}
#: Pre-registered Euler step count (PHASE_Q.md §1.2).
EULER_STEPS = 32
#: License stamp for every research-track artifact.
RESEARCH_LICENSE = ("RESEARCH_ONLY — Yandex-derived; permanently unshippable "
                    "(YANDEX_LICENSE_RESEARCH.md §8.1, PHASE_Q.md §0/§5.2)")

# ── conditioning ────────────────────────────────────────────────────────────────


def build_cond(seq: np.ndarray, centers: np.ndarray,
               n: int = N_SAMPLES) -> np.ndarray:
    """Geometry conditioning for one collapsed key sequence -> ``[COND_CH, n]``.

    Channels: 0–1 arc-uniform reference polyline R (x, y); 2–3 unit tangent;
    4 normalized arc position; 5 arc distance to the nearest polyline vertex
    (normalized by total length); 6 signed turn angle at that vertex (rad,
    0 at endpoints); 7 ``log1p`` of total ideal length; 8 ``(S − 1)/10``.

    The residual the model learns is ``trace − R`` — channels 0–1 double as the
    reconstruction reference, so the conditioning IS the scaffold and there is
    exactly one definition of it.
    """
    V = centers[seq].astype(np.float64)                    # [S, 2]
    S = len(seq)
    out = np.zeros((COND_CH, n), np.float64)
    out[4] = np.linspace(0.0, 1.0, n)
    seg = np.diff(V, axis=0) if S >= 2 else np.zeros((0, 2))
    lens = np.hypot(seg[:, 0], seg[:, 1]) if S >= 2 else np.zeros(0)
    L = float(lens.sum())
    if S < 2 or L < 1e-9:
        out[0] = V[0, 0]
        out[1] = V[0, 1]
        return out.astype(np.float32)
    cum = np.concatenate([[0.0], np.cumsum(lens)])          # [S] vertex arcs
    s = np.linspace(0.0, L, n)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, S - 2)
    lk = np.maximum(lens[k], 1e-12)
    frac = (s - cum[k]) / lk
    R = V[k] + seg[k] * frac[:, None]                       # [n, 2]
    out[0:2] = R.T
    out[2:4] = (seg[k] / lk[:, None]).T
    dv = np.abs(s[:, None] - cum[None, :])                  # [n, S]
    out[5] = dv.min(1) / L
    turn = np.zeros(S)
    if S >= 3:
        u = seg / np.maximum(lens, 1e-12)[:, None]
        cross = u[:-1, 0] * u[1:, 1] - u[:-1, 1] * u[1:, 0]
        dot = np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0)
        turn[1:S - 1] = np.arctan2(cross, dot)
    out[6] = turn[dv.argmin(1)]
    out[7] = np.log1p(L)
    out[8] = (S - 1) / 10.0
    return out.astype(np.float32)


def cond_table(words: Sequence[str], letters: str, centers: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Per-unique-word conditioning -> ``(cond [U,9,64], word_id [N], uniq)``."""
    idx = {c: i for i, c in enumerate(letters)}
    uniq = sorted(set(str(w) for w in words))
    pos = {w: i for i, w in enumerate(uniq)}
    cond = np.empty((len(uniq), COND_CH, N_SAMPLES), np.float32)
    for w, i in pos.items():
        seq = collapse(np.array([idx[c] for c in w], np.int64))
        cond[i] = build_cond(seq, centers)
    wid = np.array([pos[str(w)] for w in words], np.int64)
    return cond, wid, uniq


# ── model ───────────────────────────────────────────────────────────────────────


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_net(hidden: int = 128, dilations: Sequence[int] = (1, 2, 4, 8, 1, 2, 4, 8),
             temb_dim: int = 128, cond_ch: int = COND_CH):
    """The v3 velocity field: a FiLM-conditioned dilated residual conv net.

    Same inductive bias as the decoder the samples feed (1-D dilated convs over
    the 64-sample axis).  ~1.9 M parameters at the defaults.  The final conv is
    zero-initialised so the flow starts at the identity-ish field.
    """
    torch, nn = _torch()

    class Sinusoidal(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, t):                              # t: [B]
            half = self.dim // 2
            freqs = torch.exp(-math.log(10_000.0)
                              * torch.arange(half, device=t.device) / (half - 1))
            a = t[:, None] * freqs[None] * 1000.0
            return torch.cat([a.sin(), a.cos()], 1)        # [B, dim]

    class Block(nn.Module):
        def __init__(self, ch: int, dil: int, tdim: int) -> None:
            super().__init__()
            self.gn1 = nn.GroupNorm(8, ch)
            self.c1 = nn.Conv1d(ch, ch, 5, padding=2 * dil, dilation=dil)
            self.film = nn.Linear(tdim, 2 * ch)
            self.gn2 = nn.GroupNorm(8, ch)
            self.c2 = nn.Conv1d(ch, ch, 5, padding=2 * dil, dilation=dil)

        def forward(self, x, temb):
            h = self.c1(torch.nn.functional.silu(self.gn1(x)))
            scale, shift = self.film(temb)[:, :, None].chunk(2, 1)
            h = h * (1.0 + scale) + shift
            h = self.c2(torch.nn.functional.silu(self.gn2(h)))
            return x + h

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temb = nn.Sequential(Sinusoidal(temb_dim),
                                      nn.Linear(temb_dim, 2 * temb_dim),
                                      nn.SiLU(),
                                      nn.Linear(2 * temb_dim, 2 * temb_dim))
            self.inp = nn.Conv1d(TRACE_CH + cond_ch, hidden, 1)
            self.blocks = nn.ModuleList(
                [Block(hidden, d, 2 * temb_dim) for d in dilations])
            self.out_gn = nn.GroupNorm(8, hidden)
            self.out = nn.Conv1d(hidden, TRACE_CH, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, x_t, t, cond):                   # [B,2,64],[B],[B,9,64]
            e = self.temb(t)
            h = self.inp(torch.cat([x_t, cond], 1))
            for b in self.blocks:
                h = b(h, e)
            return self.out(torch.nn.functional.silu(self.out_gn(h)))

    return Net()


# ── the acquisition imprint (the one pre-registered repair round) ───────────────
#
# PHASE_Q.md §2 allows one documented repair round before the battery re-run.
# The first battery read named the defect precisely: the GBM's TOP feature on
# the raw v3 arm is `dup_frac` (importance 0.107) — the fraction of exact
# zero-length steps.  Real featurized traces carry exact duplicates because a
# stationary finger produces identical raw samples and the 60 Hz chain
# preserves them; a continuous flow density has probability zero of emitting
# two bit-equal points.  This is a REPRESENTATION tell, not motor behaviour —
# the same class of finding as v2's S5 (half the cornering gap was acquisition,
# not humans).  The repair puts the real chain's two discrete imprints back at
# sampling time:
#
#   1. bandwidth — draw a duration from the MIT-fit law
#      log T = a + b·log L + c·log S + ε·sd  and re-featurize through the real
#      60 Hz chain (`layout_aug.resample_bandwidth`), giving the output the
#      duration-conditional smoothness SPREAD the model blurs over;
#   2. dwell snap — steps shorter than ε_snap collapse to exact duplicates,
#      with ε_snap fit ON ENGLISH so that generated-English dup_frac matches
#      the English bank's own dup_frac (fit on MIT, checked on ru — the S4/S5
#      discipline, no Yandex statistic enters the shipping fit).
#
# The research twin fits the same two parameters on ITS training corpus
# (sealed), keeping the twins symmetric.


def snap_dwells(paths: np.ndarray, eps: float) -> np.ndarray:
    """Collapse sub-``eps`` steps to exact duplicates (vectorised, grouped).

    Group ids advance only on steps ≥ eps computed on the ORIGINAL path, so
    the snap cannot cascade; every point takes its group's first sample.
    """
    if eps <= 0:
        return paths
    P = paths.transpose(0, 2, 1)                            # [N,64,2]
    d = np.hypot(*(np.diff(P, axis=1).transpose(2, 0, 1)))   # [N,63]
    g = np.zeros((len(P), N_SAMPLES), np.int64)
    g[:, 1:] = np.cumsum(d >= eps, axis=1)
    # first sample index of each group, per row (init high, take the minimum)
    rows = np.broadcast_to(np.arange(len(P))[:, None], g.shape)
    idx = np.broadcast_to(np.arange(N_SAMPLES)[None, :], g.shape)
    first = np.full((len(P), N_SAMPLES), N_SAMPLES - 1, np.int64)
    np.minimum.at(first, (rows, g), idx)
    take = first[rows, g]
    out = P[rows, take]                                      # [N,64,2]
    return out.transpose(0, 2, 1).astype(np.float32)


def apply_imprint(paths: np.ndarray, cond: np.ndarray, imprint: Dict[str, object],
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Bandwidth + dwell-snap chain -> ``(paths, drawn_T_ms)``.

    L and S are decoded from the conditioning channels (7: log1p L, 8:
    (S−1)/10), so the imprint sees exactly the geometry the model saw.
    """
    law = imprint["duration_law"]
    a, b, c, sd = (float(law[k]) for k in ("intercept", "b_length", "c_segments",
                                           "resid_sd"))
    L = np.expm1(cond[:, 7, 0].astype(np.float64))
    S = np.round(cond[:, 8, 0].astype(np.float64) * 10.0) + 1.0
    nseg = np.maximum(S - 1.0, 1.0)
    T = np.zeros(len(paths), np.float32)
    ok = L > 1e-6
    T[ok] = np.exp(a + b * np.log(L[ok]) + c * np.log(nseg[ok])
                   + rng.standard_normal(int(ok.sum())) * sd).astype(np.float32)
    out = paths.copy()
    for i in np.nonzero(ok)[0]:
        out[i] = resample_bandwidth(out[i], float(T[i]))
    return snap_dwells(out, float(imprint["snap_eps"])), T


def fit_duration_law_npz(bank_paths: Sequence[Path], letters: str,
                         centers: np.ndarray) -> Dict[str, float]:
    """``log T = a + b·log L_ideal + c·log n_segments`` on bank npz rows."""
    idx = {ch: i for i, ch in enumerate(letters)}
    Ls: List[np.ndarray] = []
    Ss: List[np.ndarray] = []
    Ts: List[np.ndarray] = []
    for p in bank_paths:
        with np.load(p) as d:
            if "duration_ms" not in d:
                raise SystemExit(f"{p} has no duration_ms — rebuild the bank "
                                 f"(prepare_data.py records it)")
            words = [str(w) for w in d["words"]]
            T = np.asarray(d["duration_ms"], np.float64)
        L = np.empty(len(words))
        S = np.empty(len(words))
        for i, w in enumerate(words):
            seq = collapse(np.array([idx[ch] for ch in w], np.int64))
            seg = np.hypot(*np.diff(centers[seq].astype(np.float64), axis=0).T)
            L[i] = seg.sum()
            S[i] = max(len(seg), 1)
        Ls.append(L)
        Ss.append(S)
        Ts.append(T)
    L, S, T = np.concatenate(Ls), np.concatenate(Ss), np.concatenate(Ts)
    ok = (L > 1e-6) & (T > 1.0)
    X = np.stack([np.ones(int(ok.sum())), np.log(L[ok]), np.log(S[ok])], 1)
    y = np.log(T[ok])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - float((resid ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return {"intercept": float(coef[0]), "b_length": float(coef[1]),
            "c_segments": float(coef[2]), "resid_sd": float(resid.std()),
            "r2": r2, "n": int(ok.sum()),
            "median_T_ms": float(np.median(T[ok]))}


def fit_duration_law_yandex(jsonl: Path, letters: str, centers: np.ndarray,
                            stride: int = 6) -> Dict[str, float]:
    """The twin's duration law, streamed from the raw corpus (sealed footing)."""
    from prepare_yandex import iter_corpus, keep_reason
    idx = {ch: i for i, ch in enumerate(letters)}
    Ls: List[float] = []
    Ss: List[float] = []
    Ts: List[float] = []
    kept = 0
    for word, gname, xs, ys, ts, _ in iter_corpus(jsonl):
        if gname != "default":
            continue
        p, why = keep_reason(word, xs, ts)
        if p is None:
            continue
        kept += 1
        if kept % stride:
            continue
        seq = collapse(np.array([idx[ch] for ch in p], np.int64))
        if len(seq) < 2:
            continue
        seg = np.hypot(*np.diff(centers[seq].astype(np.float64), axis=0).T)
        Ls.append(float(seg.sum()))
        Ss.append(float(max(len(seg), 1)))
        Ts.append(float(ts[-1]))
    L, S, T = (np.asarray(a, np.float64) for a in (Ls, Ss, Ts))
    ok = (L > 1e-6) & (T > 1.0)
    X = np.stack([np.ones(int(ok.sum())), np.log(L[ok]), np.log(S[ok])], 1)
    y = np.log(T[ok])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - float((resid ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return {"intercept": float(coef[0]), "b_length": float(coef[1]),
            "c_segments": float(coef[2]), "resid_sd": float(resid.std()),
            "r2": r2, "n": int(ok.sum()),
            "median_T_ms": float(np.median(T[ok]))}


def bank_dup_frac(bank_paths: Sequence[Path], limit: int = 100_000) -> float:
    fr: List[float] = []
    for p in bank_paths:
        with np.load(p) as d:
            f = np.asarray(d["features"][:limit], np.float64)
        dd = np.hypot(*(np.diff(f.transpose(0, 2, 1), axis=1).transpose(2, 0, 1)))
        fr.append(float((dd < 1e-9).mean()))
    return float(np.mean(fr))


def cmd_fit_imprint(args: argparse.Namespace) -> int:
    """Fit the imprint (duration law + snap ε) on the GENERATOR'S OWN corpus."""
    gen = Generator(args.gen if args.gen.is_absolute()
                    else resolve(args.workdir, args.gen))
    out = args.out if args.out.is_absolute() else resolve(args.workdir, args.out)
    if gen.research:
        assert_sealed(out, "fit-imprint --out")
    bank_paths = []
    for name in args.bank.split(","):
        p = Path(name.strip())
        p = p if p.is_absolute() else resolve(args.workdir, p)
        bank_paths.append(p if p.exists()
                          else resolve(args.workdir, Path("cache") / name.strip()))
    layout_path = args.layout if args.layout.exists() else HERE / args.layout
    letters_l, centers = load_layout(layout_path)
    letters = "".join(letters_l)

    if args.yandex_jsonl:
        # Research twin: durations live only in the raw corpus jsonl (the
        # featurized cache drops them).  Sealed footing — the law is stamped
        # research and can never touch a shipping generator (load_imprint).
        if not gen.research:
            raise SystemExit("--yandex-jsonl is research-track only; the "
                             "shipping imprint is fit on the MIT bank npz")
        law = fit_duration_law_yandex(Path(args.yandex_jsonl), letters, centers,
                                      stride=args.yandex_stride)
    else:
        law = fit_duration_law_npz(bank_paths, letters, centers)
    print(f"[imprint] duration law: {law}", flush=True)
    target = bank_dup_frac(bank_paths)
    print(f"[imprint] bank dup_frac target {target:.4f}", flush=True)

    # ε fit: generate on the bank's own word distribution, chain with the law,
    # bisect ε so generated dup_frac matches the bank's.
    with np.load(bank_paths[0]) as d:
        words = [str(w) for w in d["words"]]
    rng = np.random.default_rng(4242)
    sample_words = [words[i] for i in rng.integers(len(words), size=args.fit_rows)]
    cond_all, wid, _ = cond_table(sample_words, letters, centers)
    cond = cond_all[wid]
    raw = gen.sample(cond, seed=NOISE_SEED + 99)
    imp = {"duration_law": law, "snap_eps": 0.0}
    band, _ = apply_imprint(raw, cond, imp, np.random.default_rng(4243))

    def dup_of(eps: float) -> float:
        f = snap_dwells(band, eps)
        dd = np.hypot(*(np.diff(f.transpose(0, 2, 1), axis=1).transpose(2, 0, 1)))
        return float((dd < 1e-9).mean())

    lo, hi = 0.0, 0.02
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if dup_of(mid) < target:
            lo = mid
        else:
            hi = mid
    eps = 0.5 * (lo + hi)
    got = dup_of(eps)
    print(f"[imprint] snap_eps {eps:.6f} -> generated dup_frac {got:.4f} "
          f"(target {target:.4f})", flush=True)
    blob = {"duration_law": law, "snap_eps": eps, "dup_frac_target": target,
            "dup_frac_fit": got, "fit_rows": args.fit_rows,
            "bank": [str(p) for p in bank_paths],
            "gen_ckpt_sha16": gen.sha16, "research": gen.research}
    if gen.research:
        blob["license"] = RESEARCH_LICENSE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1))
    print(f"[imprint] -> {out}")
    return 0


def load_imprint(path: Optional[Path], workdir: Path,
                 gen: "Generator") -> Optional[Dict[str, object]]:
    if path is None:
        return None
    p = path if path.is_absolute() else resolve(workdir, path)
    blob = json.loads(p.read_text())
    if blob.get("research") and not gen.research:
        raise SystemExit(f"{p} is a RESEARCH_ONLY imprint but the generator is "
                         f"shipping-track — refusing the cross-track mix")
    return blob


# ── research-track seal ─────────────────────────────────────────────────────────


def assert_sealed(path: Path, what: str) -> None:
    """A research artifact must live under a ``research_only/`` directory and
    carry the ``RESEARCH_ONLY`` marker in its filename.  Refuse otherwise."""
    p = path.resolve()
    if "research_only" not in p.parts:
        raise SystemExit(f"{what}: {p} is Yandex-derived and MUST live under a "
                         f"research_only/ directory (PHASE_Q.md §5.2)")
    if "RESEARCH_ONLY" not in p.name and "RESEARCH_ONLY" not in p.parent.name:
        raise SystemExit(f"{what}: {p} must carry the RESEARCH_ONLY marker in "
                         f"its file or directory name (PHASE_Q.md §5.2)")


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── training ────────────────────────────────────────────────────────────────────


def cmd_train_gen(args: argparse.Namespace) -> int:
    torch, _ = _torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out if args.out.is_absolute() else resolve(args.workdir, args.out)
    if args.research_yandex:
        assert_sealed(out, "train-gen --out")
    out.parent.mkdir(parents=True, exist_ok=True)

    layout_path = args.layout if args.layout.exists() else HERE / args.layout
    letters_l, centers = load_layout(layout_path)
    letters = "".join(letters_l)

    feats_np: List[np.ndarray] = []
    words: List[str] = []
    for name in args.bank.split(","):
        p = Path(name.strip())
        p = p if p.is_absolute() else resolve(args.workdir, p)
        p = p if p.exists() else resolve(args.workdir, Path("cache") / name.strip())
        with np.load(p) as d:
            feats_np.append(np.asarray(d["features"], np.float32))
            words.extend(str(w) for w in d["words"])
        print(f"[v3] bank {p}: {len(feats_np[-1])} rows", flush=True)
    feats = np.concatenate(feats_np) if len(feats_np) > 1 else feats_np[0]
    del feats_np
    assert len(feats) == len(words)

    print(f"[v3] conditioning table over {len(set(words))} unique words …",
          flush=True)
    cond_np, wid_np, _ = cond_table(words, letters, centers)

    # Global residual scale σ — one scalar, rms over a deterministic subsample.
    sub = np.arange(0, len(feats), max(1, len(feats) // 200_000))
    resid = feats[sub] - cond_np[wid_np[sub], 0:2]
    sigma = float(np.sqrt(np.mean(resid.astype(np.float64) ** 2)))
    print(f"[v3] sigma (rms residual, n={len(sub)}) = {sigma:.6f}", flush=True)
    del resid, sub

    # Everything on GPU: 1 M x 2 x 64 fp32 is 512 MB; the model is ~2 M params.
    F = torch.from_numpy(feats).to(dev)
    C = torch.from_numpy(cond_np).to(dev)
    W = torch.from_numpy(wid_np).to(dev)
    del feats

    # 2 % of rows for CFM-loss monitoring only (deterministic stride).
    val_mask = (torch.arange(len(F), device=dev) % 50) == 49
    tr_rows = torch.nonzero(~val_mask, as_tuple=True)[0]
    va_rows = torch.nonzero(val_mask, as_tuple=True)[0][:8192]
    print(f"[v3] rows: train {len(tr_rows)}, val-monitor {len(va_rows)}",
          flush=True)

    torch.manual_seed(args.seed)
    net = make_net(hidden=args.hidden).to(dev)
    ema = make_net(hidden=args.hidden).to(dev)
    ema.load_state_dict(net.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[v3] params {n_par:,}  device {dev}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.01 + 0.5 * (args.lr - args.lr * 0.01) * (
            1.0 + math.cos(math.pi * p))

    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)

    def cfm_batch(rows):
        x1 = (F[rows] - C[W[rows], 0:2]) / sigma
        cond = C[W[rows]]
        t = torch.rand(len(rows), device=dev, generator=g)
        x0 = torch.randn(x1.shape, device=dev, generator=g)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        return x_t, t, cond, x1 - x0

    t0 = time.time()
    cfg = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in vars(args).items() if k != "fn"}
    ckpt = {"config": cfg | {"letters": letters, "sigma": sigma,
                             "layout": str(layout_path.name),
                             "params": n_par,
                             "research": bool(args.research_yandex),
                             "license": (RESEARCH_LICENSE
                                         if args.research_yandex
                                         else "MIT-trained (shipping track)")}}
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        rows = tr_rows[torch.randint(len(tr_rows), (args.batch,), device=dev,
                                     generator=g)]
        x_t, t, cond, v_tgt = cfm_batch(rows)
        loss = torch.nn.functional.mse_loss(net(x_t, t, cond), v_tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for q, p in zip(ema.parameters(), net.parameters()):
                q.mul_(args.ema).add_(p, alpha=1.0 - args.ema)
            for q, p in zip(ema.buffers(), net.buffers()):
                q.copy_(p)
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:>7}  loss {loss.item():.5f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{(step + 1) / (time.time() - t0):.1f} it/s", flush=True)
        if step % 2000 == 0 or step == args.steps - 1:
            gv = torch.Generator(device=dev)
            gv.manual_seed(777)
            with torch.no_grad():
                x1 = (F[va_rows] - C[W[va_rows], 0:2]) / sigma
                cond = C[W[va_rows]]
                t = torch.rand(len(va_rows), device=dev, generator=gv)
                x0 = torch.randn(x1.shape, device=dev, generator=gv)
                x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
                vl = torch.nn.functional.mse_loss(ema(x_t, t, cond),
                                                  x1 - x0).item()
            print(f"  step {step:>7}  VAL(ema) {vl:.5f}", flush=True)
        if step and step % 10_000 == 0:
            torch.save(ckpt | {"model": net.state_dict(),
                               "ema": ema.state_dict(), "step": step}, out)
    torch.save(ckpt | {"model": net.state_dict(), "ema": ema.state_dict(),
                       "step": args.steps}, out)
    print(f"[v3] done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    return 0


# ── sampling ────────────────────────────────────────────────────────────────────


class Generator:
    """A loaded v3 checkpoint: EMA net + sigma + the seal flag."""

    def __init__(self, ckpt_path: Path, device: Optional[str] = None) -> None:
        torch, _ = _torch()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(ckpt_path, map_location=self.dev, weights_only=False)
        cfg = blob["config"]
        self.net = make_net(hidden=cfg.get("hidden", 128)).to(self.dev)
        self.net.load_state_dict(blob["ema"])
        self.net.eval()
        self.sigma = float(cfg["sigma"])
        self.research = bool(cfg.get("research", False))
        self.license = str(cfg.get("license", ""))
        self.step = int(blob.get("step", -1))
        self.path = Path(ckpt_path)
        self.sha16 = _sha16(Path(ckpt_path))

    def sample(self, cond: np.ndarray, seed: int, steps: int = EULER_STEPS,
               batch: int = 8192) -> np.ndarray:
        """Euler-integrate the flow -> traces ``[N,2,64]`` clipped to [0,1]."""
        torch, _ = _torch()
        g = torch.Generator(device=self.dev)
        g.manual_seed(seed)
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(cond), batch):
                c = torch.from_numpy(cond[i:i + batch]).to(self.dev)
                x = torch.randn((len(c), TRACE_CH, N_SAMPLES), device=self.dev,
                                generator=g)
                for k in range(steps):
                    t = torch.full((len(c),), k / steps, device=self.dev)
                    x = x + self.net(x, t, c) / steps
                p = (c[:, 0:2] + self.sigma * x).clamp_(0.0, 1.0)
                outs.append(p.cpu().numpy().astype(np.float32))
        return np.concatenate(outs) if len(outs) > 1 else outs[0]


def gen_provenance(gen: Generator, extra: Dict[str, object]) -> Dict[str, object]:
    prov: Dict[str, object] = dict(
        generator="synth_v3.py", generator_version="v3-cfm",
        gen_ckpt=str(gen.path), gen_ckpt_sha16=gen.sha16,
        gen_train_step=gen.step, sigma=gen.sigma, euler_steps=EULER_STEPS,
        **extra)
    if gen.research:
        prov["license"] = RESEARCH_LICENSE
    return prov


def cmd_sample_cache(args: argparse.Namespace) -> int:
    gen = Generator(args.gen if args.gen.is_absolute()
                    else resolve(args.workdir, args.gen))
    spec = SR.get(args.code)
    cache = resolve(args.workdir, Path(args.cache))
    if gen.research:
        assert_sealed(cache, "sample-cache --cache")
    cache.mkdir(parents=True, exist_ok=True)
    want_files = {"train": "train_synth.npz", "val": "val.npz",
                  "holdout": "holdout.npz"}
    want = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not args.force:
        clash = [want_files[s] for s in want if (cache / want_files[s]).exists()]
        if clash:
            raise SystemExit(f"{cache} already holds {clash} — pass --force")

    letters_l, centers = load_layout(HERE / spec.layout_json)
    letters = "".join(letters_l)
    assert letters == spec.letters
    lexicon, lex_st = spec.load_lexicon(T_OUT)
    print(f"[{spec.code}] lexicon {lex_st}", flush=True)

    # S0 / fix A verbatim: the wordfreq token-mass draw (script_synth.token_mass).
    lang = args.wf_lang or spec.lexicon.lang or spec.code
    mass, freq_st = token_mass(spec, lang, [w for w, _ in lexicon])
    weights = np.array([mass.get(w, 0.0) for w, _ in lexicon], np.float64)
    if weights.sum() <= 0:
        raise SystemExit(f"[{spec.code}] wordfreq '{lang}' gives zero mass")
    weights /= weights.sum()
    print(f"[{spec.code}] fix A draw: {freq_st}", flush=True)

    lex_words = np.array([w for w, _ in lexicon])
    print(f"[{spec.code}] conditioning table over {len(lex_words)} lexicon "
          f"words …", flush=True)
    cond_all, _, uniq = cond_table(lex_words, letters, centers)
    pos = {w: i for i, w in enumerate(uniq)}

    report: Dict[str, object] = {
        "script": spec.code, "layout": spec.layout_json, "letters": letters,
        "lexicon": lex_st, "lexicon_tier": spec.lexicon.tier,
        "generator_version": "v3-cfm", "gen_ckpt": str(gen.path),
        "gen_ckpt_sha16": gen.sha16, "sigma": gen.sigma,
        "euler_steps": EULER_STEPS, "wordfreq_draw": freq_st,
        "research": gen.research,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if gen.research:
        report["license"] = RESEARCH_LICENSE

    imprint = load_imprint(args.imprint, args.workdir, gen)
    if imprint is not None:
        report["imprint"] = imprint

    for split in want:
        n = {"train": args.rows, "val": args.val_rows,
             "holdout": args.holdout_rows}[split]
        rng = np.random.default_rng(SEEDS[split])
        draw = rng.choice(len(lex_words), size=n, p=weights)
        words = [str(w) for w in lex_words[draw]]
        cond = cond_all[[pos[w] for w in words]]
        t0 = time.time()
        feats = gen.sample(cond, seed=NOISE_SEED + NOISE_OFFSET[split],
                           steps=args.euler_steps)
        rows_extra: Dict[str, np.ndarray] = {}
        if imprint is not None:
            feats, T = apply_imprint(
                feats, cond, imprint,
                np.random.default_rng(NOISE_SEED + NOISE_OFFSET[split] + 1000))
            rows_extra["drawn_duration_ms"] = T
        dt = time.time() - t0
        st = {"made": n, "seconds": round(dt, 1), "rows_per_s": round(n / dt)}
        prov = gen_provenance(gen, dict(
            script=spec.code, split=split, seed=SEEDS[split],
            noise_seed=NOISE_SEED + NOISE_OFFSET[split], rows=n,
            imprint=bool(imprint), layout=spec.layout_json, stats=st,
            lexicon=spec.lexicon.tier))
        out = write_split(cache, want_files[split], feats, words, letters, prov,
                          rows=rows_extra or None)
        report[f"gen_{split}"] = st
        print(f"[{spec.code}] {split}: {st} -> {out}", flush=True)
        if split in ("train", "holdout"):
            v = min(2000, n)
            ep = endpoint_stats(feats[:v], words[:v], letters, centers)
            ctrl = endpoint_stats(feats[:v], words[:v], letters,
                                  wrong_geometry(letters,
                                                 np.random.default_rng(4242)))
            report[f"endpoints_{split}"] = ep
            report[f"endpoints_{split}_wrong_geo_control"] = ctrl
            print(f"[{spec.code}] {split} endpoints {ep}", flush=True)
            print(f"[{spec.code}] {split} wrong-geo control {ctrl}", flush=True)

    (cache / "synth_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[{spec.code}] wrote {cache / 'synth_stats.json'}")
    return 0


def cmd_matched(args: argparse.Namespace) -> int:
    """One v3 trace per word of an existing word-matched arm file."""
    gen = Generator(args.gen if args.gen.is_absolute()
                    else resolve(args.workdir, args.gen))
    out = args.out if args.out.is_absolute() else resolve(args.workdir, args.out)
    if gen.research:
        assert_sealed(out, "matched --out")
    src = args.words_npz if args.words_npz.is_absolute() else resolve(
        args.workdir, args.words_npz)
    with np.load(src, allow_pickle=False) as d:
        words = [str(w) for w in d["words"]]
    letters_l, centers = load_layout(HERE / "layouts" / args.layout_json)
    letters = "".join(letters_l)
    cond_all, wid, _ = cond_table(words, letters, centers)
    cond = cond_all[wid]
    feats = gen.sample(cond, seed=NOISE_SEED + NOISE_OFFSET["matched"],
                       steps=args.euler_steps)
    imprint = load_imprint(args.imprint, args.workdir, gen)
    if imprint is not None:
        feats, _ = apply_imprint(
            feats, cond, imprint,
            np.random.default_rng(NOISE_SEED + NOISE_OFFSET["matched"] + 1000))
    prov = gen_provenance(gen, dict(role="word-matched gate arm",
                                    words_npz=str(src), n=len(words),
                                    imprint=bool(imprint),
                                    euler_steps_used=args.euler_steps))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, words=np.array(words),
                        provenance=np.array(json.dumps(prov, sort_keys=True)))
    print(f"[v3] matched arm: {len(words)} rows, euler {args.euler_steps} "
          f"-> {out}")
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tg = sub.add_parser("train-gen", help="train a v3 generator")
    tg.add_argument("--bank", required=True,
                    help="comma-separated npz paths (workdir- or cache-relative)")
    tg.add_argument("--layout", type=Path, required=True,
                    help="layout json for the bank's script (letters + centers)")
    tg.add_argument("--out", type=Path, required=True)
    tg.add_argument("--research-yandex", action="store_true",
                    help="SEALED research twin: enforces research_only/ paths "
                         "and stamps the RESEARCH_ONLY license everywhere")
    tg.add_argument("--steps", type=int, default=120_000)
    tg.add_argument("--batch", type=int, default=512)
    tg.add_argument("--lr", type=float, default=3e-4)
    tg.add_argument("--warmup", type=int, default=1000)
    tg.add_argument("--ema", type=float, default=0.999)
    tg.add_argument("--hidden", type=int, default=128)
    tg.add_argument("--seed", type=int, default=1234)
    tg.set_defaults(fn=cmd_train_gen)

    sc = sub.add_parser("sample-cache", help="write train/val/holdout caches")
    sc.add_argument("--gen", type=Path, required=True)
    sc.add_argument("--code", required=True)
    sc.add_argument("--cache", required=True)
    sc.add_argument("--rows", type=int, default=1_000_000)
    sc.add_argument("--val-rows", type=int, default=5_000)
    sc.add_argument("--holdout-rows", type=int, default=10_000)
    sc.add_argument("--splits", default="train,val,holdout")
    sc.add_argument("--euler-steps", type=int, default=EULER_STEPS)
    sc.add_argument("--imprint", type=Path, default=None,
                    help="acquisition-imprint json from fit-imprint (the one "
                         "pre-registered repair round)")
    sc.add_argument("--wf-lang", default="")
    sc.add_argument("--force", action="store_true")
    sc.set_defaults(fn=cmd_sample_cache)

    ma = sub.add_parser("matched", help="word-matched arm for the gate battery")
    ma.add_argument("--gen", type=Path, required=True)
    ma.add_argument("--words-npz", type=Path, required=True)
    ma.add_argument("--out", type=Path, required=True)
    ma.add_argument("--layout-json", default="ru_jcuken_default.json")
    ma.add_argument("--euler-steps", type=int, default=EULER_STEPS)
    ma.add_argument("--imprint", type=Path, default=None)
    ma.set_defaults(fn=cmd_matched)

    fi = sub.add_parser("fit-imprint",
                        help="fit the acquisition imprint (duration law + "
                             "dwell-snap ε) on the generator's own corpus")
    fi.add_argument("--gen", type=Path, required=True)
    fi.add_argument("--bank", required=True,
                    help="featurized bank npz(s): dup_frac target + ε-fit word "
                         "distribution (and the duration law, unless "
                         "--yandex-jsonl)")
    fi.add_argument("--layout", type=Path, required=True)
    fi.add_argument("--out", type=Path, required=True)
    fi.add_argument("--fit-rows", type=int, default=20_000)
    fi.add_argument("--yandex-jsonl", default="",
                    help="research twin only: fit the law from the raw corpus "
                         "(the featurized cache has no durations)")
    fi.add_argument("--yandex-stride", type=int, default=6)
    fi.set_defaults(fn=cmd_fit_imprint)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
