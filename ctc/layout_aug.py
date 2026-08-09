#!/usr/bin/env python3
"""Layout-resampling augmentation (Phase H) — teach key RE-ARRANGEMENT.

`ALT_LAYOUT_EVAL.md` established that slot-permutation + shared-affine
augmentation yields slot invariance (perfect) and affine tolerance (good) but
NOT layout invariance: accuracy decays monotonically with mean key displacement
from QWERTY, bottoming at dvorak (t1 63.04 vs the geometric engine's 76.8).
No axis-aligned affine turns QWERTY into Dvorak; the missing augmentation is a
per-sample GEOMETRY resample: draw an alternative keyboard geometry and warp the
cached ``[2,64]`` QWERTY path onto it, so the model sees the same words swiped
on many letter arrangements.

The warp — residual re-anchoring on the word's ideal polyline
-------------------------------------------------------------
A cached path for word *w* is decomposed against the **ideal polyline** that
visits *w*'s key centers on the source (QWERTY) geometry:

1. every path point is assigned a point on the polyline — segment ``j`` and
   within-segment fraction ``t`` — by a **monotone** dynamic program (nearest
   segment subject to the assignment never moving backwards along the polyline;
   plain nearest-point projection is ambiguous when a word revisits a letter,
   e.g. "there" passes e twice);
2. the **residual** (path point minus its polyline anchor) is expressed in the
   segment's local tangent/normal frame;
3. the polyline is rebuilt through the SAME word's key centers on the target
   geometry, the anchor is re-placed on the target segment by an arc-length
   remap that is **absolute (slope 1) within a dwell radius of each vertex**
   and proportional only across the segment middle, and the residual is
   re-applied in the target segment's local frame, in absolute keyboard units.

The vertex-absolute remap matters: projection makes the residual purely normal
wherever ``t`` is interior, so an endpoint's tangential dwell (finger resting
just off the key along the outgoing direction) lands in ``t`` — a purely
proportional remap would scale that dwell with the target segment's length and
drag endpoints off their keys (measured: dvorak endpoint nearest-key hit falls
0.87→0.68 under proportional-only). With the dwell band absolute, near-key
geometry transfers untouched and only the inter-key transit stretches.

By construction the warped path visits the target geometry's key centers for
the word (an ideal path maps exactly to the target ideal path, and the source
identity warp is the identity), while the human deviation — overshoot, corner
cutting, jitter, and the implicit speed profile carried by the time-uniform
sample spacing — transfers point-for-point. The alternative considered (an
inverse-distance/thin-plate displacement field over the 26 key-center
correspondences, per `ALT_LAYOUT_EVAL.md` §"What it would take") needs no
target word but gives no guarantee that the warped path visits the word's keys
in order: between anchors the field is dominated by whatever unrelated keys sit
near the segment, exactly where QWERTY→Dvorak scrambles neighborhoods the most.
Residual re-anchoring is ~the same code size, is exact where it matters, and
costs O(64·S) per sample (S = distinct-letter segments), so it was chosen.

Residuals are carried in absolute units (not scaled by segment length): human
motor noise scales with finger/key size — near-constant across layouts (all
vendored layouts share rx 0.05) — not with inter-key distance.

Known limitation, stated: a corner residual is re-applied in the frame of the
*incoming* segment, so when the turn angle at a letter differs between
geometries the transferred corner-cut is plausible rather than exact. This is
augmentation, not simulation; the validation CLI below measures how plausible.

Synthetic geometries
--------------------
``synth_geometry()`` draws a plausible 3- or 4-row lattice (row-count pattern
from the observed family, x pitch ~ the real layouts' 0.085–0.10, per-row
stagger, per-key jitter) and assigns the 26 letters by a uniform random
permutation — the strongest re-arrangement signal available, an infinite
family. Real vendored layouts (azerty/qwertz/german/spanish az26) add realistic
structure. **dvorak is excluded from the training pool** so the dvorak
real-corpus eval remains a true transfer probe; qwerty is excluded because it
is the canonical geometry.

Validation CLI
--------------
    python3 layout_aug.py --validate --target dvorak      # endpoint stats
    python3 layout_aug.py --validate --target qwerty      # identity round-trip
    python3 layout_aug.py --selftest                      # exactness invariants

Anchors (ALT_LAYOUT_EVAL.md §2, real corpora): a healthy frame puts endpoint
nearest-key hit rates in the 0.73–0.97 band; warped-path stats on the target
geometry must land in that band, and the identity warp must be exact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAYOUT_DIR = SCRIPT_DIR / "layouts"

#: Real layouts eligible for the training pool. dvorak is deliberately absent —
#: it is the held-out transfer probe — and qwerty is the canonical geometry.
DEFAULT_REAL_POOL = ("azerty", "qwertz", "german", "spanish")

#: Row-count patterns observed across the vendored layouts (3-row 10/9/7 family)
#: and the 4-row dvorak family, plus near neighbours. A pattern is drawn
#: uniformly and its row order shuffled.
ROW_PATTERNS_3 = ((10, 9, 7), (9, 9, 8), (10, 8, 8), (11, 8, 7))
ROW_PATTERNS_4 = ((7, 7, 7, 5), (7, 7, 6, 6), (8, 7, 6, 5), (8, 6, 6, 6))

#: Per-key positional jitter sigma for synthetic lattices (keyboard units).
SYNTH_KEY_JITTER = 0.006
#: Per-row horizontal stagger range for synthetic lattices.
SYNTH_STAGGER = 0.04
#: x pitch range; real layouts sit at 0.1 (qwerty rows) with tighter packing on
#: wider rows. The lattice is clamped into cx ∈ [0.05, 0.95] regardless.
SYNTH_PITCH_LO, SYNTH_PITCH_HI = 0.085, 0.10

#: Dwell radius (keyboard units) around each polyline vertex inside which the
#: arc-length remap has slope 1 (absolute transfer). ≈ the key half-width every
#: vendored layout shares (rx 0.05): within a key, geometry is layout-invariant.
DWELL_RADIUS = 0.05


def load_az_centers(path: Path) -> np.ndarray:
    """Layout json -> ``[26,2] float32`` a–z key centers in a..z order.

    Works for both ``en_qwerty.json`` (letters == key order) and the vendored
    ``futo_*.json`` (extra non-a-z keys are ignored — the az26 arm, which is the
    training regime and the headline eval arm).
    """
    obj = json.loads(Path(path).read_text())
    by = {k["letter"]: (float(k["cx"]), float(k["cy"])) for k in obj["keys"]}
    missing = [chr(97 + i) for i in range(26) if chr(97 + i) not in by]
    if missing:
        raise SystemExit(f"{path}: layout lacks a-z keys {missing}")
    return np.array([by[chr(97 + i)] for i in range(26)], np.float32)


def synth_geometry() -> np.ndarray:
    """Draw a random plausible letter-key geometry -> ``[26,2] float32`` (a..z).

    Uses the global ``np.random`` stream (matching ``train.py``'s augmentation,
    which is per-worker seeded by the DataLoader).
    """
    if np.random.rand() < 0.5:
        pattern = ROW_PATTERNS_3[np.random.randint(len(ROW_PATTERNS_3))]
    else:
        pattern = ROW_PATTERNS_4[np.random.randint(len(ROW_PATTERNS_4))]
    counts = list(pattern)
    np.random.shuffle(counts)
    n_rows = len(counts)
    # Row centerlines follow the layout convention: (2r+1)/(2R) — matches both
    # qwerty (1/6, 1/2, 5/6) and dvorak (0.125, 0.375, 0.625, 0.875 ≈ vendored).
    row_y = (2.0 * np.arange(n_rows) + 1.0) / (2.0 * n_rows)
    n_max = max(counts)
    pitch_hi = min(SYNTH_PITCH_HI, 0.9 / max(n_max - 1, 1))
    pitch = float(np.random.uniform(min(SYNTH_PITCH_LO, pitch_hi), pitch_hi))
    pos = np.empty((26, 2), np.float32)
    k = 0
    for r, n in enumerate(counts):
        width = (n - 1) * pitch
        c0 = 0.5 - width / 2.0 + float(np.random.uniform(-SYNTH_STAGGER,
                                                         SYNTH_STAGGER))
        c0 = min(max(c0, 0.05), 0.95 - width)       # keep the row inside [0.05,0.95]
        for i in range(n):
            pos[k] = (c0 + i * pitch, row_y[r])
            k += 1
    pos += np.random.normal(0.0, SYNTH_KEY_JITTER, pos.shape).astype(np.float32)
    np.clip(pos, 0.02, 0.98, out=pos)
    # Random letter assignment: position k holds letter perm[k].
    perm = np.random.permutation(26)
    centers = np.empty((26, 2), np.float32)
    centers[perm] = pos
    return centers


def _polyline_letters(target: np.ndarray) -> np.ndarray:
    """Collapse adjacent repeated letters ('tt' in "letter") — zero-length
    polyline segments carry no direction and no information."""
    t = np.asarray(target)
    if len(t) <= 1:
        return t
    keep = np.ones(len(t), bool)
    keep[1:] = t[1:] != t[:-1]
    return t[keep]


def warp_path(feats: np.ndarray, target: Sequence[int], src_centers: np.ndarray,
              dst_centers: np.ndarray) -> np.ndarray:
    """Warp a cached ``[2,64]`` path for ``target`` from src onto dst geometry.

    :param feats: ``[2,64] float32`` — x row then y row (the cached featurized
        path; time-uniform samples).
    :param target: letter indices 0..25 of the word, in order (the CTC target).
    :param src_centers: ``[26,2]`` a..z key centers the path was swiped on.
    :param dst_centers: ``[26,2]`` a..z key centers to warp onto.
    :returns: warped ``[2,64] float32`` (NOT clipped — the training loop clips
        after its noise stage, matching the existing augmentation order).

    Exactness invariants (asserted by ``--selftest``):
      * ``dst_centers is src_centers``-equal → identity to float32 precision;
      * a path lying ON the source ideal polyline maps ONTO the target ideal
        polyline (zero residual in, zero residual out).
    """
    letters = _polyline_letters(target)
    P = np.stack([feats[0], feats[1]], axis=1).astype(np.float64)   # [N,2]
    n_pts = P.shape[0]
    if len(letters) == 0:
        return feats.astype(np.float32, copy=True)
    if len(letters) == 1:
        # Single-key word: the "polyline" is a point; transfer the residual
        # cloud by pure translation.
        shift = dst_centers[letters[0]].astype(np.float64) - \
            src_centers[letters[0]].astype(np.float64)
        Pd = P + shift
        return np.stack([Pd[:, 0], Pd[:, 1]]).astype(np.float32)

    A_s = src_centers[letters[:-1]].astype(np.float64)              # [S,2]
    B_s = src_centers[letters[1:]].astype(np.float64)
    d_s = B_s - A_s
    L2_s = np.maximum((d_s * d_s).sum(1), 1e-12)                    # [S]
    n_seg = A_s.shape[0]

    # Point-to-segment projection, all pairs: t[i,j] ∈ [0,1], D[i,j] = sq dist.
    diff = P[:, None, :] - A_s[None, :, :]                          # [N,S,2]
    t = np.clip((diff * d_s[None]).sum(-1) / L2_s[None], 0.0, 1.0)  # [N,S]
    proj = A_s[None] + t[..., None] * d_s[None]                     # [N,S,2]
    D = ((P[:, None, :] - proj) ** 2).sum(-1)                       # [N,S]

    # Monotone assignment: segment index never decreases along the path (the
    # finger visits the word's letters in order). Forward DP with cumulative
    # minima; backpointers recover the per-point segment. Boundary conditions
    # pin the first point to segment 0 and the last to segment S-1: a swipe
    # starts on its first key and ends on its last BY CONSTRUCTION of the task,
    # and without the pin a segment that passes over a later key of the word
    # (e.g. "has": h→a runs straight through s) absorbs the tail points and the
    # re-anchored endpoint lands wildly off the target key.
    seg_idx = np.arange(n_seg)
    F_prev = np.full(n_seg, np.inf)
    F_prev[0] = D[0, 0]
    back = np.empty((n_pts, n_seg), np.int32)
    for i in range(1, n_pts):
        cm = np.minimum.accumulate(F_prev)
        is_new = F_prev <= np.concatenate(([np.inf], cm[:-1]))
        back[i] = np.maximum.accumulate(np.where(is_new, seg_idx, 0))
        F_prev = D[i] + cm
    j = np.empty(n_pts, np.int64)
    j[-1] = n_seg - 1
    for i in range(n_pts - 2, -1, -1):
        j[i] = back[i + 1, j[i + 1]]

    # Residual in the source segment's tangent/normal frame.
    u_s = d_s / np.sqrt(L2_s)[:, None]                              # [S,2]
    v_s = np.stack([-u_s[:, 1], u_s[:, 0]], axis=1)
    tj = t[np.arange(n_pts), j]                                     # [N]
    anchor_s = A_s[j] + tj[:, None] * d_s[j]
    r = P - anchor_s
    a = (r * u_s[j]).sum(1)
    b = (r * v_s[j]).sum(1)

    # Re-anchor on the target polyline. The arc-length remap per segment is
    # piecewise linear: slope 1 within DWELL_RADIUS of either vertex (near-key
    # geometry transfers in absolute units), proportional across the middle.
    A_d = dst_centers[letters[:-1]].astype(np.float64)
    B_d = dst_centers[letters[1:]].astype(np.float64)
    d_d = B_d - A_d
    L2_d = (d_d * d_d).sum(1)
    L_s = np.sqrt(L2_s)                                             # [S]
    L_d = np.sqrt(np.maximum(L2_d, 1e-12))
    u_d = np.where((L2_d > 1e-12)[:, None],
                   d_d / L_d[:, None],
                   u_s)          # zero-length dst segment: keep the src frame
    v_d = np.stack([-u_d[:, 1], u_d[:, 0]], axis=1)
    h = np.minimum(DWELL_RADIUS, 0.5 * np.minimum(L_s, L_d))        # [S]
    s_src = tj * L_s[j]                                             # abs arc pos
    hj, Lsj, Ldj = h[j], L_s[j], L_d[j]
    mid_ratio = (Ldj - 2.0 * hj) / np.maximum(Lsj - 2.0 * hj, 1e-12)
    s_dst = np.where(
        s_src < hj, s_src,
        np.where(s_src > Lsj - hj,
                 Ldj - (Lsj - s_src),
                 hj + (s_src - hj) * mid_ratio))
    frac = s_dst / np.maximum(Ldj, 1e-12)
    Pd = A_d[j] + frac[:, None] * d_d[j] + a[:, None] * u_d[j] + b[:, None] * v_d[j]
    return np.stack([Pd[:, 0], Pd[:, 1]]).astype(np.float32)


class LayoutAugmenter:
    """Per-sample geometry resampler for ``train.py``'s ``SwipeDataset``.

    With probability ``p`` a sample is re-targeted: a geometry is drawn
    (synthetic with probability ``synth_frac``, else uniformly from the real
    pool) and the cached path is warped onto it; the sample's key centers are
    replaced by the drawn geometry. With probability ``1-p`` the sample stays
    canonical QWERTY — the fraction that guards the en_qwerty bar.

    Uses the global ``np.random`` stream (per-worker seeded by the DataLoader,
    like every other augmentation draw in ``train.py``).
    """

    def __init__(self, p: float, synth_frac: float,
                 real_centers: Sequence[np.ndarray]) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"layout-alt p must be in [0,1], got {p}")
        if not 0.0 <= synth_frac <= 1.0:
            raise ValueError(f"synth_frac must be in [0,1], got {synth_frac}")
        self.p = p
        self.synth_frac = synth_frac
        self.reals = [np.asarray(c, np.float32) for c in real_centers]
        if not self.reals and synth_frac < 1.0 and p > 0.0:
            raise ValueError("synth_frac < 1 needs a non-empty real-layout pool")

    def sample_geometry(self) -> Optional[np.ndarray]:
        """-> a ``[26,2]`` geometry to re-target this sample onto, or ``None``."""
        if self.p <= 0.0 or np.random.rand() >= self.p:
            return None
        if not self.reals or np.random.rand() < self.synth_frac:
            return synth_geometry()
        return self.reals[np.random.randint(len(self.reals))]


# ── validation ──────────────────────────────────────────────────────────────────

def endpoint_stats(feats: np.ndarray, words: Sequence[str],
                   centers: np.ndarray) -> Dict[str, float]:
    """`eval_altlayout.endpoint_proximity` on featurized ``[N,2,64]`` paths.

    The 64-point resample preserves the raw endpoints exactly, so these numbers
    are directly comparable to the raw-trace stats in ALT_LAYOUT_EVAL.md §2.
    """
    hit_s = hit_e = n = 0
    dsum_s = dsum_e = 0.0
    for f, w in zip(feats, words):
        if not w:
            continue
        n += 1
        for px, py, letter, first in ((f[0][0], f[1][0], w[0], True),
                                      (f[0][-1], f[1][-1], w[-1], False)):
            d = np.hypot(centers[:, 0] - px, centers[:, 1] - py)
            hit = int(d.argmin()) == ord(letter) - 97
            if first:
                dsum_s += float(d[ord(letter) - 97])
                hit_s += hit
            else:
                dsum_e += float(d[ord(letter) - 97])
                hit_e += hit
    return {"n": n, "start_hit": hit_s / max(n, 1), "end_hit": hit_e / max(n, 1),
            "start_d": dsum_s / max(n, 1), "end_d": dsum_e / max(n, 1)}


def ideal_path(letters: np.ndarray, centers: np.ndarray, n: int = 64) -> np.ndarray:
    """``[2,n]`` path sampled uniformly along the ideal polyline (for selftests)."""
    pts = centers[letters].astype(np.float64)                      # [L,2]
    if len(pts) == 1:
        return np.repeat(pts.T, n, axis=1).astype(np.float32)
    seg = np.diff(pts, axis=0)
    slen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(slen)])
    s = np.linspace(0.0, cum[-1], n)
    out = np.empty((n, 2))
    for k, sv in enumerate(s):
        i = min(int(np.searchsorted(cum, sv, "right")) - 1, len(seg) - 1)
        f = (sv - cum[i]) / max(slen[i], 1e-12)
        out[k] = pts[i] + f * seg[i]
    return out.T.astype(np.float32)


def _dist_to_polyline(P: np.ndarray, pts: np.ndarray) -> float:
    """Max distance from points ``P [N,2]`` to the polyline through ``pts``."""
    if len(pts) == 1:                       # single-key word: polyline is a point
        return float(np.hypot(*(P - pts[0]).T).max())
    A, B = pts[:-1], pts[1:]
    d = B - A
    L2 = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(-1) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * d[None]
    return float(np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(1).max())


def selftest() -> int:
    """Exactness invariants of the warp. Exits non-zero on failure."""
    np.random.seed(7)
    qw = load_az_centers(SCRIPT_DIR / "en_qwerty.json")
    dv = load_az_centers(LAYOUT_DIR / "futo_dvorak.json")
    words = [b"hello", b"there", b"letter", b"a", b"aa", b"minimum",
             b"acknowledgement", b"queue"]
    max_id = 0.0
    max_ideal = 0.0
    for wb in words:
        tgt = np.frombuffer(wb, np.uint8).astype(np.int64) - ord("a")
        # (1) identity: src == dst → the warp is the identity.
        path = ideal_path(_polyline_letters(tgt), qw)
        noisy = path + np.random.normal(0, 0.02, path.shape).astype(np.float32)
        back = warp_path(noisy, tgt, qw, qw)
        max_id = max(max_id, float(np.abs(back - noisy).max()))
        # (2) ideal → ideal: a zero-residual path lands on the dst polyline.
        warped = warp_path(path, tgt, qw, dv)
        dpts = dv[_polyline_letters(tgt)]
        max_ideal = max(max_ideal, _dist_to_polyline(warped.T, dpts.astype(np.float64)))
    print(f"selftest: identity max|Δ| {max_id:.2e}  "
          f"ideal→ideal max dist to dst polyline {max_ideal:.2e}")
    ok = max_id < 1e-5 and max_ideal < 1e-5
    # (3) synthetic geometries: containment + all 26 letters placed distinctly.
    for _ in range(200):
        g = synth_geometry()
        assert g.shape == (26, 2) and g.min() >= 0.0 and g.max() <= 1.0
        assert len(np.unique(np.round(g, 4), axis=0)) == 26
    print("selftest: 200 synthetic geometries contained in [0,1], 26 distinct keys")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def validate(args: argparse.Namespace) -> int:
    """Warp a val-split sample onto a target geometry; print endpoint stats."""
    from paths import DEFAULT_WORKDIR, resolve      # noqa: WPS433 (repo-local)
    workdir = args.workdir or DEFAULT_WORKDIR
    qw = load_az_centers(SCRIPT_DIR / "en_qwerty.json")
    if args.target == "synth":
        np.random.seed(args.seed)
        dst = synth_geometry()
    else:
        dst = load_az_centers(LAYOUT_DIR / f"futo_{args.target}.json")
    with np.load(resolve(workdir, Path("cache/val.npz"))) as d:
        feats = np.array(d["features"][: args.n])
        words = [str(w) for w in d["words"][: args.n]]
    tgts = [np.frombuffer(w.encode("ascii"), np.uint8).astype(np.int64) - 97
            for w in words]
    import time
    t0 = time.time()
    warped = np.stack([warp_path(f, t, qw, dst) for f, t in zip(feats, tgts)])
    dt = time.time() - t0
    src_stats = endpoint_stats(feats, words, qw)
    dst_stats = endpoint_stats(warped, words, dst)
    print(f"warped {len(words)} val paths onto '{args.target}' "
          f"({dt / len(words) * 1e3:.2f} ms/path)")
    print(f"{'':<24}{'n':>6} {'start-hit':>10} {'end-hit':>9} {'start-d':>9} {'end-d':>7}")
    for label, s in ((f"source (qwerty)", src_stats),
                     (f"warped ({args.target})", dst_stats)):
        print(f"{label:<24}{s['n']:>6} {s['start_hit']:>10.3f} {s['end_hit']:>9.3f} "
              f"{s['start_d']:>9.4f} {s['end_d']:>7.4f}")
    print("reference bands (ALT_LAYOUT_EVAL.md §2, real corpora): "
          "start-hit 0.79–0.91, end-hit 0.73–0.97, distances 0.03–0.15")
    if args.target != "synth":
        disp = float(np.hypot(*(dst - qw).T).mean())
        print(f"mean a-z key displacement vs qwerty: {disp:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="exactness invariants (identity, ideal→ideal, synth containment)")
    ap.add_argument("--validate", action="store_true",
                    help="warp val paths onto --target and print endpoint stats")
    ap.add_argument("--target", default="dvorak",
                    help="futo layout name, or 'synth' for a random synthetic")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.validate:
        return validate(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
