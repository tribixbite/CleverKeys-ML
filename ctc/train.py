#!/usr/bin/env python3
"""Train the CTC swipe encoder from scratch. Single GPU, pure fp32.

Usage:
  python train.py --run-name base
  python train.py --resume ckpt/base/last.pt --epochs 400 --run-name base

Audit fixes applied here:
  * #5  fully resumable checkpoints (model/optimizer/scheduler/step/epoch/best/
        args/RNG for torch+cuda+numpy+python), written atomically.
  * #7  bf16 autocast removed — it cost ~0.9 % per-frame argmax agreement and
        bought nothing on a 0.4 M-param model whose epochs take seconds.
  * #8  run isolation: ckpt/<run-name>/{last.pt,best.pt,metrics.jsonl}.
  * #10 epoch budget re-based on the measured ~5 s/epoch (defaults 300/40).
  * #12 warmup uses (step+1)/warmup so the first step does not run at lr 0.
  * #13 the shared affine keeps every transformed key center inside [0,1];
        centers are never clipped. Phase G replaced the rejection loop (which
        silently truncated + biased the x-scale, ALT_LAYOUT_EVAL.md §7.2b) with
        a coupled sampler that is exact by construction; --affine-sampler legacy
        reproduces the Phase A–F behaviour.
  * #14 persistent_workers + explicit NpzFile close.
  * #16 --workdir pathing; layout defaults to the script's own directory.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_ceiling import (ENC_BETA, ENC_BETA_PRUNE, ENC_GAMMA,  # noqa: E402
                                  ENC_GAMMA_PRUNE, ENC_LAMBDA, futo_viterbi_beam)
from futo_decoder_eval import (load_combined_vocab, load_layout,  # noqa: E402
                               load_test)
from layout_aug import (DEFAULT_REAL_POOL, LAYOUT_DIR, LayoutAugmenter,  # noqa: E402
                        load_az_centers, synth_geometry, warp_path)
from model import (MAX_KEYS, T_OUT, CtcSwipeEncoder,  # noqa: E402
                   encoder_from_checkpoint, slice_head_torch)
from paths import (DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve,  # noqa: E402
                   sha256_file)

BLANK = MAX_KEYS  # 64 — full-head blank index (the Kotlin slice relocates it to K)

#: Args that define the architecture; a --resume must agree on all of them.
ARCH_ARGS = ("ch", "embed_hid", "feat_version", "block", "dilations", "t_out")

#: Affine augmentation bounds (audit fix #13; Phase G samples the feasible
#: sub-region of these exactly — see SwipeDataset._sample_affine).
SCALE_LO, SCALE_HI = 0.85, 1.15
TRANS_ABS = 0.05
MIRROR_P = 0.25
AFFINE_TRIES = 10
PATH_NOISE = 0.005
CENTER_NOISE = 0.01
PERMUTE_P = 0.5


def affine_axis_bounds(centers: np.ndarray) -> List[Tuple[float, float, float]]:
    """Per-axis ``(lo, hi, s_hi)`` for the coupled affine sampler (Phase G).

    A scale about 0.5 with translate ``|t| <= TRANS_ABS`` keeps every center in
    [0,1] iff all three hold (lo/hi = min/max center on the axis):
      span:  (hi - lo) * s <= 1
      left:  (0.5 - lo) * s <= 0.5 + TRANS_ABS
      right: (hi - 0.5) * s <= 0.5 + TRANS_ABS
    Phase H computes these per sampled geometry, not just for the canonical
    layout, so the shared affine stays exact under layout resampling.
    """
    out = []
    for ax in range(2):
        lo, hi = float(centers[:, ax].min()), float(centers[:, ax].max())
        s_max = 1.0 / max(hi - lo, 1e-9)
        if lo < 0.5:
            s_max = min(s_max, (0.5 + TRANS_ABS) / (0.5 - lo))
        if hi > 0.5:
            s_max = min(s_max, (0.5 + TRANS_ABS) / (hi - 0.5))
        out.append((lo, hi, max(SCALE_LO, min(SCALE_HI, s_max))))
    return out


def load_layout_centers(path: Path) -> np.ndarray:
    """Read the canonical layout -> ``[K,2] float32`` centers in emission order.

    Emission class column ``c`` corresponds to ``letters[c]``; the layout's
    ``keys`` array is stored in that same order, and we assert it here so a
    re-ordered layout file can never silently permute the training targets.
    """
    obj = json.loads(Path(path).read_text())
    letters = list(obj["letters"])
    by_letter = {k["letter"]: (float(k["cx"]), float(k["cy"])) for k in obj["keys"]}
    key_order = [k["letter"] for k in obj["keys"]]
    if key_order != letters:
        raise SystemExit(f"{path}: keys[] order {''.join(key_order)} != letters "
                         f"{''.join(letters)}; emission columns would be permuted")
    return np.array([by_letter[c] for c in letters], np.float32)   # [K,2]


class SwipeDataset(Dataset):
    """Cached ``[2,64]`` features + slot-space CTC targets.

    :param npz_path: output of ``prepare_data.py``. A **sequence** of paths is
        concatenated in order, and a path repeated N times contributes its rows
        N times — which is how the Phase-E oversampling arms are built. Exact
        repetition through a plain without-replacement shuffle is used in
        preference to a weighted sampler, because ``prepare_data.py``'s exact
        self-dedup would silently undo duplicate rows written into a tier jsonl.
    :param centers: ``[K,2]`` canonical key centers.
    :param augment: enable affine/noise augmentation.
    :param permute: also scatter the K keys into random slots of the 64 (only
        meaningful when ``augment``). The phase-2 refinement head trains on the
        sliced 27-class view, which is defined for the canonical identity slot
        assignment only, so ``train_refine.py`` passes ``permute=False``.
    """

    def __init__(self, npz_path, centers: np.ndarray, augment: bool,
                 permute: bool = True, path_offset_sigma: float = 0.0,
                 path_scale_sigma: float = 0.0,
                 affine_sampler: str = "coupled",
                 layout_aug: "Optional[LayoutAugmenter]" = None,
                 source_centers: "Optional[List[Optional[np.ndarray]]]" = None,
                 cr_views: bool = False,
                 aug_shear: float = 0.0, aug_rot_deg: float = 0.0,
                 aug_timerev: float = 0.0, aug_maskhold: float = 0.0,
                 mask_frac: float = 0.25, mask_spans: int = 3) -> None:
        paths = [npz_path] if isinstance(npz_path, (str, Path)) else list(npz_path)
        feats, tgts, tlens = [], [], []
        for p in paths:
            with np.load(p) as d:                       # audit fix #14: close handle
                feats.append(np.array(d["features"]))   # [N,2,64]
                tgts.append(np.array(d["targets"]))
                tlens.append(np.array(d["target_lengths"]))
        # Phase J: per-source geometry. Entry i of *source_centers* (or the
        # canonical *centers* when None) is the geometry rows of npz i live on —
        # what admits real alt-layout pools and non-Latin scripts into one run.
        # Row -> source is recovered by searchsorted over the cumulative counts.
        per_src = source_centers or [None] * len(paths)
        if len(per_src) != len(paths):
            raise ValueError("source_centers must match npz_path list length")
        self._src_end = np.cumsum([len(t) for t in tlens])
        self.src_centers = [centers if c is None else c for c in per_src]
        self._src_bounds = [affine_axis_bounds(c) for c in self.src_centers]
        self.features = feats[0] if len(feats) == 1 else np.concatenate(feats)
        self.tgt_flat = tgts[0] if len(tgts) == 1 else np.concatenate(tgts)
        self.tgt_len = tlens[0] if len(tlens) == 1 else np.concatenate(tlens)
        self.sources = paths
        self.tgt_off = np.concatenate([[0], np.cumsum(self.tgt_len)])
        self.centers = centers                          # [K,2] canonical
        self.augment = augment
        self.permute = permute
        self.path_offset_sigma = path_offset_sigma
        self.path_scale_sigma = path_scale_sigma
        self.k = centers.shape[0]
        if affine_sampler not in ("coupled", "legacy"):
            raise ValueError(f"affine_sampler must be 'coupled' or 'legacy', "
                             f"got {affine_sampler!r}")
        self.affine_sampler = affine_sampler
        # Phase H: optional per-sample geometry resampler (layout_aug.py). The
        # canonical bounds are precomputed; alternative geometries get theirs
        # computed per draw (26x2 min/max — negligible).
        self.layout_aug = layout_aug
        # Phase J levers (all default-off => the pre-J RNG stream is untouched):
        # CR-CTC dual views (shared layout draw + slot permutation, independent
        # affine/noise/masking per view — RESEARCH_SCAN.md Spec A) and the
        # FUTO-parity augmentations (Spec B).
        self.cr_views = cr_views
        self.aug_shear = aug_shear
        self.aug_rot_deg = aug_rot_deg
        self.aug_timerev = aug_timerev
        self.aug_maskhold = aug_maskhold
        self.mask_frac = mask_frac
        self.mask_spans = mask_spans
        # Phase G: per-axis feasible scale ceiling for the coupled sampler; see
        # affine_axis_bounds. For en_qwerty x (cx 0.05..0.95) the span bound
        # binds at s = 1/0.90 = 1.111 — the nominal SCALE_HI = 1.15 is
        # geometrically infeasible in x on this layout under center containment,
        # at ANY translate. The coupled sampler realizes
        # U(SCALE_LO, min(SCALE_HI, s_max)) exactly, instead of the legacy
        # rejection loop whose survivors were biased toward compression
        # (realized sx mean 0.955, ALT_LAYOUT_EVAL.md §7.2b).
        self._axis_bounds = affine_axis_bounds(centers)

    def __len__(self) -> int:
        return len(self.tgt_len)

    def _sample_affine(self, centers: np.ndarray,
                       axis_bounds: List[Tuple[float, float, float]]
                       ) -> Tuple[float, float, float, float, bool]:
        """Sample an affine that keeps every key center in [0,1].

        ``coupled`` (Phase G, the default): per axis, draw the scale uniformly
        over the *feasible* range precomputed in ``__init__``, then draw the
        translate uniformly over the exact interval that keeps the transformed
        centers inside [0,1], intersected with the nominal ±``TRANS_ABS`` window.
        Acceptance is 1.0 by construction and the realized scale is uniform over
        the feasible range — no compression bias, no identity fallback.

        ``legacy`` (pre-G, kept for reproduction of Phase A–F runs):
        rejection-sample scale and translate independently; fall back to the
        identity affine after ``AFFINE_TRIES`` failures (audit fix #13). On
        en_qwerty this silently truncated and biased the x-scale (accepted sx
        mean 0.955, max 1.111, 31.5 % first-draw rejects — ALT_LAYOUT_EVAL.md
        §7.2b).

        Mirroring maps [0,1] onto itself and never affects acceptance.
        """
        if self.affine_sampler == "coupled":
            out = []
            for lo, hi, s_hi in axis_bounds:
                s = float(np.random.uniform(SCALE_LO, s_hi))
                t_lo = max(-TRANS_ABS, -((lo - 0.5) * s + 0.5))
                t_hi = min(TRANS_ABS, 0.5 - (hi - 0.5) * s)
                if t_hi < t_lo:                     # degenerate layout guard
                    t_lo = t_hi = 0.5 * (t_lo + t_hi)
                out.append((s, float(np.random.uniform(t_lo, t_hi))))
            (sx, tx), (sy, ty) = out
            return sx, sy, tx, ty, bool(np.random.rand() < MIRROR_P)
        cx, cy = centers[:, 0], centers[:, 1]
        for _ in range(AFFINE_TRIES):
            sx, sy = np.random.uniform(SCALE_LO, SCALE_HI, 2)
            tx, ty = np.random.uniform(-TRANS_ABS, TRANS_ABS, 2)
            nx = (cx - 0.5) * sx + 0.5 + tx
            ny = (cy - 0.5) * sy + 0.5 + ty
            if nx.min() >= 0.0 and nx.max() <= 1.0 and ny.min() >= 0.0 and ny.max() <= 1.0:
                return float(sx), float(sy), float(tx), float(ty), bool(np.random.rand() < MIRROR_P)
        return 1.0, 1.0, 0.0, 0.0, bool(np.random.rand() < MIRROR_P)

    def _augment_view(self, feats: np.ndarray, base_centers: np.ndarray,
                      axis_bounds) -> Tuple[np.ndarray, np.ndarray]:
        """One independently-augmented view of a sample -> ``(feats, centers)``.

        Draw order for the default flag set is IDENTICAL to the pre-J code
        (affine -> C1 jitter -> path noise -> center noise); the Phase-J levers
        (shear / rotation / frame-hold masking) consume RNG only when enabled.
        """
        feats = feats.copy()
        centers = base_centers.copy()
        sx, sy, tx, ty, mirror = self._sample_affine(base_centers, axis_bounds)
        for arr_x, arr_y in ((feats[0], feats[1]),
                             (centers[:, 0], centers[:, 1])):
            arr_x[:] = (arr_x - 0.5) * sx + 0.5 + tx
            arr_y[:] = (arr_y - 0.5) * sy + 0.5 + ty
            if mirror:
                arr_x[:] = 1.0 - arr_x
        if self.aug_shear > 0.0:
            # Shared-frame shear x += k*(y-0.5) (RESEARCH_SCAN Spec B1). The
            # feasible k range keeping every center in [0,1] is exact:
            # per key, 0 <= cx + k*(cy-0.5) <= 1 bounds k on each side of
            # cy = 0.5; intersect with the nominal ±aug_shear window.
            k_lo, k_hi = -self.aug_shear, self.aug_shear
            cx, cy = centers[:, 0], centers[:, 1]
            dy = cy - 0.5
            for j in np.nonzero(np.abs(dy) > 1e-9)[0]:
                a = (0.0 - cx[j]) / dy[j]
                b = (1.0 - cx[j]) / dy[j]
                k_lo = max(k_lo, min(a, b))
                k_hi = min(k_hi, max(a, b))
            k = float(np.random.uniform(k_lo, k_hi)) if k_hi > k_lo else 0.0
            feats[0] += k * (feats[1] - 0.5)
            centers[:, 0] += k * dy
        if self.aug_rot_deg > 0.0:
            # In-bounds rotation about (0.5, 0.5) (Spec B2): rejection-sampled
            # with an identity fallback, mirroring the legacy affine discipline.
            for _ in range(AFFINE_TRIES):
                th = np.deg2rad(np.random.uniform(-self.aug_rot_deg,
                                                  self.aug_rot_deg))
                c, s = np.cos(th), np.sin(th)
                rx = (centers[:, 0] - 0.5) * c - (centers[:, 1] - 0.5) * s + 0.5
                ry = (centers[:, 0] - 0.5) * s + (centers[:, 1] - 0.5) * c + 0.5
                if rx.min() >= 0 and rx.max() <= 1 and ry.min() >= 0 and ry.max() <= 1:
                    fx = (feats[0] - 0.5) * c - (feats[1] - 0.5) * s + 0.5
                    fy = (feats[0] - 0.5) * s + (feats[1] - 0.5) * c + 0.5
                    feats[0], feats[1] = fx, fy
                    centers[:, 0], centers[:, 1] = rx, ry
                    break
        if self.path_offset_sigma > 0.0 or self.path_scale_sigma > 0.0:
            # Independent path-vs-layout misalignment (Phase C1). The shared
            # affine above moves the path AND the key centers together, so the
            # model never sees the two frames disagree — yet in the wild they
            # do: the HWS half sits ~0.064 off the FUTO half in y against the
            # same layout. Perturbing the path alone, with the keys untouched,
            # is the only augmentation that trains that tolerance.
            jx = 1.0 + np.random.normal(0.0, self.path_scale_sigma)
            jy = 1.0 + np.random.normal(0.0, self.path_scale_sigma)
            ox = np.random.normal(0.0, self.path_offset_sigma)
            oy = np.random.normal(0.0, self.path_offset_sigma)
            feats[0] = (feats[0] - 0.5) * jx + 0.5 + ox
            feats[1] = (feats[1] - 0.5) * jy + 0.5 + oy
        feats += np.random.normal(0.0, PATH_NOISE, feats.shape).astype(np.float32)
        centers += np.random.normal(0.0, CENTER_NOISE, centers.shape).astype(np.float32)
        np.clip(feats, 0.0, 1.0, out=feats)     # path only; centers stay exact
        if self.aug_maskhold > 0.0 and np.random.rand() < self.aug_maskhold:
            feats = self._mask_hold(feats)
        return feats, centers

    def _mask_hold(self, feats: np.ndarray) -> np.ndarray:
        """Frame-hold temporal masking (Spec A): 1..mask_spans random spans,
        total <= mask_frac of the input frames, each span frozen at the last
        coordinate before it (a hold reads as a stall; a zero would be a legal
        position in the corner)."""
        t = feats.shape[1]
        budget = int(self.mask_frac * t)
        n_spans = int(np.random.randint(1, self.mask_spans + 1))
        for _ in range(n_spans):
            if budget <= 1:
                break
            span = int(np.random.randint(1, budget + 1))
            a = int(np.random.randint(0, t - span + 1))
            hold = feats[:, a - 1] if a > 0 else feats[:, min(a + span, t - 1)]
            feats[:, a:a + span] = hold[:, None]
            budget -= span
        return feats

    def __getitem__(self, i: int):
        feats = self.features[i].astype(np.float32).copy()          # [2,64]
        target = self.tgt_flat[self.tgt_off[i]:self.tgt_off[i + 1]].copy()
        si = int(np.searchsorted(self._src_end, i, side="right"))
        base_centers = self.src_centers[si]
        axis_bounds = self._src_bounds[si]
        k = base_centers.shape[0]

        if self.augment and self.aug_timerev > 0.0 and \
                np.random.rand() < self.aug_timerev:
            # Time reversal WITH reversed target (Spec B3) — applied before the
            # layout warp, whose polyline correspondence is direction-agnostic.
            feats = feats[:, ::-1].copy()
            target = target[::-1].copy()

        # Phase H: with probability p, re-target the sample onto an alternative
        # geometry — warp the cached path onto the same word's key centers there
        # and swap the key centers the model is shown. Runs BEFORE the shared
        # affine, so the affine/noise/permutation stack applies identically to
        # canonical and resampled geometries. Sources whose alphabet is not the
        # 26-key a-z contract (ru, toki pona) skip the resampler — its synth and
        # real pools are a-z geometries.
        if self.augment and self.layout_aug is not None and k == 26:
            alt = self.layout_aug.sample_geometry()
            if alt is not None:
                feats = warp_path(feats, target, base_centers, alt)
                base_centers = alt
                axis_bounds = affine_axis_bounds(alt)

        if not self.augment:
            centers = base_centers.copy()
            keys = np.zeros((MAX_KEYS, 2), np.float32)
            mask = np.zeros((MAX_KEYS,), bool)
            slots = np.arange(k)
            keys[slots] = centers
            mask[slots] = True
            target_slots = slots[target]
            return (torch.from_numpy(feats), torch.from_numpy(keys),
                    torch.from_numpy(mask),
                    torch.from_numpy(target_slots.astype(np.int64)))

        # CR-CTC (Spec A): the layout draw above and the slot permutation below
        # are SHARED by both views — emission columns must mean the same key in
        # both — while affine/noise/masking are drawn independently per view.
        views = [self._augment_view(feats, base_centers, axis_bounds)]
        if self.cr_views:
            views.append(self._augment_view(feats, base_centers, axis_bounds))

        # Slot assignment: identity (the inference-time layout) or a random
        # permutation into the 64 slots, which forces the model to read key
        # geometry rather than slot index.
        mask = np.zeros((MAX_KEYS,), bool)
        if self.permute and np.random.rand() < PERMUTE_P:
            slots = np.random.permutation(MAX_KEYS)[:k]
        else:
            slots = np.arange(k)
        mask[slots] = True
        target_slots = slots[target]                    # letters -> slot indices

        packed = []
        for f, c in views:
            keys = np.zeros((MAX_KEYS, 2), np.float32)
            keys[slots] = c
            packed.append((f, keys))
        if not self.cr_views:
            f, keys = packed[0]
            return (torch.from_numpy(f), torch.from_numpy(keys),
                    torch.from_numpy(mask),
                    torch.from_numpy(target_slots.astype(np.int64)))
        return (torch.from_numpy(np.stack([p[0] for p in packed])),   # [2,2,64]
                torch.from_numpy(np.stack([p[1] for p in packed])),   # [2,64,2]
                torch.from_numpy(mask),
                torch.from_numpy(target_slots.astype(np.int64)))


def collate(batch):
    """Stack fixed-size tensors and flatten the ragged CTC targets.

    Handles both the single-view items (``feats [2,64]``) and the CR-CTC
    dual-view items (``feats [2,2,64]``, ``keys [2,64,2]``) — the loop branches
    on the stacked tensor's rank.
    """
    feats = torch.stack([b[0] for b in batch])
    keys = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    tlens = torch.tensor([len(b[3]) for b in batch], dtype=torch.long)
    targets = torch.cat([b[3] for b in batch])
    return feats, keys, mask, targets, tlens


@torch.no_grad()
def greedy_accuracy(model: torch.nn.Module, loader: DataLoader, device: str,
                    forward=None, blank: int = BLANK) -> float:
    """Val metric: greedy-CTC collapse == target word.

    FUTO-floor anchor: ~44 % greedy on test-2400 corresponded to 79.25 % beam
    top-1, so a plateau near 40 % is expected, not a failure.

    :param model: module toggled into eval mode for the sweep.
    :param forward: ``(feats, keys, mask) -> log_probs [B,T,C]``; defaults to the
        base encoder's first output. ``train_refine.py`` passes the refinement
        head so the same collapse logic serves both heads.
    :param blank: blank class index in the returned ``log_probs`` (64 for the
        full head, 26 for the refined sliced view).
    """
    was_training = model.training
    model.eval()
    if forward is None:
        def forward(f, k, m):
            return model(f, k, m)[0]
    hit = n = 0
    for feats, keys, mask, targets, tlens in loader:
        log_p = forward(feats.to(device), keys.to(device), mask.to(device))
        am = log_p.argmax(-1).cpu().numpy()             # [B,T]
        off = 0
        for b in range(am.shape[0]):
            tgt = targets[off:off + tlens[b]].numpy().tolist()
            off += int(tlens[b])
            out, prev = [], -1
            for c in am[b]:
                c = int(c)
                if c != prev and c != blank:
                    out.append(c)
                prev = c
            hit += int(out == tgt)
            n += 1
    if was_training:
        model.train()
    return hit / max(n, 1)


# ── beam-t1 checkpoint selection (Phase D) ─────────────────────────────────────
#
# Phase B measured greedy accuracy moving +5.16 pt while beam top-1 moved −1.38 pt
# on the same arm, so `best val_greedy` selects checkpoints by a metric that
# ANTI-CORRELATES with the metric that ships. From Phase D on, `best.pt` is chosen
# by lexicon-beam top-1 over a fixed val prefix.
#
# Worker globals: populated once in the parent and inherited through fork, so the
# 147 k-word trie is never pickled and never copied (copy-on-write). The emission
# buffer is a shared RawArray the parent overwrites in place at every validation,
# which is what lets the pool be forked ONCE — before CUDA is initialised — and
# reused for the whole run.
_BV_TRIE = None
_BV_LETTERS: List[str] = []
_BV_TARGETS: List[str] = []
_BV_EM: Optional[np.ndarray] = None
_BV_WIDTH = 100
#: ``(gamma, lambda, beta, gammaPrune, betaPrune)`` the selection beam scores at.
_BV_PRESET = (ENC_GAMMA, ENC_LAMBDA, ENC_BETA, ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)


def _beam_init(trie, letters: List[str], targets: List[str], shm, n: int,
               width: int, preset: Tuple[float, float, float, float, float],
               t_out: int = T_OUT) -> None:
    global _BV_TRIE, _BV_LETTERS, _BV_TARGETS, _BV_EM, _BV_WIDTH, _BV_PRESET
    _BV_TRIE, _BV_LETTERS, _BV_TARGETS, _BV_WIDTH = trie, letters, targets, width
    _BV_PRESET = preset
    _BV_EM = np.frombuffer(shm, np.float32).reshape(n, t_out, len(letters) + 1)


def _beam_chunk(bounds: Tuple[int, int]) -> Tuple[int, int, int]:
    """Decode rows ``[lo,hi)`` from the shared buffer -> ``(hits1, hits3, hits5)``.

    The beam is called with :data:`_BV_PRESET` and ``top_k=5``, which is exactly
    what ``eval_beam.py`` does; ``sweep_scoring``'s raw-score trick is not needed
    here because the preset is fixed, and ``top_k=5`` is cheaper than returning
    the whole terminal beam.
    """
    lo, hi = bounds
    n_letters = len(_BV_LETTERS)
    g, lam, b, gp, bp = _BV_PRESET
    h1 = h3 = h5 = 0
    for i in range(lo, hi):
        cands = futo_viterbi_beam(_BV_EM[i], _BV_LETTERS, n_letters, _BV_TRIE,
                                  _BV_WIDTH, 5, g, lam, b, gp, bp)
        words = [w for w, _ in cands]
        tgt = _BV_TARGETS[i]
        try:
            r = words.index(tgt)
        except ValueError:
            continue
        h1 += r < 1
        h3 += r < 3
        h5 += r < 5
    return h1, h3, h5


def _probe_centers(spec: str) -> np.ndarray:
    """Selection-probe spec -> ``[26,2]`` a-z centers.

    ``synth:<seed>`` draws a deterministic :func:`synth_geometry` (the global
    numpy stream is saved and restored, so the training RNG trajectory is
    untouched and paired runs with/without probes stay comparable); any other
    name loads ``layouts/futo_<name>.json``. dvorak is refused — it is the
    held-out transfer probe and selecting checkpoints on it would convert the
    transfer eval into a fitting exercise (same argument as PHASE_H §1).
    """
    if spec.startswith("synth:"):
        state = np.random.get_state()
        np.random.seed(int(spec.split(":", 1)[1]))
        try:
            return synth_geometry()
        finally:
            np.random.set_state(state)
    if spec == "dvorak":
        raise SystemExit("--select-layout-probes: dvorak is the held-out "
                         "transfer probe; selecting on it voids that eval")
    return load_az_centers(LAYOUT_DIR / f"futo_{spec}.json")


class BeamValidator:
    """Lexicon-beam top-1/3/5 over a fixed val prefix, run in-process during training.

    :param cache_dir: holds ``val.npz`` (the featurized canonical val split, in
        jsonl row order — asserted against the jsonl here so a re-prepared cache
        cannot silently misalign the targets).
    :param n_rows: prefix length. 2,000 rows carry a binomial SE of ~0.87 pt on a
        ~80 % rate, which is noisy in absolute terms but is used only to rank
        checkpoints *within* one run, where the comparison is paired.
    :param jobs: fork-pool size. The pool is created once, at construction time,
        and reused for every validation.
    :param probes: Phase I: selection-probe layout specs (``synth:<seed>`` or a
        futo layout name). For each probe the first ``probe_rows`` a-z-clean val
        prefix rows are warped onto that geometry (:func:`layout_aug.warp_path`)
        once, at construction; every validation then also decodes them with the
        probe's key centers, and :attr:`score` blends the canonical and probe
        top-1 rates. PHASE_H §6.2 documented that QWERTY-only selection does not
        order checkpoints by transfer accuracy; this is the closing of that gap.
    :param probe_weight: ``score = (t1_qwerty + w * mean(probe_t1)) / (1 + w)``.
    """

    def __init__(self, workdir: Path, layout: Path, cache_dir: Path,
                 val_jsonl: Path, vocab: Path, n_rows: int, beam_width: int,
                 jobs: int,
                 preset: Tuple[float, float, float, float, float] = _BV_PRESET,
                 t_out: int = T_OUT,
                 probes: Optional[List[str]] = None, probe_rows: int = 2000,
                 probe_weight: float = 1.0) -> None:
        letters, centers = load_layout(layout)
        rows = load_test(val_jsonl)
        with np.load(cache_dir / "val.npz") as d:
            feats = np.array(d["features"])
            cached_words = [str(w) for w in d["words"]]
        if len(feats) != len(rows):
            raise SystemExit(
                f"beam-val: val.npz has {len(feats)} rows but {val_jsonl.name} has "
                f"{len(rows)}; re-run prepare_data.py (row order must match)")
        self.n = min(n_rows, len(rows))
        self.t_out = t_out
        # Targets are the jsonl words lowercased — identical to eval_arms.py, so
        # the selection metric and the reported metric are the same quantity.
        self.targets = [w.lower() for w, _, _, _ in rows[:self.n]]
        mismatch = sum(1 for a, b in zip(self.targets, cached_words[:self.n])
                       if a != b)
        self.n_letters = len(letters)
        self.probe_weight = probe_weight
        self.probe_t1: Dict[str, float] = {}
        self.score = float("nan")

        def pack(c26: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            keys = np.zeros((MAX_KEYS, 2), np.float32)
            keys[:self.n_letters] = c26
            mask = np.zeros((MAX_KEYS,), bool)
            mask[:self.n_letters] = True
            return torch.from_numpy(keys)[None], torch.from_numpy(mask)[None]

        # Segments: (name, features [m,2,64], keys [1,64,2], mask [1,64], targets).
        qk, qm = pack(centers)
        segs = [("qwerty", torch.from_numpy(feats[:self.n]), qk, qm, self.targets)]
        if probes:
            # a-z-clean rows only: warp_path needs letter indices 0..25.
            clean = [i for i in range(self.n)
                     if self.targets[i] and
                     all("a" <= c <= "z" for c in self.targets[i])][:probe_rows]
            if len(clean) < probe_rows:
                print(f"beam-val: only {len(clean)} a-z-clean rows for the "
                      f"layout probes (asked {probe_rows})")
            src26 = np.asarray(centers, np.float32)[:26]
            for spec in probes:
                c26 = _probe_centers(spec)
                warped = np.stack([
                    warp_path(feats[i].astype(np.float32),
                              np.frombuffer(self.targets[i].encode(), np.uint8)
                              .astype(np.int64) - 97, src26, c26)
                    for i in clean])
                pk, pm = pack(c26)
                segs.append((spec, torch.from_numpy(warped), pk, pm,
                             [self.targets[i] for i in clean]))
        self.segments = []          # (name, lo, hi, features, keys, mask)
        all_targets: List[str] = []
        off = 0
        for name, f, k, m, tg in segs:
            self.segments.append((name, off, off + len(tg), f, k, m))
            all_targets.extend(tg)
            off += len(tg)
        self.n_total = off

        trie = load_combined_vocab(vocab)
        self._shm = mp.RawArray("f", self.n_total * t_out * (self.n_letters + 1))
        self._view = np.frombuffer(self._shm, np.float32).reshape(
            self.n_total, t_out, self.n_letters + 1)
        # Chunks never span a segment boundary, so per-segment hit counts are a
        # sum over whole chunks.
        self.chunks: List[Tuple[int, int]] = []
        self._chunk_seg: List[int] = []
        for si, (_, lo, hi, _, _, _) in enumerate(self.segments):
            m = hi - lo
            step = max(1, (m + jobs - 1) // jobs)
            for a in range(lo, hi, step):
                self.chunks.append((a, min(a + step, hi)))
                self._chunk_seg.append(si)
        ctx = mp.get_context("fork")
        self.pool = ctx.Pool(jobs, initializer=_beam_init,
                             initargs=(trie, letters, all_targets, self._shm,
                                       self.n_total, beam_width, preset, t_out))
        print(f"beam-val: {self.n} rows"
              + (f" + {self.n_total - self.n} layout-probe rows "
                 f"({', '.join(p for p in probes)}; weight {probe_weight})"
                 if probes else "")
              + f", trie {trie.num_words} words, width {beam_width}, {jobs} "
              f"procs, preset {preset}"
              + (f"  ({mismatch} targets differ from the a-z-normalised cache)"
                 if mismatch else ""))

    @torch.no_grad()
    def run(self, model: torch.nn.Module, device: str, batch: int = 512,
            forward=None) -> Tuple[float, float, float]:
        """-> ``(t1, t3, t5)`` percentages on the canonical prefix.

        Side effects: :attr:`probe_t1` holds per-probe top-1 and :attr:`score`
        the blended selection score (= canonical t1 when no probes are set).

        :param forward: ``(feats, keys, mask) -> sliced log-probs [B,T',K+1]``.
            Defaults to the base encoder's full head, sliced. ``train_refine.py``
            passes the refinement head so the head is selected on the same
            lexicon-beam top-1 the encoder arms are, rather than on greedy.
        """
        was_training = model.training
        model.eval()
        for _, lo, hi, feats, keys, mask in self.segments:
            for s in range(0, hi - lo, batch):
                f = feats[s:s + batch].to(device)
                b = f.shape[0]
                k = keys.to(device).expand(b, -1, -1)
                m = mask.to(device).expand(b, -1)
                if forward is None:
                    sliced = slice_head_torch(model(f, k, m)[0], self.n_letters)
                else:
                    sliced = forward(f, k, m)                      # [b,T',27]
                self._view[lo + s:lo + s + b] = sliced.detach().cpu().numpy()
        if was_training:
            model.train()
        seg_hits = [[0, 0, 0] for _ in self.segments]
        for (a, b_, c), si in zip(self.pool.map(_beam_chunk, self.chunks),
                                  self._chunk_seg):
            seg_hits[si][0] += a
            seg_hits[si][1] += b_
            seg_hits[si][2] += c
        h1, h3, h5 = seg_hits[0]
        t1 = h1 / self.n * 100.0
        self.probe_t1 = {}
        for (name, lo, hi, _, _, _), (p1, _, _) in zip(self.segments[1:],
                                                       seg_hits[1:]):
            self.probe_t1[name] = p1 / (hi - lo) * 100.0
        if self.probe_t1:
            pm = sum(self.probe_t1.values()) / len(self.probe_t1)
            self.score = (t1 + self.probe_weight * pm) / (1.0 + self.probe_weight)
        else:
            self.score = t1
        return (t1, h3 / self.n * 100.0, h5 / self.n * 100.0)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


class ModelEMA:
    """Exponential moving average of the model's float state (Phase C2).

    Evaluated and exported in place of the live weights: the average of the last
    few thousand steps is a flatter, lower-variance point than whichever step the
    validation grid happened to land on, which matters here because Phase B showed
    checkpoint choice alone moves beam top-1 by ~0.5 pt.

    The decay is warmed up as ``min(decay, (1+step)/(10+step))`` so the average is
    not anchored to the random initialisation for its first few hundred steps.
    Integer buffers are copied, not averaged.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() if v.is_floating_point()
                       else v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        """Load the averaged weights into *model* (used for the val pass)."""
        model.load_state_dict({k: v.to(dtype=p.dtype) if p.is_floating_point() else v
                               for (k, v), p in zip(self.shadow.items(),
                                                    model.state_dict().values())})


# ── checkpointing (audit fix #5) ────────────────────────────────────────────────

def rng_state() -> Dict[str, object]:
    """Capture every RNG stream the training loop consumes.

    numpy's MT19937 key is converted to a plain int list so the checkpoint stays
    loadable under torch's ``weights_only=True`` default.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [np_state[0], [int(v) for v in np_state[1]],
                  int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "python": [py_state[0], [int(v) for v in py_state[1]], py_state[2]],
    }


def restore_rng(state: Dict[str, object]) -> None:
    """Restore the streams captured by :func:`rng_state`."""
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                        else state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except (RuntimeError, ValueError) as e:   # different GPU count than the saver
            print(f"[resume] cuda RNG not restored ({e}); continuing")
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    p = state["python"]
    random.setstate((p[0], tuple(p[1]), p[2]))


def atomic_save(payload: Dict[str, object], path: Path) -> None:
    """Write a checkpoint via tmp file + ``os.replace`` so a crash can't truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_checkpoint(model, opt, sched, step: int, epoch: int, best: float,
                     best_epoch: int, args: argparse.Namespace,
                     val_greedy: float, ema: "Optional[ModelEMA]" = None,
                     val_beam: "Optional[Tuple[float, float, float]]" = None,
                     select_metric: str = "val_greedy") -> Dict[str, object]:
    """Assemble the full resumable checkpoint payload.

    When EMA is on, ``model`` holds the AVERAGED weights — that is what every
    downstream consumer (export, eval, latency) should see, and it is what the
    reported val number was measured on. The live weights are kept under
    ``model_raw`` so a resume continues the actual trajectory rather than the
    average.

    ``best``/``best_epoch`` track whichever metric ``select_metric`` names — from
    Phase D that is ``val_beam_t1`` (percent), not ``val_greedy`` (fraction), so
    the field is recorded explicitly rather than left to be inferred from scale.
    """
    payload = {
        "model": (ema.state_dict() if ema is not None else model.state_dict()),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "epoch": epoch,
        "best": best,
        "best_epoch": best_epoch,
        "val_greedy": val_greedy,
        "select_metric": select_metric,
        "val_beam_t1": float(val_beam[0]) if val_beam else float("nan"),
        "val_beam_t3": float(val_beam[1]) if val_beam else float("nan"),
        "val_beam_t5": float(val_beam[2]) if val_beam else float("nan"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rng": rng_state(),
        # Convenience duplicates so eval/export can read the arch without --args.
        "ch": args.ch,
        "embed_hid": args.embed_hid,
        "feat_version": args.feat_version,
        "block": args.block,
        "dilations": tuple(int(v) for v in args.dilations.split(",")),
        "t_out": getattr(args, "t_out", T_OUT),
    }
    if ema is not None:
        payload["model_raw"] = model.state_dict()
        payload["ema_decay"] = ema.decay
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--train-npz", default="train.npz", dest="train_npz",
                    help="training cache(s) inside --cache (tier arm selector). "
                         "Comma-separated names are concatenated, and a repeated "
                         "name repeats its rows — e.g. "
                         "'train_t3.npz,train_t3hws.npz,train_t3hws.npz' is T3 "
                         "with its HWS half oversampled 3x")
    ap.add_argument("--run-name", default="", dest="run_name",
                    help="ckpt/<run-name>/ (default: timestamped)")
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--embed-hid", type=int, default=96, dest="embed_hid")
    ap.add_argument("--feat-version", type=int, default=1, choices=(1, 2),
                    dest="feat_version",
                    help="1 = 8 path channels; 2 = 14 kinematic channels + a "
                         "learned reduction of the key-proximity field (B1)")
    ap.add_argument("--block", default="res",
                    choices=("res", "convnext", "dwsep", "resbn"),
                    help="trunk block: original residual, depthwise/GLU/GRN/SE (B2), "
                         "the Phase-F depthwise-separable block (dwsep), or the "
                         "original dense block with foldable BatchNorms (resbn). "
                         "Both Phase-F blocks export with zero normalization nodes.")
    ap.add_argument("--dilations", default="1,2,4,8",
                    help="comma-separated dilation per trunk block")
    ap.add_argument("--t-out", type=int, default=T_OUT, choices=(32, 64),
                    dest="t_out",
                    help="Phase I probe: emission frame count. 32 = the shipped "
                         "contract (stride-2 stem); 64 = stride-1 stem, doubled "
                         "emission resolution. 64 is CONTRACT-BREAKING — "
                         "measurement only, the app decides any move")
    ap.add_argument("--short-loss-weight", type=float, default=1.0,
                    dest="short_loss_weight",
                    help="Phase K4: multiply the per-sample CTC loss of short "
                         "targets (len <= 3) by this factor before the batch "
                         "mean (weighted mean, so 1.0 is EXACTLY the default "
                         "reduction). Aims training capacity at the one val "
                         "stratum the finalist misses. Incompatible with "
                         "--cr-alpha.")
    ap.add_argument("--total-steps", type=int, default=0, dest="total_steps",
                    help="step-equalized budget: cosine horizon and stopping are "
                         "measured in optimizer steps, not epochs (0 = use --epochs)")
    ap.add_argument("--val-every", type=int, default=0, dest="val_every",
                    help="validate/checkpoint every N steps (0 = once per epoch)")
    ap.add_argument("--path-offset-sigma", type=float, default=0.0,
                    dest="path_offset_sigma",
                    help="C1: per-trace gaussian offset applied to the PATH only, "
                         "keys untouched (0 = off)")
    ap.add_argument("--path-scale-sigma", type=float, default=0.0,
                    dest="path_scale_sigma",
                    help="C1: per-trace gaussian scale jitter on the PATH only")
    ap.add_argument("--affine-sampler", default="coupled",
                    choices=("coupled", "legacy"), dest="affine_sampler",
                    help="Phase G: 'coupled' samples the shared-affine scale "
                         "uniformly over the per-axis feasible range and couples "
                         "the translate to it (acceptance 1.0, no compression "
                         "bias); 'legacy' is the pre-G rejection loop that "
                         "silently truncated sx (ALT_LAYOUT_EVAL.md §7.2b), kept "
                         "for reproducing Phase A–F arms")
    ap.add_argument("--layout-alt-p", type=float, default=0.0, dest="layout_alt_p",
                    help="Phase H: probability a training sample is re-targeted "
                         "onto an alternative keyboard geometry (path warped via "
                         "layout_aug.warp_path, key centers swapped). 0 = off. "
                         "The 1-p remainder stays canonical QWERTY")
    ap.add_argument("--layout-synth-frac", type=float, default=2.0 / 3.0,
                    dest="layout_synth_frac",
                    help="Phase H: within re-targeted samples, fraction drawn "
                         "from synth_geometry() (random letter arrangement on a "
                         "plausible 3/4-row lattice); the rest come uniformly "
                         "from --layout-alt-real")
    ap.add_argument("--layout-alt-real", default=",".join(DEFAULT_REAL_POOL),
                    dest="layout_alt_real",
                    help="comma-separated futo layout names for the real pool. "
                         "dvorak is EXCLUDED by default — it is the held-out "
                         "transfer probe (ALT_LAYOUT_EVAL.md) — and qwerty is "
                         "the canonical geometry. Empty = synthetic only")
    ap.add_argument("--train-layouts", default="", dest="train_layouts",
                    help="Phase J: comma-separated per---train-npz layout files "
                         "(same count/order; empty entry = the canonical "
                         "--layout). A pool's rows are trained on its own "
                         "geometry — real alt-layout data and non-Latin scripts "
                         "join one run this way. Files may be letters-ordered "
                         "(en_qwerty.json style) or FUTO cx/cy layouts (a-z "
                         "extracted)")
    ap.add_argument("--cr-alpha", type=float, default=0.0, dest="cr_alpha",
                    help="CR-CTC consistency weight (RESEARCH_SCAN Spec A; "
                         "paper optimum 0.2). >0 trains two independently "
                         "augmented views per sample (shared layout draw + slot "
                         "permutation) with a symmetric stop-grad frame KL; "
                         "~2x GPU/step. 0 = off")
    ap.add_argument("--cr-mask-frac", type=float, default=0.25, dest="cr_mask_frac",
                    help="temporal frame-hold masking budget (fraction of the 64 "
                         "input frames) applied per CR view / by --aug-maskhold")
    ap.add_argument("--cr-mask-spans", type=int, default=3, dest="cr_mask_spans",
                    help="max random spans the masking budget is split across")
    ap.add_argument("--aug-shear", type=float, default=0.0, dest="aug_shear",
                    help="FUTO-parity shear (Spec B1): x += k*(y-0.5), k uniform "
                         "over ±this ∩ the exact center-containment range. 0=off")
    ap.add_argument("--aug-rot", type=float, default=0.0, dest="aug_rot",
                    help="FUTO-parity rotation (Spec B2): degrees, uniform ±this "
                         "about (0.5,0.5), containment-rejection-sampled. 0=off")
    ap.add_argument("--aug-timerev", type=float, default=0.0, dest="aug_timerev",
                    help="FUTO-parity time reversal (Spec B3): probability of "
                         "reversing frames AND target. 0=off")
    ap.add_argument("--aug-maskhold", type=float, default=0.0, dest="aug_maskhold",
                    help="standalone frame-hold masking probability (isolates "
                         "Spec A's masking from CR). In CR mode masking is "
                         "applied per view at this probability too")
    ap.add_argument("--snapshot-every", type=int, default=0, dest="snapshot_every",
                    help="Phase J: keep a numbered checkpoint every N "
                         "validations (snap_<step>.pt) — the supply for the "
                         "beam-selected checkpoint soup. 0 = off")
    ap.add_argument("--ema-decay", type=float, default=0.0, dest="ema_decay",
                    help="C2: EMA decay for the evaluated/exported weights (0 = off)")
    ap.add_argument("--beam-val-rows", type=int, default=2000, dest="beam_val_rows",
                    help="Phase D: select best.pt on lexicon-beam top-1 over this "
                         "many val rows instead of greedy accuracy (0 = greedy, "
                         "the pre-Phase-D behaviour). Phase B showed greedy "
                         "anti-correlates with beam top-1.")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--beam-jobs", type=int, default=12, dest="beam_jobs")
    ap.add_argument("--select-layout-probes", default="",
                    dest="select_layout_probes",
                    help="Phase I: comma-separated selection-probe layouts "
                         "('synth:<seed>' or a futo layout name; dvorak "
                         "refused). best.pt is then chosen on a blend of "
                         "canonical and warped-probe beam top-1 — PHASE_H §6.2 "
                         "showed QWERTY-only selection does not order "
                         "checkpoints by transfer accuracy")
    ap.add_argument("--select-layout-rows", type=int, default=2000,
                    dest="select_layout_rows",
                    help="val-prefix rows warped per selection probe")
    ap.add_argument("--select-layout-weight", type=float, default=1.0,
                    dest="select_layout_weight",
                    help="w in score = (t1_qwerty + w*mean(probe_t1)) / (1+w)")
    ap.add_argument("--select-preset", default="", dest="select_preset",
                    help="scoring preset the SELECTION beam uses, as "
                         "'gamma,lambda,beta,gammaPrune,betaPrune' (default: the "
                         "published enc preset). A checkpoint should be selected "
                         "at the preset it will be reported at — Phase E's E1 arm "
                         "moved that preset a long way from the published one")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined",
                    help="AOSP lexicon for the selection beam")
    ap.add_argument("--val-jsonl", default="data/val_hwsfuto.jsonl", dest="val_jsonl",
                    help="canonical val split; supplies the beam targets")
    ap.add_argument("--kd-teacher", default="", dest="kd_teacher",
                    help="checkpoint whose log_emissions the student is distilled "
                         "towards (Phase F). Must be one of OUR OWN checkpoints — "
                         "distilling any FUTO output would re-import their licence.")
    ap.add_argument("--kd-weight", type=float, default=0.0, dest="kd_weight",
                    help="weight on the KD term; 0 disables distillation")
    ap.add_argument("--kd-temp", type=float, default=2.0, dest="kd_temp",
                    help="softmax temperature for the KD term (loss is scaled by "
                         "T^2 so its gradient magnitude is temperature-invariant)")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if "test" in Path(args.val_jsonl).name:
        raise SystemExit(f"refusing to select checkpoints on {args.val_jsonl}: "
                         "test-2400 is sealed")

    if not args.run_name:
        args.run_name = time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = resolve(args.workdir, Path("ckpt") / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    cache_dir = resolve(args.workdir, args.cache)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    centers = load_layout_centers(args.layout)
    # Built BEFORE the model touches the GPU: the beam pool is forked once here so
    # the 147 k-word trie is inherited copy-on-write and no child ever inherits a
    # live CUDA context.
    beam_val = None
    if args.beam_val_rows > 0:
        sel_preset = _BV_PRESET
        if args.select_preset:
            vals = [float(v) for v in args.select_preset.split(",")]
            if len(vals) != 5:
                raise SystemExit("--select-preset needs 5 floats: "
                                 "gamma,lambda,beta,gammaPrune,betaPrune")
            sel_preset = (vals[0], vals[1], vals[2], vals[3], vals[4])
        probes = [p.strip() for p in args.select_layout_probes.split(",")
                  if p.strip()]
        beam_val = BeamValidator(args.workdir, args.layout, cache_dir,
                                 resolve(args.workdir, args.val_jsonl),
                                 resolve(args.workdir, args.vocab),
                                 args.beam_val_rows, args.beam_width,
                                 args.beam_jobs, sel_preset, t_out=args.t_out,
                                 probes=probes or None,
                                 probe_rows=args.select_layout_rows,
                                 probe_weight=args.select_layout_weight)
    if beam_val is None:
        select_metric = "val_greedy"
    elif beam_val.segments[1:]:
        select_metric = "val_beam_mix"
    else:
        select_metric = "val_beam_t1"
    # A comma-separated --train-npz concatenates caches; repeating a name repeats
    # its rows, which is how the HWS-oversampling arm is expressed.
    train_npz = [cache_dir / p.strip() for p in args.train_npz.split(",") if p.strip()]
    for p in train_npz:
        if not p.exists():
            raise SystemExit(f"--train-npz: missing {p}")
    layout_aug = None
    if args.layout_alt_p > 0.0:
        real_names = [n.strip() for n in args.layout_alt_real.split(",") if n.strip()]
        if "dvorak" in real_names:
            # Not an error — but it must never happen silently: dvorak is the
            # held-out transfer probe and training on it voids that eval.
            print("⚠ layout-alt real pool INCLUDES dvorak — the dvorak eval is "
                  "no longer a held-out transfer test")
        layout_dir = Path(__file__).resolve().parent / "layouts"
        reals = [load_az_centers(layout_dir / f"futo_{n}.json") for n in real_names]
        layout_aug = LayoutAugmenter(args.layout_alt_p, args.layout_synth_frac,
                                     reals)
        print(f"layout-alt: p={args.layout_alt_p} synth_frac="
              f"{args.layout_synth_frac:.3f} real pool={real_names or 'none'}")
    source_centers: "Optional[List[Optional[np.ndarray]]]" = None
    if args.train_layouts:
        specs = [s.strip() for s in args.train_layouts.split(",")]
        if len(specs) != len(train_npz):
            raise SystemExit(f"--train-layouts has {len(specs)} entries for "
                             f"{len(train_npz)} --train-npz pools")
        source_centers = []
        for s in specs:
            if not s or s == "canonical":
                source_centers.append(None)
                continue
            p = resolve(args.workdir, s)
            try:
                c = load_layout_centers(p)
            except (KeyError, SystemExit):
                c = load_az_centers(p)
            source_centers.append(c)
            print(f"train-layout: {p.name} -> {c.shape[0]} keys")
    train_ds = SwipeDataset(train_npz, centers, augment=True,
                            path_offset_sigma=args.path_offset_sigma,
                            path_scale_sigma=args.path_scale_sigma,
                            affine_sampler=args.affine_sampler,
                            layout_aug=layout_aug,
                            source_centers=source_centers,
                            cr_views=args.cr_alpha > 0.0,
                            aug_shear=args.aug_shear, aug_rot_deg=args.aug_rot,
                            aug_timerev=args.aug_timerev,
                            aug_maskhold=args.aug_maskhold,
                            mask_frac=args.cr_mask_frac,
                            mask_spans=args.cr_mask_spans)
    val_ds = SwipeDataset(cache_dir / "val.npz", centers, augment=False)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, collate_fn=collate,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_workers = min(2, args.workers)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, collate_fn=collate,
                        num_workers=val_workers,
                        persistent_workers=val_workers > 0)

    dilations = tuple(int(v) for v in args.dilations.split(","))
    model = CtcSwipeEncoder(ch=args.ch, embed_hid=args.embed_hid,
                            dilations=dilations, feat_version=args.feat_version,
                            block=args.block, t_out=args.t_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run {args.run_name}  params: {n_params / 1e6:.3f}M ({n_params})  "
          f"device: {device}  feat_v{args.feat_version} block={args.block} "
          f"ch={args.ch} dil={dilations} t_out={args.t_out}  "
          f"train {len(train_ds)}  val {len(val_ds)}")

    # EMA shadow + a second module to run the val pass on the averaged weights.
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    eval_model = model
    if ema is not None:
        eval_model = CtcSwipeEncoder(ch=args.ch, embed_hid=args.embed_hid,
                                     dilations=dilations,
                                     feat_version=args.feat_version,
                                     block=args.block, t_out=args.t_out).to(device)
        print(f"EMA on (decay {args.ema_decay}); val and export use averaged weights")

    # ── knowledge distillation (Phase F) ────────────────────────────────────────
    # The teacher is one of OUR checkpoints (never FUTO weights or FUTO outputs —
    # the licence defines models trained on their outputs as derivatives). It runs
    # frozen in eval() on the same augmented batch as the student, so the two see
    # the identical slot permutation and the identical geometric jitter; the KD
    # term is therefore a per-frame comparison of two distributions over the SAME
    # column assignment, which is what makes it well defined under permutation.
    # Phase G: --kd-teacher accepts a comma-separated list; two or more
    # checkpoints form an ENSEMBLE teacher whose target is the log of the
    # arithmetic mean of the members' probabilities (logsumexp - log n, exact
    # and finite — pad columns sit at MASK_NEG for every member).
    teachers: List[torch.nn.Module] = []
    if args.kd_teacher and args.kd_weight > 0:
        shas = []
        for spec in args.kd_teacher.split(","):
            tpath = resolve(args.workdir, spec.strip())
            tck = torch.load(tpath, map_location=device, weights_only=True)
            teacher = encoder_from_checkpoint(tck).to(device).eval()
            teacher.load_state_dict(tck["model"])
            for p in teacher.parameters():
                p.requires_grad_(False)
            shas.append(sha256_file(tpath))
            tp = sum(p.numel() for p in teacher.parameters())
            print(f"KD: teacher {tpath} ch={teacher.ch} block={teacher.block} "
                  f"{tp / 1e6:.3f}M params, weight {args.kd_weight}, "
                  f"T {args.kd_temp}, sha {shas[-1][:12]}")
            teachers.append(teacher)
        args.kd_teacher_sha256 = ",".join(shas)
        if len(teachers) > 1:
            print(f"KD: ensemble of {len(teachers)} teachers "
                  f"(target = log mean prob)")
    elif args.kd_teacher or args.kd_weight > 0:
        raise SystemExit("--kd-teacher and --kd-weight must be given together")
    if args.cr_alpha > 0 and teachers:
        raise SystemExit("--cr-alpha and --kd-teacher are mutually exclusive "
                         "(the CR branch runs two student forwards)")

    def kd_loss(student_log_e: torch.Tensor,
                teacher_log_e: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled KL(teacher || student) over the emission head.

        Both tensors are already log-softmaxed over the 65 columns, so they are
        valid logits (they differ from the pre-softmax logits by a per-frame
        constant, which the softmax removes). Dividing by ``T`` and re-normalizing
        is therefore exactly the usual temperature-scaled distillation.

        Taken over all 65 columns rather than the 27-wide sliced view, which is
        the same quantity: the 38 pad columns sit at ``MASK_NEG`` for *both*
        models, so ``exp(teacher/T)`` is exactly 0 there and they contribute
        nothing to the sum — while the sum over the real key columns plus blank is
        precisely the sliced view. ``MASK_NEG`` is finite (-1e4), so no ``-inf``
        ever enters the expression.
        """
        t = args.kd_temp
        tgt = torch.log_softmax(teacher_log_e / t, dim=-1)
        src = torch.log_softmax(student_log_e / t, dim=-1)
        # Summed over classes, averaged over frames AND batch, so the term is a
        # per-frame KL in nats and sits on the same scale as the per-sequence CTC
        # loss. `batchmean` would leave a factor T=32 in it and make any published
        # --kd-weight meaningless outside this exact frame count.
        kl = F.kl_div(src, tgt, reduction="sum", log_target=True)
        return kl / (student_log_e.shape[0] * student_log_e.shape[1]) * (t * t)

    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    if args.short_loss_weight != 1.0 and args.cr_alpha > 0:
        raise SystemExit("--short-loss-weight is not plumbed through the CR "
                         "dual-view branch; run one lever at a time")
    ctc_none = (torch.nn.CTCLoss(blank=BLANK, zero_infinity=True,
                                 reduction="none")
                if args.short_loss_weight != 1.0 else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Step-equalized mode lets tiers of very different sizes share one budget.
    total_steps = args.total_steps if args.total_steps > 0 \
        else max(1, args.epochs * len(train_dl))
    val_every = args.val_every if args.val_every > 0 else len(train_dl)
    if args.total_steps > 0:
        args.epochs = 10 ** 6      # step budget is the real stopping rule

    def lr_at(step: int) -> float:
        """Linear warmup then cosine decay to 0 over the full step horizon."""
        if step < args.warmup:
            return (step + 1) / args.warmup          # audit fix #12
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best, best_epoch, step, start_epoch = -1.0, -1, 0, 0
    if args.resume:
        rpath = resolve(args.workdir, args.resume)
        ck = torch.load(rpath, map_location=device, weights_only=True)
        prev = ck.get("args", {})
        for k in ARCH_ARGS:
            if k in prev and prev[k] != getattr(args, k):
                raise SystemExit(f"--resume {rpath}: arch mismatch on '{k}' "
                                 f"(checkpoint {prev[k]} != requested {getattr(args, k)})")
        model.load_state_dict(ck.get("model_raw") or ck["model"])
        if ema is not None:
            ema.shadow = {k: v.detach().clone().float() if v.is_floating_point()
                          else v.detach().clone()
                          for k, v in ck["model"].items()}
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step = int(ck["step"])
        start_epoch = int(ck["epoch"]) + 1
        best, best_epoch = float(ck["best"]), int(ck["best_epoch"])
        prev_metric = ck.get("select_metric", "val_greedy")
        if prev_metric != select_metric:
            # `best` is on a different scale (greedy is a fraction, beam t1 a
            # percent), so carrying it across would silently freeze or thrash
            # best.pt. Restart the tracker instead of comparing incomparables.
            print(f"[resume] selection metric changed {prev_metric} -> "
                  f"{select_metric}; resetting best")
            best, best_epoch = -1.0, -1
        restore_rng(ck["rng"])
        print(f"[resume] {rpath}: continuing at epoch {start_epoch}, step {step}, "
              f"best {select_metric} {best:.4f} @ epoch {best_epoch}")

    if start_epoch >= args.epochs:
        print(f"nothing to do: checkpoint already at epoch {start_epoch - 1} "
              f"of --epochs {args.epochs}")
        return 0

    evals = best_eval = 0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = 0.0
        running_kd = 0.0
        nb = 0
        for feats, keys, mask, targets, tlens in train_dl:
            feats = feats.to(device, non_blocking=True)
            keys = keys.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            kd = None
            if feats.dim() == 4:
                # CR-CTC (Spec A): two grad-carrying forwards on the two views,
                # loss = ½(CTC_a + CTC_b) + α·½Σ[KL(sg(b)‖a) + KL(sg(a)‖b)]
                # per-frame over all 65 columns (pad columns contribute exactly
                # 0 — both views hold them at MASK_NEG, same argument as KD).
                log_a, _, _ = model(feats[:, 0], keys[:, 0], mask)
                log_b, _, _ = model(feats[:, 1], keys[:, 1], mask)
                in_lens = torch.full((log_a.shape[0],), args.t_out,
                                     dtype=torch.long)
                la = ctc(log_a.permute(1, 0, 2), targets, in_lens, tlens)
                lb = ctc(log_b.permute(1, 0, 2), targets, in_lens, tlens)
                loss = 0.5 * (la + lb)
                ctc_val = float(loss.detach())
                nf = log_a.shape[0] * log_a.shape[1]
                cr = 0.5 * (F.kl_div(log_a, log_b.detach(), reduction="sum",
                                     log_target=True)
                            + F.kl_div(log_b, log_a.detach(), reduction="sum",
                                       log_target=True)) / nf
                running_kd += float(cr.detach())
                loss = loss + args.cr_alpha * cr
            else:
                log_e, _, _ = model(feats, keys, mask)          # [B,32,65] fp32
                if teachers:
                    with torch.no_grad():
                        t_logs = [t(feats, keys, mask)[0] for t in teachers]
                        t_log_e = t_logs[0] if len(t_logs) == 1 else \
                            torch.logsumexp(torch.stack(t_logs), 0) - math.log(len(t_logs))
                    kd = kd_loss(log_e, t_log_e)
                log_e = log_e.permute(1, 0, 2)                  # [T',B,65] for CTCLoss
                in_lens = torch.full((log_e.shape[1],), args.t_out, dtype=torch.long)
                if ctc_none is not None:
                    # K4 short-word emphasis: weighted mean of the per-sample
                    # length-normalized losses ('mean' reduction == weights 1).
                    per = ctc_none(log_e, targets, in_lens, tlens) / \
                        tlens.to(log_e.device).clamp(min=1)
                    w = torch.where(tlens.to(log_e.device) <= 3,
                                    torch.as_tensor(args.short_loss_weight,
                                                    device=log_e.device), 1.0)
                    loss = (per * w).sum() / w.sum()
                    # logged ctc_loss stays the UNWEIGHTED mean -> comparable
                    # across arms.
                    ctc_val = float(per.detach().mean())
                else:
                    loss = ctc(log_e, targets, in_lens, tlens)
                    ctc_val = float(loss.detach())
                # `running` stays the CTC term alone, so the logged `ctc_loss`
                # remains comparable across every arm whether KD/CR is on or off.
                if kd is not None:
                    running_kd += float(kd.detach())
                    loss = loss + args.kd_weight * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += ctc_val
            nb += 1
            step += 1
            if ema is not None:
                ema.update(model, step)

            if step % val_every == 0 or step >= total_steps:
                if ema is not None:
                    ema.copy_to(eval_model)
                acc = greedy_accuracy(eval_model, val_dl, device)
                bt0 = time.time()
                bm = beam_val.run(eval_model, device) if beam_val is not None else None
                beam_secs = time.time() - bt0 if bm else 0.0
                # Phase D: the checkpoint is chosen by the metric that ships.
                # Phase I: with layout probes on, by the blended score instead.
                score = beam_val.score if bm else acc
                lr_now = sched.get_last_lr()[0]
                secs = time.time() - t0
                mean_loss = running / max(nb, 1)
                mean_kd = running_kd / max(nb, 1)
                running, running_kd, nb, t0 = 0.0, 0.0, 0, time.time()
                evals += 1
                bstr = (f"  beam t1 {bm[0]:5.2f} t3 {bm[1]:5.2f} t5 {bm[2]:5.2f} "
                        f"({beam_secs:.1f}s)" if bm else "")
                if bm and beam_val.probe_t1:
                    bstr += ("  probes "
                             + " ".join(f"{k} {v:5.2f}"
                                        for k, v in beam_val.probe_t1.items())
                             + f"  mix {beam_val.score:5.2f}")
                kdstr = (f"  kd {mean_kd:.4f}" if teachers else
                         f"  cr {mean_kd:.4f}" if args.cr_alpha > 0 else "")
                print(f"epoch {epoch:3d} step {step:6d}  ctc_loss {mean_loss:.4f}"
                      f"{kdstr}  val_greedy {acc * 100:.2f}%{bstr}  "
                      f"lr {lr_now:.2e}  {secs:.1f}s", flush=True)
                with open(metrics_path, "a") as mf:
                    rec = {"epoch": epoch, "step": step, "ctc_loss": mean_loss,
                           "val_greedy": acc, "lr": lr_now,
                           "seconds": round(secs, 3)}
                    if teachers:
                        rec["kd_loss"] = mean_kd
                    elif args.cr_alpha > 0:
                        rec["cr_loss"] = mean_kd
                    if bm:
                        rec.update({"val_beam_t1": bm[0], "val_beam_t3": bm[1],
                                    "val_beam_t5": bm[2],
                                    "beam_seconds": round(beam_secs, 3)})
                        if beam_val.probe_t1:
                            rec["probe_t1"] = beam_val.probe_t1
                            rec["val_beam_mix"] = beam_val.score
                    mf.write(json.dumps(rec) + "\n")
                if score > best:
                    best, best_epoch, best_eval = score, epoch, evals
                ckpt = build_checkpoint(model, opt, sched, step, epoch, best,
                                        best_epoch, args, acc, ema, bm,
                                        select_metric)
                atomic_save(ckpt, run_dir / "last.pt")
                if args.snapshot_every > 0 and evals % args.snapshot_every == 0:
                    # Soup supply (Phase J): numbered, never pruned; the soup
                    # script selects on the logged val_beam_t1 of each.
                    atomic_save(ckpt, run_dir / f"snap_{step}.pt")
                if best_eval == evals:
                    atomic_save(ckpt, run_dir / "best.pt")
                elif evals - best_eval >= args.patience:
                    print(f"early stop (best {select_metric} {best:.4f})", flush=True)
                    stop = True
                if step >= total_steps:
                    print(f"reached step budget {total_steps}", flush=True)
                    stop = True
                if stop:
                    break
        if stop:
            break

    if beam_val is not None:
        beam_val.close()
    print(f"done. best {select_metric} {best:.4f} @ epoch {best_epoch} "
          f"-> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
