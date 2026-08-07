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
  * #13 the shared affine is rejection-sampled to keep every transformed key
        center inside [0,1]; centers are never clipped.
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
from model import MAX_KEYS, T_OUT, CtcSwipeEncoder, slice_head_torch  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

BLANK = MAX_KEYS  # 64 — full-head blank index (the Kotlin slice relocates it to K)

#: Args that define the architecture; a --resume must agree on all of them.
ARCH_ARGS = ("ch", "embed_hid", "feat_version", "block", "dilations")

#: Affine augmentation bounds (audit fix #13 rejection-samples within these).
SCALE_LO, SCALE_HI = 0.85, 1.15
TRANS_ABS = 0.05
MIRROR_P = 0.25
AFFINE_TRIES = 10
PATH_NOISE = 0.005
CENTER_NOISE = 0.01
PERMUTE_P = 0.5


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
                 path_scale_sigma: float = 0.0) -> None:
        paths = [npz_path] if isinstance(npz_path, (str, Path)) else list(npz_path)
        feats, tgts, tlens = [], [], []
        for p in paths:
            with np.load(p) as d:                       # audit fix #14: close handle
                feats.append(np.array(d["features"]))   # [N,2,64]
                tgts.append(np.array(d["targets"]))
                tlens.append(np.array(d["target_lengths"]))
        self.features = feats[0] if len(feats) == 1 else np.concatenate(feats)
        self.tgt_flat = tgts[0] if len(tgts) == 1 else np.concatenate(tgts)
        self.tgt_len = tlens[0] if len(tlens) == 1 else np.concatenate(tlens)
        self.sources = paths
        self.tgt_off = np.concatenate([[0], np.cumsum(self.tgt_len)])
        self.centers = centers                          # [K,2]
        self.augment = augment
        self.permute = permute
        self.path_offset_sigma = path_offset_sigma
        self.path_scale_sigma = path_scale_sigma
        self.k = centers.shape[0]

    def __len__(self) -> int:
        return len(self.tgt_len)

    def _sample_affine(self) -> Tuple[float, float, float, float, bool]:
        """Rejection-sample an affine that keeps every key center in [0,1].

        Falls back to the identity affine after ``AFFINE_TRIES`` failures, so a
        center is never clipped into a neighbour (audit fix #13). Mirroring maps
        [0,1] onto itself and therefore never affects acceptance.
        """
        cx, cy = self.centers[:, 0], self.centers[:, 1]
        for _ in range(AFFINE_TRIES):
            sx, sy = np.random.uniform(SCALE_LO, SCALE_HI, 2)
            tx, ty = np.random.uniform(-TRANS_ABS, TRANS_ABS, 2)
            nx = (cx - 0.5) * sx + 0.5 + tx
            ny = (cy - 0.5) * sy + 0.5 + ty
            if nx.min() >= 0.0 and nx.max() <= 1.0 and ny.min() >= 0.0 and ny.max() <= 1.0:
                return float(sx), float(sy), float(tx), float(ty), bool(np.random.rand() < MIRROR_P)
        return 1.0, 1.0, 0.0, 0.0, bool(np.random.rand() < MIRROR_P)

    def __getitem__(self, i: int):
        feats = self.features[i].astype(np.float32).copy()          # [2,64]
        target = self.tgt_flat[self.tgt_off[i]:self.tgt_off[i + 1]].copy()
        centers = self.centers.copy()                               # [K,2]

        if self.augment:
            sx, sy, tx, ty, mirror = self._sample_affine()
            for arr_x, arr_y in ((feats[0], feats[1]),
                                 (centers[:, 0], centers[:, 1])):
                arr_x[:] = (arr_x - 0.5) * sx + 0.5 + tx
                arr_y[:] = (arr_y - 0.5) * sy + 0.5 + ty
                if mirror:
                    arr_x[:] = 1.0 - arr_x
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

        # Slot assignment: identity (the inference-time layout) or a random
        # permutation into the 64 slots, which forces the model to read key
        # geometry rather than slot index.
        keys = np.zeros((MAX_KEYS, 2), np.float32)
        mask = np.zeros((MAX_KEYS,), bool)
        if self.augment and self.permute and np.random.rand() < PERMUTE_P:
            slots = np.random.permutation(MAX_KEYS)[: self.k]
        else:
            slots = np.arange(self.k)
        keys[slots] = centers
        mask[slots] = True
        target_slots = slots[target]                    # letters -> slot indices

        return (torch.from_numpy(feats), torch.from_numpy(keys),
                torch.from_numpy(mask), torch.from_numpy(target_slots.astype(np.int64)))


def collate(batch):
    """Stack fixed-size tensors and flatten the ragged CTC targets."""
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
               width: int, preset: Tuple[float, float, float, float, float]) -> None:
    global _BV_TRIE, _BV_LETTERS, _BV_TARGETS, _BV_EM, _BV_WIDTH, _BV_PRESET
    _BV_TRIE, _BV_LETTERS, _BV_TARGETS, _BV_WIDTH = trie, letters, targets, width
    _BV_PRESET = preset
    _BV_EM = np.frombuffer(shm, np.float32).reshape(n, T_OUT, len(letters) + 1)


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
    """

    def __init__(self, workdir: Path, layout: Path, cache_dir: Path,
                 val_jsonl: Path, vocab: Path, n_rows: int, beam_width: int,
                 jobs: int,
                 preset: Tuple[float, float, float, float, float] = _BV_PRESET
                 ) -> None:
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
        # Targets are the jsonl words lowercased — identical to eval_arms.py, so
        # the selection metric and the reported metric are the same quantity.
        self.targets = [w.lower() for w, _, _, _ in rows[:self.n]]
        mismatch = sum(1 for a, b in zip(self.targets, cached_words[:self.n])
                       if a != b)
        self.features = torch.from_numpy(feats[:self.n])          # [N,2,64]
        self.n_letters = len(letters)
        keys = np.zeros((MAX_KEYS, 2), np.float32)
        keys[:self.n_letters] = centers
        mask = np.zeros((MAX_KEYS,), bool)
        mask[:self.n_letters] = True
        self.keys = torch.from_numpy(keys)[None]                  # [1,64,2]
        self.mask = torch.from_numpy(mask)[None]                  # [1,64]

        trie = load_combined_vocab(vocab)
        self._shm = mp.RawArray("f", self.n * T_OUT * (self.n_letters + 1))
        self._view = np.frombuffer(self._shm, np.float32).reshape(
            self.n, T_OUT, self.n_letters + 1)
        step = max(1, (self.n + jobs - 1) // jobs)
        self.chunks = [(a, min(a + step, self.n)) for a in range(0, self.n, step)]
        ctx = mp.get_context("fork")
        self.pool = ctx.Pool(jobs, initializer=_beam_init,
                             initargs=(trie, letters, self.targets, self._shm,
                                       self.n, beam_width, preset))
        print(f"beam-val: {self.n} rows, trie {trie.num_words} words, "
              f"width {beam_width}, {jobs} procs, preset {preset}"
              + (f"  ({mismatch} targets differ from the a-z-normalised cache)"
                 if mismatch else ""))

    @torch.no_grad()
    def run(self, model: torch.nn.Module, device: str, batch: int = 512,
            forward=None) -> Tuple[float, float, float]:
        """-> ``(t1, t3, t5)`` percentages for *model* on the fixed prefix.

        :param forward: ``(feats, keys, mask) -> sliced log-probs [B,32,K+1]``.
            Defaults to the base encoder's full head, sliced. ``train_refine.py``
            passes the refinement head so the head is selected on the same
            lexicon-beam top-1 the encoder arms are, rather than on greedy.
        """
        was_training = model.training
        model.eval()
        off = 0
        for s in range(0, self.n, batch):
            f = self.features[s:s + batch].to(device)
            b = f.shape[0]
            keys = self.keys.to(device).expand(b, -1, -1)
            mask = self.mask.to(device).expand(b, -1)
            if forward is None:
                sliced = slice_head_torch(model(f, keys, mask)[0], self.n_letters)
            else:
                sliced = forward(f, keys, mask)                    # [b,32,27]
            self._view[off:off + b] = sliced.detach().cpu().numpy()
            off += b
        if was_training:
            model.train()
        h1 = h3 = h5 = 0
        for a, b_, c in self.pool.map(_beam_chunk, self.chunks):
            h1 += a
            h3 += b_
            h5 += c
        return (h1 / self.n * 100.0, h3 / self.n * 100.0, h5 / self.n * 100.0)

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
    ap.add_argument("--block", default="res", choices=("res", "convnext"),
                    help="trunk block: original residual, or depthwise/GLU/GRN/SE (B2)")
    ap.add_argument("--dilations", default="1,2,4,8",
                    help="comma-separated dilation per trunk block")
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
    ap.add_argument("--ema-decay", type=float, default=0.0, dest="ema_decay",
                    help="C2: EMA decay for the evaluated/exported weights (0 = off)")
    ap.add_argument("--beam-val-rows", type=int, default=2000, dest="beam_val_rows",
                    help="Phase D: select best.pt on lexicon-beam top-1 over this "
                         "many val rows instead of greedy accuracy (0 = greedy, "
                         "the pre-Phase-D behaviour). Phase B showed greedy "
                         "anti-correlates with beam top-1.")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--beam-jobs", type=int, default=12, dest="beam_jobs")
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
        beam_val = BeamValidator(args.workdir, args.layout, cache_dir,
                                 resolve(args.workdir, args.val_jsonl),
                                 resolve(args.workdir, args.vocab),
                                 args.beam_val_rows, args.beam_width,
                                 args.beam_jobs, sel_preset)
    select_metric = "val_beam_t1" if beam_val is not None else "val_greedy"
    # A comma-separated --train-npz concatenates caches; repeating a name repeats
    # its rows, which is how the HWS-oversampling arm is expressed.
    train_npz = [cache_dir / p.strip() for p in args.train_npz.split(",") if p.strip()]
    for p in train_npz:
        if not p.exists():
            raise SystemExit(f"--train-npz: missing {p}")
    train_ds = SwipeDataset(train_npz, centers, augment=True,
                            path_offset_sigma=args.path_offset_sigma,
                            path_scale_sigma=args.path_scale_sigma)
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
                            block=args.block).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run {args.run_name}  params: {n_params / 1e6:.3f}M ({n_params})  "
          f"device: {device}  feat_v{args.feat_version} block={args.block} "
          f"ch={args.ch} dil={dilations}  train {len(train_ds)}  val {len(val_ds)}")

    # EMA shadow + a second module to run the val pass on the averaged weights.
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    eval_model = model
    if ema is not None:
        eval_model = CtcSwipeEncoder(ch=args.ch, embed_hid=args.embed_hid,
                                     dilations=dilations,
                                     feat_version=args.feat_version,
                                     block=args.block).to(device)
        print(f"EMA on (decay {args.ema_decay}); val and export use averaged weights")

    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
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
        nb = 0
        for feats, keys, mask, targets, tlens in train_dl:
            feats = feats.to(device, non_blocking=True)
            keys = keys.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            log_e, _, _ = model(feats, keys, mask)              # [B,32,65] fp32
            log_e = log_e.permute(1, 0, 2)                      # [T=32,B,65] for CTCLoss
            in_lens = torch.full((log_e.shape[1],), T_OUT, dtype=torch.long)
            loss = ctc(log_e, targets, in_lens, tlens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
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
                score = bm[0] if bm else acc
                lr_now = sched.get_last_lr()[0]
                secs = time.time() - t0
                mean_loss = running / max(nb, 1)
                running, nb, t0 = 0.0, 0, time.time()
                evals += 1
                bstr = (f"  beam t1 {bm[0]:5.2f} t3 {bm[1]:5.2f} t5 {bm[2]:5.2f} "
                        f"({beam_secs:.1f}s)" if bm else "")
                print(f"epoch {epoch:3d} step {step:6d}  ctc_loss {mean_loss:.4f}  "
                      f"val_greedy {acc * 100:.2f}%{bstr}  lr {lr_now:.2e}  "
                      f"{secs:.1f}s", flush=True)
                with open(metrics_path, "a") as mf:
                    rec = {"epoch": epoch, "step": step, "ctc_loss": mean_loss,
                           "val_greedy": acc, "lr": lr_now,
                           "seconds": round(secs, 3)}
                    if bm:
                        rec.update({"val_beam_t1": bm[0], "val_beam_t3": bm[1],
                                    "val_beam_t5": bm[2],
                                    "beam_seconds": round(beam_secs, 3)})
                    mf.write(json.dumps(rec) + "\n")
                if score > best:
                    best, best_epoch, best_eval = score, epoch, evals
                ckpt = build_checkpoint(model, opt, sched, step, epoch, best,
                                        best_epoch, args, acc, ema, bm,
                                        select_metric)
                atomic_save(ckpt, run_dir / "last.pt")
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
