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
from model import MAX_KEYS, T_OUT, CtcSwipeEncoder  # noqa: E402
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

    :param npz_path: output of ``prepare_data.py``.
    :param centers: ``[K,2]`` canonical key centers.
    :param augment: enable affine/noise augmentation.
    :param permute: also scatter the K keys into random slots of the 64 (only
        meaningful when ``augment``). The phase-2 refinement head trains on the
        sliced 27-class view, which is defined for the canonical identity slot
        assignment only, so ``train_refine.py`` passes ``permute=False``.
    """

    def __init__(self, npz_path: Path, centers: np.ndarray, augment: bool,
                 permute: bool = True, path_offset_sigma: float = 0.0,
                 path_scale_sigma: float = 0.0) -> None:
        with np.load(npz_path) as d:                    # audit fix #14: close handle
            self.features = np.array(d["features"])     # [N,2,64]
            self.tgt_flat = np.array(d["targets"])
            self.tgt_len = np.array(d["target_lengths"])
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
                     val_greedy: float, ema: "Optional[ModelEMA]" = None
                     ) -> Dict[str, object]:
    """Assemble the full resumable checkpoint payload.

    When EMA is on, ``model`` holds the AVERAGED weights — that is what every
    downstream consumer (export, eval, latency) should see, and it is what the
    reported val number was measured on. The live weights are kept under
    ``model_raw`` so a resume continues the actual trajectory rather than the
    average.
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
                    help="training cache inside --cache (tier arm selector)")
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
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

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
    train_ds = SwipeDataset(cache_dir / args.train_npz, centers, augment=True,
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
        restore_rng(ck["rng"])
        print(f"[resume] {rpath}: continuing at epoch {start_epoch}, step {step}, "
              f"best val_greedy {best * 100:.2f}% @ epoch {best_epoch}")

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
                lr_now = sched.get_last_lr()[0]
                secs = time.time() - t0
                mean_loss = running / max(nb, 1)
                running, nb, t0 = 0.0, 0, time.time()
                evals += 1
                print(f"epoch {epoch:3d} step {step:6d}  ctc_loss {mean_loss:.4f}  "
                      f"val_greedy {acc * 100:.2f}%  lr {lr_now:.2e}  {secs:.1f}s",
                      flush=True)
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({"epoch": epoch, "step": step,
                                         "ctc_loss": mean_loss, "val_greedy": acc,
                                         "lr": lr_now, "seconds": round(secs, 3)}) + "\n")
                if acc > best:
                    best, best_epoch, best_eval = acc, epoch, evals
                ckpt = build_checkpoint(model, opt, sched, step, epoch, best,
                                        best_epoch, args, acc, ema)
                atomic_save(ckpt, run_dir / "last.pt")
                if best_eval == evals:
                    atomic_save(ckpt, run_dir / "best.pt")
                elif evals - best_eval >= args.patience:
                    print(f"early stop (best val_greedy {best * 100:.2f}%)", flush=True)
                    stop = True
                if step >= total_steps:
                    print(f"reached step budget {total_steps}", flush=True)
                    stop = True
                if stop:
                    break
        if stop:
            break

    print(f"done. best val_greedy {best * 100:.2f}% @ epoch {best_epoch} "
          f"-> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
