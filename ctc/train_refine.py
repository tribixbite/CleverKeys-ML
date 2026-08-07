#!/usr/bin/env python3
"""Phase 2 — train the CTC refinement head on top of a frozen base encoder (guide §11).

The head is our ``magic_macaw`` analogue: per frame it consumes
``concat(sliced_emissions[27] | coefficients[64] | lambda[1]) = [T'=32, 92]`` from
the frozen encoder and emits refined ``log_probs[32, 27]`` that REPLACE the
emissions before the trie beam. FUTO measured +5.88 pt top-1 from this lever.

Usage:
  python train_refine.py --base-ckpt ckpt/r2/best.pt --run-name r2-refine
  python train_refine.py --resume ckpt/r2-refine/last.pt --epochs 90 --run-name r2-refine

Canonical-QWERTY gating (important)
-----------------------------------
The base encoder is trained with slot-permutation augmentation, so its 65-wide
head is layout-agnostic. The *sliced* 27-class view, however, is only the
alphabet when keys occupy slots ``[0..26)`` in emission order — i.e. the
canonical identity assignment. Refinement training therefore runs with
``permute=False`` (geometric jitter is still on). Exactly like FUTO's
``magic_macaw``, which is layout-fingerprint-gated to en_qwerty, this head is
valid only for the canonical layout; other layouts must fall back to the
encoder-only path (guide §11 step 5).

Reuses ``train.py`` wholesale for the dataset, collate, greedy metric,
checkpoint/RNG plumbing and pathing; only the model, the loss view (blank = 26)
and the optional unfreeze schedule are new.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (CtcRefineHead, CtcSwipeEncoder, encoder_from_checkpoint, T_OUT,  # noqa: E402
                   build_refine_input, refine_input_dim)
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve, sha256_file  # noqa: E402
from train import (BeamValidator, SwipeDataset, atomic_save, collate,  # noqa: E402
                   greedy_accuracy, load_layout_centers, restore_rng, rng_state)

#: Args that define the head's architecture; a --resume must agree on all of them.
ARCH_ARGS = ("hidden",)

#: Base-encoder lr multiplier once --unfreeze-after fires.
UNFREEZE_LR_SCALE = 0.1


def load_base(base_ckpt: Path, device: str) -> CtcSwipeEncoder:
    """Load the trained encoder that produces the head's inputs."""
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    base = encoder_from_checkpoint(ck)
    base.load_state_dict(ck["model"])
    return base.to(device)


def make_forward(base: CtcSwipeEncoder, head: CtcRefineHead, num_letters: int,
                 grad_through_base: bool):
    """Build the ``(feats, keys, mask) -> refined log_probs [B,32,27]`` callable."""
    def forward(feats, keys, mask):
        with torch.set_grad_enabled(grad_through_base and torch.is_grad_enabled()):
            log_e, coeff, lam = base(feats, keys, mask)
        if not grad_through_base:
            log_e, coeff, lam = log_e.detach(), coeff.detach(), lam.detach()
        return head(build_refine_input(log_e, coeff, lam, num_letters))
    return forward


def build_checkpoint(head, base, opt, sched, step: int, epoch: int, best: float,
                     best_epoch: int, args: argparse.Namespace, val_greedy: float,
                     unfrozen: bool, val_beam=None,
                     select_metric: str = "val_greedy") -> Dict[str, object]:
    """Assemble the full resumable checkpoint payload.

    The base encoder's weights are stored too, because ``--unfreeze-after`` may
    have changed them; ``eval_beam``/``export_refine_onnx`` read the head alone.

    ``best``/``best_epoch`` track whichever metric ``select_metric`` names — from
    Phase E that is ``val_beam_t1`` (percent), not ``val_greedy`` (fraction).
    """
    return {
        "head": head.state_dict(),
        "base": base.state_dict(),
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
        "unfrozen": unfrozen,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rng": rng_state(),
        "hidden": args.hidden,
        "num_letters": args.num_letters,
        "base_ckpt": str(args.base_ckpt),
        "base_sha256": args.base_sha256,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--base-ckpt", default="ckpt/r2/best.pt", dest="base_ckpt",
                    help="frozen encoder to refine")
    ap.add_argument("--train-npz", default="train.npz", dest="train_npz",
                    help="training cache(s) inside --cache; comma-separated names "
                         "are concatenated, exactly as in train.py. Must be the "
                         "tier the base encoder was trained on")
    ap.add_argument("--run-name", default="", dest="run_name")
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--total-steps", type=int, default=0, dest="total_steps",
                    help="step budget (overrides --epochs); the cosine horizon "
                         "follows it, so tiers of different sizes are comparable")
    ap.add_argument("--val-every", type=int, default=0, dest="val_every",
                    help="validate every N steps (default: once per epoch)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--unfreeze-after", type=int, default=-1, dest="unfreeze_after",
                    help="epoch at which to unfreeze the base at 10x lower lr "
                         "(default -1 = never)")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--beam-val-rows", type=int, default=5000, dest="beam_val_rows",
                    help="select best.pt on lexicon-beam top-1 over this val prefix "
                         "(0 = fall back to greedy). Phase D showed greedy "
                         "anti-correlates with beam top-1, and the head is judged "
                         "by the beam, so it must be selected by it too")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--beam-jobs", type=int, default=12, dest="beam_jobs")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    ap.add_argument("--val-jsonl", default="data/val_hwsfuto.jsonl", dest="val_jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if "test" in Path(args.val_jsonl).name:
        raise SystemExit(f"refusing to select checkpoints on {args.val_jsonl}: "
                         "test-2400 is sealed")

    if not args.run_name:
        args.run_name = time.strftime("refine-%Y%m%d-%H%M%S")
    run_dir = resolve(args.workdir, Path("ckpt") / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    cache_dir = resolve(args.workdir, args.cache)
    base_path = resolve(args.workdir, args.base_ckpt)
    args.base_sha256 = sha256_file(base_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    centers = load_layout_centers(args.layout)
    args.num_letters = int(centers.shape[0])
    blank = args.num_letters                          # 26 in the sliced view

    # Built BEFORE the model touches the GPU: the beam pool forks once here, so
    # the trie is inherited copy-on-write and no child holds a CUDA context.
    beam_val = None
    if args.beam_val_rows > 0:
        beam_val = BeamValidator(args.workdir, args.layout, cache_dir,
                                 resolve(args.workdir, args.val_jsonl),
                                 resolve(args.workdir, args.vocab),
                                 args.beam_val_rows, args.beam_width, args.beam_jobs)
    select_metric = "val_beam_t1" if beam_val is not None else "val_greedy"

    # permute=False: the sliced 27-view is only the alphabet under identity slots.
    train_npz = [cache_dir / p.strip() for p in args.train_npz.split(",") if p.strip()]
    for p in train_npz:
        if not p.exists():
            raise SystemExit(f"--train-npz: missing {p}")
    train_ds = SwipeDataset(train_npz, centers, augment=True, permute=False)
    val_ds = SwipeDataset(cache_dir / "val.npz", centers, augment=False, permute=False)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, collate_fn=collate,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_workers = min(2, args.workers)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, collate_fn=collate,
                        num_workers=val_workers, persistent_workers=val_workers > 0)

    base = load_base(base_path, device)
    head = CtcRefineHead(num_letters=args.num_letters, hidden=args.hidden).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    n_base = sum(p.numel() for p in base.parameters())
    print(f"run {args.run_name}  head {n_head / 1e3:.1f}K params "
          f"(in_dim {refine_input_dim(args.num_letters)}, hidden {args.hidden})  "
          f"frozen base {n_base / 1e6:.2f}M from {base_path.name}  device: {device}  "
          f"train {len(train_ds)}  val {len(val_ds)}")

    # Both param groups exist from the start so LambdaLR's base_lrs stay valid;
    # the base group simply carries no gradients until --unfreeze-after fires.
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    unfrozen = False
    opt = torch.optim.AdamW(
        [{"params": list(head.parameters()), "lr": args.lr},
         {"params": list(base.parameters()), "lr": args.lr * UNFREEZE_LR_SCALE}],
        lr=args.lr, weight_decay=args.weight_decay)

    ctc = torch.nn.CTCLoss(blank=blank, zero_infinity=True)
    # Step-equalized mode (as in train.py) so a head trained on a 1 M-row tier is
    # comparable with one trained on 110 k rows.
    total_steps = args.total_steps if args.total_steps > 0 \
        else max(1, args.epochs * len(train_dl))
    val_every = args.val_every if args.val_every > 0 else len(train_dl)
    if args.total_steps > 0:
        args.epochs = 10 ** 6            # the step budget is the real stopping rule

    def lr_at(step: int) -> float:
        """Linear warmup then cosine decay to 0 over the full step horizon."""
        if step < args.warmup:
            return (step + 1) / args.warmup
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
        if ck.get("base_sha256") and ck["base_sha256"] != args.base_sha256:
            raise SystemExit(f"--resume {rpath}: base encoder mismatch "
                             f"(checkpoint {ck['base_sha256'][:12]} != "
                             f"{args.base_sha256[:12]} from {base_path})")
        head.load_state_dict(ck["head"])
        base.load_state_dict(ck["base"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step = int(ck["step"])
        start_epoch = int(ck["epoch"]) + 1
        best, best_epoch = float(ck["best"]), int(ck["best_epoch"])
        unfrozen = bool(ck.get("unfrozen", False))
        prev_metric = ck.get("select_metric", "val_greedy")
        if prev_metric != select_metric:
            # greedy is a fraction, beam t1 a percent — carrying `best` across
            # would freeze or thrash best.pt. Restart the tracker instead.
            print(f"[resume] selection metric changed {prev_metric} -> "
                  f"{select_metric}; resetting best")
            best, best_epoch = -1.0, -1
        restore_rng(ck["rng"])
        print(f"[resume] {rpath}: continuing at epoch {start_epoch}, step {step}, "
              f"best {select_metric} {best:.4f} @ epoch {best_epoch}, "
              f"base {'unfrozen' if unfrozen else 'frozen'}")

    if start_epoch >= args.epochs:
        print(f"nothing to do: checkpoint already at epoch {start_epoch - 1} "
              f"of --epochs {args.epochs}")
        return 0

    evals = best_eval = 0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if not unfrozen and 0 <= args.unfreeze_after <= epoch:
            unfrozen = True
            for p in base.parameters():
                p.requires_grad_(True)
            print(f"[unfreeze] epoch {epoch}: base encoder now trains at "
                  f"{UNFREEZE_LR_SCALE}x the head lr", flush=True)
        base.train() if unfrozen else base.eval()
        head.train()
        forward = make_forward(base, head, args.num_letters, grad_through_base=unfrozen)

        t0 = time.time()
        running = 0.0
        nb = 0
        for feats, keys, mask, targets, tlens in train_dl:
            feats = feats.to(device, non_blocking=True)
            keys = keys.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            log_p = forward(feats, keys, mask)                  # [B,32,27] fp32
            log_p = log_p.permute(1, 0, 2)                      # [T=32,B,27]
            in_lens = torch.full((log_p.shape[1],), T_OUT, dtype=torch.long)
            loss = ctc(log_p, targets, in_lens, tlens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = list(head.parameters()) + (list(base.parameters()) if unfrozen else [])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            nb += 1
            step += 1

            if step % val_every == 0 or step >= total_steps:
                was_base_training = base.training
                base.eval()
                eval_forward = make_forward(base, head, args.num_letters,
                                            grad_through_base=False)
                acc = greedy_accuracy(head, val_dl, device, forward=eval_forward,
                                      blank=blank)
                bt0 = time.time()
                bm = (beam_val.run(head, device, forward=eval_forward)
                      if beam_val is not None else None)
                beam_secs = time.time() - bt0 if bm else 0.0
                if was_base_training:
                    base.train()
                # The head is judged by the beam, so it is selected by the beam.
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
                           "seconds": round(secs, 3), "unfrozen": unfrozen}
                    if bm:
                        rec.update({"val_beam_t1": bm[0], "val_beam_t3": bm[1],
                                    "val_beam_t5": bm[2],
                                    "beam_seconds": round(beam_secs, 3)})
                    mf.write(json.dumps(rec) + "\n")

                if score > best:
                    best, best_epoch, best_eval = score, epoch, evals
                ckpt = build_checkpoint(head, base, opt, sched, step, epoch, best,
                                        best_epoch, args, acc, unfrozen, bm,
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
