#!/usr/bin/env python3
"""Coupled-pair CTC trainer — pipeline-v2 element E1 (see PIPELINE_V2_PROPOSAL.md).

Trains TWO encoders in one loop on the SAME augmented batches, with a ramped
mutual per-frame KL between their emission heads. The coupling pins the CTC
alignment phase — the arbitrary gauge Phase K discovered the loss never fixes
(PHASE_K.md §4.1–4.3: same-recipe seeds do not share an alignment; only pairs
with per-frame argmax agreement ≥ ~95 % ensemble well, a gate confirmed
prospectively in §8.5). Here the agreement is trained-in instead of hoped-for,
so the exported pair is mixable (arithmetic probability averaging before the
beam, the mix2 contract) BY CONSTRUCTION, while the two members stay free to
differ on the axes that made mix2 good — data emphasis and loss weighting.

Member differentiation (the designed asymmetry):
  * distinct init seeds (--init-seed-a/--init-seed-b) — deliberately NOT the
    same-seed trick, so a working pair demonstrates the coupling does the
    pinning, not the init;
  * per-member short-word CTC loss weights (--slw-a 1.0 --slw-b 1.5): member B
    carries the ≤3 emphasis Phase K measured (+0.12 ≤3 seed-mean at w=2, at a
    designed 4+/spanish cost — §8.3); W between 1 and 2 is the registered,
    never-run interpolation arm. The averaged pair is the mirror-image merge.

Losses (per batch, log_e ∈ [B,T',65] log-softmaxed):
  L_ctc^m   = Σ_i w_i·CTC_i / Σ_i w_i,  CTC_i length-normalized per sample,
              w_i = slw_m if target_len_i ≤ 3 else 1   (== train.py --short-loss-weight)
  L_pair    = ½·(KL(sg(p_B)‖p_A) + KL(sg(p_A)‖p_B)) / (B·T′)   [nats/frame,
              summed over all 65 columns — pad columns contribute 0, same
              argument as train.py's CR/KD terms]
  L_geo^m   = mean_t Σ_ℓ p_t(ℓ)·max(0, d(path_t, key_ℓ) − r)   [optional,
              --geo-align-weight; blank exempt; an explicit alignment prior
              that pins letter mass near the finger]
  L_total   = L_ctc^A + L_ctc^B + λ_pair(step)·L_pair + λ_geo·(L_geo^A + L_geo^B)
  λ_pair(step) ramps 0 → --pair-weight over [--pair-ramp-start,
  --pair-ramp-start + --pair-ramp-len]: early CTC is blank-dominant and an
  immediate mutual pull would lock a degenerate all-blank agreement.

Selection: each member runs the standard 5 k-row lexicon-beam validator
(beam-t1, optionally blended with layout probes exactly as train.py); the PAIR
is selected jointly — best mean member score among evals whose measured
per-frame argmax agreement on the val prefix passes --agree-gate (default
0.95, the Phase-K validated threshold). Individual member bests are also kept
(the single-model finalist question). Member checkpoints are written in the
train.py/export_onnx.py-compatible format, so the existing export → quantize →
eval_beam --ens-avg prob toolchain runs unchanged.

Never touches test-2400 (same guard as train.py).

Usage (the v2 reference arm):
  python train_v2.py --run-name v2pair-s1234 --ch 192 --layout-alt-p 0.65 \
      --train-npz train_t3.npz,train_t3hws.npz,train_t3hws.npz,tier_sw234.npz,tier_sw5q.npz \
      --total-steps 188000 --beam-val-rows 5000 --workers 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_aug import DEFAULT_REAL_POOL, LayoutAugmenter, load_az_centers  # noqa: E402
from model import MAX_KEYS, T_IN, T_OUT, CtcSwipeEncoder  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
from train import (BLANK, SwipeDataset, BeamValidator, _BV_PRESET,  # noqa: E402
                   atomic_save, collate, greedy_accuracy, load_layout_centers,
                   restore_rng, rng_state)

#: Args that define each member's architecture; --resume must agree on all.
ARCH_ARGS = ("ch", "embed_hid", "feat_version", "block", "dilations", "t_out")


def build_member(args: argparse.Namespace, dilations: Tuple[int, ...],
                 device: str, init_seed: int) -> CtcSwipeEncoder:
    """Construct one member with its own init stream (global RNG is re-seeded
    by the caller afterwards, so member init never perturbs the data order)."""
    torch.manual_seed(init_seed)
    return CtcSwipeEncoder(ch=args.ch, embed_hid=args.embed_hid,
                           dilations=dilations, feat_version=args.feat_version,
                           block=args.block, t_out=args.t_out).to(device)


def weighted_ctc(ctc_none: torch.nn.CTCLoss, log_e: torch.Tensor,
                 targets: torch.Tensor, tlens: torch.Tensor,
                 slw: float,
                 src_w: "Optional[torch.Tensor]" = None
                 ) -> Tuple[torch.Tensor, float]:
    """Per-member short-weighted CTC. -> (loss, unweighted-mean float for logs).

    Identical math to train.py's --short-loss-weight branch: per-sample loss is
    length-normalized (the stock 'mean' reduction semantics), then a weighted
    mean with weight `slw` on len ≤ 3 targets. slw == 1.0 reproduces the stock
    reduction exactly. Phase N2a: *src_w* ([B], per-row source weight) simply
    multiplies into the same weighted mean; None reproduces the pre-N path.
    """
    t = log_e.permute(1, 0, 2)                                  # [T',B,65]
    in_lens = torch.full((t.shape[1],), t.shape[0], dtype=torch.long)
    per = ctc_none(t, targets, in_lens, tlens) / \
        tlens.to(log_e.device).clamp(min=1)
    w = torch.where(tlens.to(log_e.device) <= 3,
                    torch.as_tensor(float(slw), device=log_e.device),
                    torch.as_tensor(1.0, device=log_e.device))
    if src_w is not None:
        w = w * src_w.to(log_e.device)
    return (per * w).sum() / w.sum(), float(per.detach().mean())


def pair_kl(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Symmetric stop-grad per-frame KL in nats/frame over all 65 columns.

    Same normalization as train.py's CR/KD terms (sum over classes, mean over
    batch × frames), so --pair-weight lives on the same scale as --cr-alpha.
    """
    nf = log_a.shape[0] * log_a.shape[1]
    return 0.5 * (F.kl_div(log_a, log_b.detach(), reduction="sum",
                           log_target=True)
                  + F.kl_div(log_b, log_a.detach(), reduction="sum",
                             log_target=True)) / nf


def geo_align_penalty(log_e: torch.Tensor, feats: torch.Tensor,
                      keys: torch.Tensor, mask: torch.Tensor,
                      radius: float) -> torch.Tensor:
    """Expected excess distance of emitted letter mass from the finger.

    For emission frame t (input frames strided by T_IN // T′), penalize
    probability on letters whose key center sits further than `radius` from the
    path point: Σ_ℓ p_t(ℓ)·relu(d(path_t, key_ℓ) − radius). Blank is exempt —
    blanks are the alignment's slack and their placement is exactly what the
    sharp blank-penalty optimum (PHASE_J §2) says not to distort; this term
    only constrains WHERE letter mass may sit, pinning the alignment gauge.
    Slot-permutation-safe: emission columns and `keys` rows share the same slot
    indexing, and pad slots are masked.
    """
    t_out = log_e.shape[1]
    stride = T_IN // t_out
    p = log_e[..., :MAX_KEYS].exp()                             # [B,T',64]
    px = feats[:, 0, ::stride]                                  # [B,T']
    py = feats[:, 1, ::stride]
    dx = px.unsqueeze(-1) - keys[..., 0].unsqueeze(1)           # [B,T',64]
    dy = py.unsqueeze(-1) - keys[..., 1].unsqueeze(1)
    d = torch.sqrt(dx * dx + dy * dy + 1e-8)
    excess = torch.clamp(d - radius, min=0.0) * mask.unsqueeze(1).to(p.dtype)
    return (p * excess).sum(-1).mean()


@torch.no_grad()
def frame_agreement(model_a: torch.nn.Module, model_b: torch.nn.Module,
                    loader: DataLoader, device: str,
                    max_rows: int = 2000) -> float:
    """Per-frame argmax agreement of the two members on unaugmented val rows.

    The Phase-K label-free pair-compatibility gate (PHASE_K.md §4.3, validated
    prospectively in §8.5): ≥ ~0.95 predicts the ensemble works.
    """
    was_a, was_b = model_a.training, model_b.training
    model_a.eval()
    model_b.eval()
    agree = total = rows = 0
    for feats, keys, mask, _targets, _tlens in loader:
        feats = feats.to(device)
        keys = keys.to(device)
        mask = mask.to(device)
        la, _, _ = model_a(feats, keys, mask)
        lb, _, _ = model_b(feats, keys, mask)
        aa = la.argmax(-1)
        ab = lb.argmax(-1)
        agree += int((aa == ab).sum())
        total += aa.numel()
        rows += feats.shape[0]
        if rows >= max_rows:
            break
    model_a.train(was_a)
    model_b.train(was_b)
    return agree / max(total, 1)


def member_checkpoint(model: CtcSwipeEncoder, args: argparse.Namespace,
                      dilations: Tuple[int, ...], step: int, epoch: int,
                      score: float, member: str) -> Dict[str, object]:
    """train.py/export_onnx.py-compatible single-member checkpoint payload."""
    return {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "ch": args.ch, "embed_hid": args.embed_hid, "dilations": list(dilations),
        "feat_version": args.feat_version, "block": args.block,
        "t_out": args.t_out, "step": step, "epoch": epoch,
        "best": score, "best_epoch": epoch,
        "select_metric": "val_beam_t1", "member": member,
        "pair_args": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
    }


def pair_checkpoint(model_a, model_b, opt_a, opt_b, sched_a, sched_b,
                    step: int, epoch: int, best: Dict[str, float],
                    args: argparse.Namespace) -> Dict[str, object]:
    """Fully resumable pair state (model/opt/sched ×2 + RNG + trackers)."""
    return {
        "model_a": model_a.state_dict(), "model_b": model_b.state_dict(),
        "opt_a": opt_a.state_dict(), "opt_b": opt_b.state_dict(),
        "sched_a": sched_a.state_dict(), "sched_b": sched_b.state_dict(),
        "step": step, "epoch": epoch, "best": best,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "rng": rng_state(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--train-npz", dest="train_npz",
                    default="train_t3.npz,train_t3hws.npz,train_t3hws.npz,"
                            "tier_sw234.npz,tier_sw5q.npz",
                    help="comma-separated cache npzs (repeat a name to "
                         "oversample); default = the sw2345 finalist mix")
    ap.add_argument("--run-name", default="", dest="run_name")
    ap.add_argument("--resume", default="", help="pair checkpoint (last.pt)")
    # ── recipe (defaults = the Phase-J finalist recipe) ──────────────────────
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--total-steps", type=int, default=188000, dest="total_steps")
    ap.add_argument("--val-every", type=int, default=0, dest="val_every")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--embed-hid", type=int, default=96, dest="embed_hid")
    ap.add_argument("--feat-version", type=int, default=1, choices=(1, 2),
                    dest="feat_version")
    ap.add_argument("--block", default="resbn",
                    choices=("res", "convnext", "dwsep", "resbn"))
    ap.add_argument("--dilations", default="1,2,4,8")
    ap.add_argument("--t-out", type=int, default=T_OUT, choices=(32, 64),
                    dest="t_out")
    ap.add_argument("--affine-sampler", default="coupled",
                    choices=("coupled", "legacy"), dest="affine_sampler")
    ap.add_argument("--layout-alt-p", type=float, default=0.65, dest="layout_alt_p")
    ap.add_argument("--layout-synth-frac", type=float, default=2.0 / 3.0,
                    dest="layout_synth_frac")
    ap.add_argument("--layout-alt-real", default=",".join(DEFAULT_REAL_POOL),
                    dest="layout_alt_real")
    ap.add_argument("--train-layouts", default="", dest="train_layouts",
                    help="per-npz geometry (train.py semantics): empty/"
                         "'canonical' or a layout json path per pool")
    # ── the pair (E1/E3) ─────────────────────────────────────────────────────
    ap.add_argument("--seed", type=int, default=1234,
                    help="data/augmentation stream seed (shared by the pair)")
    ap.add_argument("--init-seed-a", type=int, default=1111, dest="init_seed_a")
    ap.add_argument("--init-seed-b", type=int, default=2222, dest="init_seed_b")
    ap.add_argument("--source-loss-weight", default="", dest="source_loss_weight",
                    help="comma floats, one per --train-npz entry: per-SOURCE "
                         "CTC loss weight (Phase N2a — the slw move aimed at a "
                         "corpus instead of a length stratum). Empty = all 1.0, "
                         "which reproduces the stock path bit-for-bit.")
    ap.add_argument("--slw-a", type=float, default=1.0, dest="slw_a",
                    help="member-A short-word (len ≤ 3) CTC loss weight")
    ap.add_argument("--slw-b", type=float, default=1.5, dest="slw_b",
                    help="member-B short-word CTC loss weight (the registered "
                         "W ∈ (1,2) interpolation, carried by one member)")
    ap.add_argument("--pair-weight", type=float, default=0.3, dest="pair_weight",
                    help="target weight of the mutual per-frame KL")
    ap.add_argument("--pair-ramp-start", type=int, default=5000,
                    dest="pair_ramp_start")
    ap.add_argument("--pair-ramp-len", type=int, default=15000,
                    dest="pair_ramp_len")
    ap.add_argument("--agree-gate", type=float, default=0.95, dest="agree_gate",
                    help="per-frame argmax agreement the pair selection requires")
    ap.add_argument("--geo-align-weight", type=float, default=0.0,
                    dest="geo_align_weight",
                    help="optional geometric alignment prior (E6; ablation-gated)")
    ap.add_argument("--geo-align-radius", type=float, default=0.18,
                    dest="geo_align_radius")
    # ── selection / infra (train.py semantics) ───────────────────────────────
    ap.add_argument("--beam-val-rows", type=int, default=5000, dest="beam_val_rows")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--beam-jobs", type=int, default=12, dest="beam_jobs")
    ap.add_argument("--select-layout-probes", default="",
                    dest="select_layout_probes")
    ap.add_argument("--select-layout-rows", type=int, default=2000,
                    dest="select_layout_rows")
    ap.add_argument("--select-layout-weight", type=float, default=1.0,
                    dest="select_layout_weight")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    ap.add_argument("--val-jsonl", default="data/val_hwsfuto.jsonl",
                    dest="val_jsonl")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 by default: the Phase-K deadlock rule for "
                         "unattended runs on this box")
    args = ap.parse_args()

    if "test" in Path(args.val_jsonl).name:
        raise SystemExit(f"refusing to select checkpoints on {args.val_jsonl}: "
                         "test-2400 is sealed")
    if not args.run_name:
        args.run_name = time.strftime("v2pair-%Y%m%d-%H%M%S")
    run_dir = resolve(args.workdir, Path("ckpt") / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    cache_dir = resolve(args.workdir, args.cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Beam validators are built BEFORE any CUDA touch (train.py's fork rule);
    # ONE validator serves both members — it holds no model state.
    beam_val: Optional[BeamValidator] = None
    if args.beam_val_rows > 0:
        probes = [p.strip() for p in args.select_layout_probes.split(",")
                  if p.strip()]
        beam_val = BeamValidator(args.workdir, args.layout, cache_dir,
                                 resolve(args.workdir, args.val_jsonl),
                                 resolve(args.workdir, args.vocab),
                                 args.beam_val_rows, args.beam_width,
                                 args.beam_jobs, _BV_PRESET, t_out=args.t_out,
                                 probes=probes or None,
                                 probe_rows=args.select_layout_rows,
                                 probe_weight=args.select_layout_weight)

    centers = load_layout_centers(args.layout)
    train_npz = [cache_dir / p.strip() for p in args.train_npz.split(",")
                 if p.strip()]
    for p in train_npz:
        if not p.exists():
            raise SystemExit(f"--train-npz: missing {p}")
    layout_aug = None
    if args.layout_alt_p > 0.0:
        real_names = [n.strip() for n in args.layout_alt_real.split(",")
                      if n.strip()]
        if "dvorak" in real_names:
            print("⚠ layout-alt real pool INCLUDES dvorak — the dvorak eval "
                  "is no longer a held-out transfer test")
        layout_dir = Path(__file__).resolve().parent / "layouts"
        reals = [load_az_centers(layout_dir / f"futo_{n}.json")
                 for n in real_names]
        layout_aug = LayoutAugmenter(args.layout_alt_p, args.layout_synth_frac,
                                     reals)
        print(f"layout-alt: p={args.layout_alt_p} "
              f"synth_frac={args.layout_synth_frac:.3f} real={real_names}")
    source_centers: Optional[List[Optional[np.ndarray]]] = None
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

    # Global streams seeded ONCE for the data/aug path; member inits use their
    # own seeds and the global stream is re-seeded afterwards so the shuffle
    # and augmentation draws are identical whatever the init seeds are.
    dilations = tuple(int(v) for v in args.dilations.split(","))
    model_a = build_member(args, dilations, device, args.init_seed_a)
    model_b = build_member(args, dilations, device, args.init_seed_b)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_ds = SwipeDataset(train_npz, centers, augment=True,
                            affine_sampler=args.affine_sampler,
                            layout_aug=layout_aug,
                            source_centers=source_centers)
    val_ds = SwipeDataset(cache_dir / "val.npz", centers, augment=False)
    # Phase N2a: per-source CTC loss weights. Row -> source via the dataset's
    # own cumulative counts; the wrapper draws no randomness and, when the flag
    # is empty, is not installed at all — the stock path is untouched.
    src_w: Optional[np.ndarray] = None
    if args.source_loss_weight:
        ws = [float(v) for v in args.source_loss_weight.split(",")]
        if len(ws) != len(train_ds.sources):
            raise SystemExit(f"--source-loss-weight has {len(ws)} entries for "
                             f"{len(train_ds.sources)} --train-npz files")
        src_w = np.ones(len(train_ds), np.float32)
        start = 0
        for end, w in zip(train_ds._src_end.tolist(), ws):
            src_w[start:end] = w
            start = end
        print("source-loss-weight: " + ", ".join(
            f"{Path(str(p)).name}={w} (n={n})" for p, w, n in
            zip(train_ds.sources, ws,
                np.diff([0, *train_ds._src_end.tolist()]).tolist())))

    class _SrcWeighted(torch.utils.data.Dataset):
        """Append the row's source weight to each sample tuple."""

        def __init__(self, ds, w): self.ds, self.w = ds, w
        def __len__(self): return len(self.ds)
        def __getitem__(self, i): return (*self.ds[i], self.w[i])

    def collate_w(batch):
        base = collate([b[:-1] for b in batch])
        return (*base, torch.tensor([b[-1] for b in batch]))

    train_dl = DataLoader(
        train_ds if src_w is None else _SrcWeighted(train_ds, src_w),
        args.batch, shuffle=True,
        collate_fn=collate if src_w is None else collate_w,
        num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)
    val_workers = min(2, args.workers)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, collate_fn=collate,
                        num_workers=val_workers,
                        persistent_workers=val_workers > 0)

    n_params = sum(p.numel() for p in model_a.parameters())
    print(f"run {args.run_name}  pair of {n_params / 1e6:.3f}M "
          f"({n_params} × 2)  device {device}  block={args.block} ch={args.ch} "
          f"dil={dilations} t_out={args.t_out}  slw A/B "
          f"{args.slw_a}/{args.slw_b}  pair_w {args.pair_weight} "
          f"(ramp {args.pair_ramp_start}+{args.pair_ramp_len})  "
          f"geo {args.geo_align_weight}  train {len(train_ds)}  "
          f"val {len(val_ds)}")

    ctc_none = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True,
                                reduction="none")
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    total_steps = args.total_steps if args.total_steps > 0 \
        else max(1, args.epochs * len(train_dl))
    val_every = args.val_every if args.val_every > 0 else len(train_dl)
    if args.total_steps > 0:
        args.epochs = 10 ** 6

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    def pair_w_at(step: int) -> float:
        if args.pair_weight <= 0.0:
            return 0.0
        f = (step - args.pair_ramp_start) / max(1, args.pair_ramp_len)
        return args.pair_weight * min(1.0, max(0.0, f))

    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lr_at)
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_at)

    best: Dict[str, float] = {"a": -1.0, "b": -1.0, "pair": -1.0}
    best_eval = {"a": -1, "b": -1, "pair": -1}
    step, start_epoch = 0, 0
    if args.resume:
        rpath = resolve(args.workdir, args.resume)
        ck = torch.load(rpath, map_location=device, weights_only=True)
        prev = ck.get("args", {})
        for k in ARCH_ARGS:
            if k in prev and str(prev[k]) != str(getattr(args, k)):
                raise SystemExit(f"--resume {rpath}: arch mismatch on '{k}' "
                                 f"({prev[k]} != {getattr(args, k)})")
        model_a.load_state_dict(ck["model_a"])
        model_b.load_state_dict(ck["model_b"])
        opt_a.load_state_dict(ck["opt_a"])
        opt_b.load_state_dict(ck["opt_b"])
        sched_a.load_state_dict(ck["sched_a"])
        sched_b.load_state_dict(ck["sched_b"])
        step = int(ck["step"])
        start_epoch = int(ck["epoch"]) + 1
        best = dict(ck["best"])
        restore_rng(ck["rng"])
        print(f"[resume] {rpath}: epoch {start_epoch}, step {step}, best {best}")

    evals = 0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        run_a = run_b = run_kl = run_geo = 0.0
        nb = 0
        for batch in train_dl:
            feats, keys, mask, targets, tlens = batch[:5]
            bw = batch[5] if len(batch) > 5 else None           # N2a source weights
            feats = feats.to(device, non_blocking=True)
            keys = keys.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            log_a, _, _ = model_a(feats, keys, mask)            # [B,T',65]
            log_b, _, _ = model_b(feats, keys, mask)
            la, la_log = weighted_ctc(ctc_none, log_a, targets, tlens,
                                      args.slw_a, bw)
            lb, lb_log = weighted_ctc(ctc_none, log_b, targets, tlens,
                                      args.slw_b, bw)
            loss = la + lb
            pw = pair_w_at(step)
            if pw > 0.0:
                kl = pair_kl(log_a, log_b)
                run_kl += float(kl.detach())
                loss = loss + pw * kl
            if args.geo_align_weight > 0.0:
                geo = geo_align_penalty(log_a, feats, keys, mask,
                                        args.geo_align_radius) \
                    + geo_align_penalty(log_b, feats, keys, mask,
                                        args.geo_align_radius)
                run_geo += float(geo.detach())
                loss = loss + args.geo_align_weight * geo

            opt_a.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            opt_a.step()
            opt_b.step()
            sched_a.step()
            sched_b.step()
            run_a += la_log
            run_b += lb_log
            nb += 1
            step += 1

            if step % val_every == 0 or step >= total_steps:
                greedy_a = greedy_accuracy(model_a, val_dl, device)
                greedy_b = greedy_accuracy(model_b, val_dl, device)
                score_a = score_b = None
                bm_a = bm_b = None
                if beam_val is not None:
                    bm_a = beam_val.run(model_a, device)
                    score_a = beam_val.score
                    bm_b = beam_val.run(model_b, device)
                    score_b = beam_val.score
                agree = frame_agreement(model_a, model_b, val_dl, device)
                sa = score_a if score_a is not None else greedy_a
                sb = score_b if score_b is not None else greedy_b
                pair_score = 0.5 * (sa + sb) if agree >= args.agree_gate else -1.0
                evals += 1
                secs = time.time() - t0
                print(f"epoch {epoch:3d} step {step:6d}  "
                      f"ctc A {run_a / max(nb, 1):.4f} B {run_b / max(nb, 1):.4f}  "
                      f"kl {run_kl / max(nb, 1):.4f} (w {pw:.2f})  "
                      f"geo {run_geo / max(nb, 1):.4f}  "
                      f"greedy A {greedy_a * 100:.2f} B {greedy_b * 100:.2f}  "
                      + (f"beam A {bm_a[0]:5.2f} B {bm_b[0]:5.2f}  "
                         if bm_a and bm_b else "")
                      + f"agree {agree * 100:.1f}%  "
                      f"pair {pair_score:.2f}  {secs:.1f}s", flush=True)
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({
                        "epoch": epoch, "step": step,
                        "ctc_a": run_a / max(nb, 1), "ctc_b": run_b / max(nb, 1),
                        "pair_kl": run_kl / max(nb, 1), "pair_w": pw,
                        "geo": run_geo / max(nb, 1),
                        "greedy_a": greedy_a, "greedy_b": greedy_b,
                        "beam_a": list(bm_a) if bm_a else None,
                        "beam_b": list(bm_b) if bm_b else None,
                        "score_a": sa, "score_b": sb,
                        "agreement": agree, "pair_score": pair_score,
                        "seconds": round(secs, 3)}) + "\n")
                run_a = run_b = run_kl = run_geo = 0.0
                nb = 0
                t0 = time.time()

                if sa > best["a"]:
                    best["a"], best_eval["a"] = sa, evals
                    atomic_save(member_checkpoint(model_a, args, dilations,
                                                  step, epoch, sa, "a"),
                                run_dir / "a_best.pt")
                if sb > best["b"]:
                    best["b"], best_eval["b"] = sb, evals
                    atomic_save(member_checkpoint(model_b, args, dilations,
                                                  step, epoch, sb, "b"),
                                run_dir / "b_best.pt")
                if pair_score > best["pair"]:
                    best["pair"], best_eval["pair"] = pair_score, evals
                    atomic_save(member_checkpoint(model_a, args, dilations,
                                                  step, epoch, pair_score,
                                                  "pair_a"),
                                run_dir / "pair_a_best.pt")
                    atomic_save(member_checkpoint(model_b, args, dilations,
                                                  step, epoch, pair_score,
                                                  "pair_b"),
                                run_dir / "pair_b_best.pt")
                atomic_save(pair_checkpoint(model_a, model_b, opt_a, opt_b,
                                            sched_a, sched_b, step, epoch,
                                            best, args),
                            run_dir / "last.pt")
                if evals - max(best_eval.values()) >= args.patience:
                    print(f"early stop (best {best})", flush=True)
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
    print(f"done. best {best} -> {run_dir}/{{a,b,pair_a,pair_b}}_best.pt\n"
          f"export each member with export_onnx.py, then decode the pair with\n"
          f"eval_beam.py --onnx pair_a.onnx,pair_b.onnx --ens-avg prob; the\n"
          f"pair selection already enforced agreement ≥ {args.agree_gate}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
