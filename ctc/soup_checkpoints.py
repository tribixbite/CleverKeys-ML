#!/usr/bin/env python3
"""Greedy beam-selected checkpoint soup with BN re-estimation (Phase J, J6).

RESEARCH_SCAN.md §1.1 #4: average the weights of several within-run checkpoints,
selected greedily on the SHIPPING metric (lexicon-beam top-1 over the 5,000-row
val prefix — the same quantity ``train.py`` selects ``best.pt`` on), not on a
blind exponential window (the refuted EMA). Trap handled here: ``resbn`` running
BatchNorm statistics are NOT weights — after averaging they are re-estimated by
forward passes over augmented training rows (the distribution they were
accumulated under), before any export-time fold.

Cross-seed soups are refused by construction (one run dir).

Usage:
  python soup_checkpoints.py --run phaseJ-xyz [--max-members 8] [--bn-rows 20480]
Writes ``ckpt/<run>/soup.pt`` (same checkpoint schema as best.pt) and prints the
selection ledger. Export via export_onnx.py as usual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_aug import DEFAULT_REAL_POOL, LayoutAugmenter, load_az_centers  # noqa: E402
from model import encoder_from_checkpoint  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
import train as train_mod  # noqa: E402


def average_state(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Uniform average of float tensors; integer buffers copied from member 0."""
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


@torch.no_grad()
def reestimate_bn(model: torch.nn.Module, ds, device: str, rows: int,
                  batch: int = 256, seed: int = 999) -> None:
    """Reset BN running stats and re-accumulate them over augmented train rows."""
    any_bn = False
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None          # cumulative moving average
            any_bn = True
    if not any_bn:
        return
    model.train()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(rows, len(ds)), replace=False)
    np.random.seed(seed)               # augmentation stream, deterministic
    for a in range(0, len(idx), batch):
        items = [ds[int(i)] for i in idx[a:a + batch]]
        feats = torch.stack([it[0] for it in items]).to(device)
        keys = torch.stack([it[1] for it in items]).to(device)
        mask = torch.stack([it[2] for it in items]).to(device)
        model(feats, keys, mask)
    model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--run", required=True, help="run name under ckpt/")
    ap.add_argument("--max-members", type=int, default=8, dest="max_members")
    ap.add_argument("--bn-rows", type=int, default=20480, dest="bn_rows")
    ap.add_argument("--beam-jobs", type=int, default=8, dest="beam_jobs")
    ap.add_argument("--out", default="soup.pt")
    args = ap.parse_args()

    run_dir = resolve(args.workdir, Path("ckpt") / args.run)
    snaps = sorted(run_dir.glob("snap_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if len(snaps) < 2:
        raise SystemExit(f"{run_dir}: need >= 2 snap_*.pt (train with "
                         f"--snapshot-every); found {len(snaps)}")
    # Rank candidates by the val_beam_t1 logged at their step.
    logged: Dict[int, float] = {}
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "val_beam_t1" in r:
                logged[int(r["step"])] = float(r["val_beam_t1"])
    cands = sorted(((logged.get(int(p.stem.split("_")[1]), -1.0), p)
                    for p in snaps), reverse=True)
    print(f"{len(cands)} snapshots; top5 logged t1: "
          f"{[(round(t, 2), p.name) for t, p in cands[:5]]}")

    ck0 = torch.load(cands[0][1], map_location="cpu", weights_only=True)
    a = ck0["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    centers = train_mod.load_layout_centers(Path(a["layout"]))
    cache_dir = resolve(args.workdir, Path(a["cache"]))
    layout_aug = None
    if a.get("layout_alt_p", 0) > 0:
        reals = [load_az_centers(Path(__file__).resolve().parent / "layouts" /
                                 f"futo_{n}.json")
                 for n in a["layout_alt_real"].split(",") if n]
        layout_aug = LayoutAugmenter(a["layout_alt_p"], a["layout_synth_frac"], reals)
    train_npz = [cache_dir / p for p in a["train_npz"].split(",")]
    ds = train_mod.SwipeDataset(train_npz, centers, augment=True,
                                affine_sampler=a.get("affine_sampler", "coupled"),
                                layout_aug=layout_aug)
    bv = train_mod.BeamValidator(
        args.workdir, Path(a["layout"]), cache_dir,
        resolve(args.workdir, a["val_jsonl"]), resolve(args.workdir, a["vocab"]),
        int(a["beam_val_rows"]), int(a["beam_width"]), args.beam_jobs,
        t_out=int(a.get("t_out", 32)))

    model = encoder_from_checkpoint(ck0).to(device)

    def score(state: Dict[str, torch.Tensor], bn: bool) -> float:
        model.load_state_dict(state)
        if bn:
            reestimate_bn(model, ds, device, args.bn_rows)
        model.eval()
        t1, t3, t5 = bv.run(model, device)
        return t1

    members = [cands[0][1]]
    states = [torch.load(cands[0][1], map_location="cpu",
                         weights_only=True)["model"]]
    best_t1 = score(states[0], bn=False)
    print(f"seed member {members[0].name}: beam t1 {best_t1:.2f}")
    ledger = [{"member": members[0].name, "t1": best_t1, "kept": True}]
    for t_logged, p in cands[1:]:
        if len(members) >= args.max_members:
            break
        cand_states = states + [torch.load(p, map_location="cpu",
                                           weights_only=True)["model"]]
        t1 = score(average_state(cand_states), bn=True)
        keep = t1 > best_t1
        print(f"  + {p.name} (logged {t_logged:.2f}) -> soup t1 {t1:.2f} "
              f"{'KEPT' if keep else 'rejected'}")
        ledger.append({"member": p.name, "t1": t1, "kept": keep})
        if keep:
            members.append(p)
            states = cand_states
            best_t1 = t1
    final = average_state(states)
    model.load_state_dict(final)
    if len(states) > 1:
        reestimate_bn(model, ds, device, args.bn_rows)
    model.eval()
    t1, t3, t5 = bv.run(model, device)
    bv.close()
    print(f"\nFINAL soup: {len(members)} members, beam t1/t3/t5 "
          f"{t1:.2f}/{t3:.2f}/{t5:.2f} (single-best baseline "
          f"{ledger[0]['t1']:.2f})")
    out = dict(ck0)
    out["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
    out["soup"] = {"members": [m.name for m in members], "ledger": ledger,
                   "final": [t1, t3, t5]}
    train_mod.atomic_save(out, run_dir / args.out)
    print(f"wrote {run_dir / args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
