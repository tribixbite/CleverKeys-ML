#!/usr/bin/env python3
"""Phase-K3 rescorer: train the tiny listwise candidate ranker on mined slates.

Data = ``mine_candidates.py`` shards (features already computed by
``ranker_features.slate_features``). Supervision: listwise softmax
cross-entropy over the slate with the gold's slot as the target; slates whose
gold is absent from the beam's top-k are excluded (a reranker cannot conjure a
candidate — that is K2's job).

Model: standardization affine (buffers, baked from train stats) + MLP
14 -> 64 -> 64 -> 1 (~5.2 k params, far under the 100 k budget). Exported to
ONNX (input ``features [N,14]`` -> output ``score [N,1]``) for the eval-time
blend ``final' = beam_final + w * ranker`` in ``eval_beam.py --ranker-onnx``.

The blend weight ``w`` is NOT chosen here: it is swept on val[0:half] and
confirmed on the holdout half, per the campaign protocol.

Usage:
  python train_ranker.py --shards 'phaseK/mined_sw2345_*.npz' \
      --out phaseK/ranker_sw2345
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DEFAULT_WORKDIR  # noqa: E402
from ranker_features import NUM_FEATURES  # noqa: E402


class Ranker(nn.Module):
    """Standardize -> MLP -> scalar score per candidate."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, hidden: int = 64) -> None:
        super().__init__()
        self.register_buffer("mu", torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer("sd", torch.from_numpy(sd.astype(np.float32)))
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, F] -> [N, 1]
        return self.net((x - self.mu) / self.sd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--shards", required=True,
                    help="glob under workdir, e.g. 'phaseK/mined_sw2345_*.npz'")
    ap.add_argument("--out", required=True,
                    help="output prefix under workdir (writes .pt and .onnx)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--err-weight", type=float, default=4.0, dest="err_weight",
                    help="loss weight multiplier for slates whose gold is NOT "
                         "already rank 1")
    ap.add_argument("--short-weight", type=float, default=2.0,
                    dest="short_weight",
                    help="loss weight multiplier for len(gold) <= 3 slates "
                         "(the K3 target stratum)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024, help="slates per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dev-frac", type=float, default=0.05, dest="dev_frac")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    files = sorted(glob.glob(str(args.workdir / args.shards)))
    if not files:
        raise SystemExit(f"no shards match {args.shards}")
    FX, GR, SL, WLEN = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        keep = d["gold_rank"] >= 0
        FX.append(d["features"][keep])
        GR.append(d["gold_rank"][keep])
        SL.append(d["slate_len"][keep])
        WLEN.append(np.char.str_len(d["words"][keep]))
        print(f"{f}: {keep.sum()}/{len(keep)} supervisable slates", flush=True)
    fx = np.concatenate(FX)                       # [S, k, F]
    gr = np.concatenate(GR).astype(np.int64)      # [S]
    sl = np.concatenate(SL).astype(np.int64)
    wl = np.concatenate(WLEN)
    S, K, F = fx.shape
    assert F == NUM_FEATURES
    print(f"{S} slates  (gold@1 {(gr == 0).mean() * 100:.1f}%, "
          f"short {(wl <= 3).mean() * 100:.1f}%)", flush=True)

    rng = np.random.default_rng(args.seed)
    dev_mask = rng.random(S) < args.dev_frac
    tr, dv = np.where(~dev_mask)[0], np.where(dev_mask)[0]

    valid = (np.arange(K)[None, :] < sl[:, None])          # [S, K]
    flat = fx[valid]                                       # only real candidates
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-6] = 1.0

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = Ranker(mu, sd, args.hidden).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ranker params: {n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    x_all = torch.from_numpy(fx).to(dev)                    # [S, K, F]
    y_all = torch.from_numpy(gr).to(dev)
    m_all = torch.from_numpy(valid).to(dev)
    w_all = torch.ones(S)
    w_all[gr != 0] *= args.err_weight
    w_all[wl <= 3] *= args.short_weight
    w_all = w_all.to(dev)

    def dev_stats() -> str:
        model.eval()
        with torch.no_grad():
            s = model(x_all[dv].reshape(-1, F)).reshape(len(dv), K)
            s = s.masked_fill(~m_all[dv], -1e30)
            pred = s.argmax(1)
        y = y_all[dv]
        acc = (pred == y).float().mean().item()
        err = y != 0
        accerr = (pred[err] == y[err]).float().mean().item()
        sh = torch.from_numpy(wl[dv] <= 3).to(dev)
        accsh = (pred[sh] == y[sh]).float().mean().item()
        model.train()
        return (f"dev pure-ranker top1 {acc * 100:.2f}%  "
                f"on-error slates {accerr * 100:.2f}%  short {accsh * 100:.2f}%")

    print(f"pre-train {dev_stats()}", flush=True)
    steps = 0
    for ep in range(args.epochs):
        perm = torch.from_numpy(rng.permutation(tr)).to(dev)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            logits = model(x_all[b].reshape(-1, F)).reshape(len(b), K)
            logits = logits.masked_fill(~m_all[b], -1e30)
            loss = (nn.functional.cross_entropy(logits, y_all[b],
                                                reduction="none")
                    * w_all[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
        print(f"epoch {ep} loss {loss.item():.4f}  {dev_stats()}", flush=True)

    out_pt = args.workdir / f"{args.out}.pt"
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "hidden": args.hidden, "num_features": NUM_FEATURES}, out_pt)
    model.eval().cpu()
    out_onnx = args.workdir / f"{args.out}.onnx"
    torch.onnx.export(model, torch.zeros(1, NUM_FEATURES), str(out_onnx),
                      input_names=["features"], output_names=["score"],
                      dynamic_axes={"features": {0: "n"}, "score": {0: "n"}},
                      opset_version=17)
    # parity probe
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx),
                                providers=["CPUExecutionProvider"])
    probe = fx[dv[:64], 0] if len(dv) >= 64 else flat[:64]
    with torch.no_grad():
        ref = model(torch.from_numpy(probe)).numpy()
    got = sess.run(["score"], {"features": probe})[0]
    err = float(np.abs(ref - got).max())
    print(f"wrote {out_pt} and {out_onnx}  (onnx parity max abs {err:.2e})",
          flush=True)
    assert err < 1e-4
    return 0


if __name__ == "__main__":
    sys.exit(main())
