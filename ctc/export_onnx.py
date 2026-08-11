#!/usr/bin/env python3
"""Export a trained checkpoint -> ctc_swipe_encoder.onnx (fixed shapes, opset 17)
plus a torch/ONNX parity check.

Fixed shapes, batch 1, opset 17: the CTC engine makes exactly one NN call per
swipe with constant shapes, so dynamic axes buy nothing and cost graph
optimization.

Audit fixes applied here:
  * #1  parity is asserted on the CONTRACT view, not the raw 65-wide head. The
        38 pad columns carry ``MASK_NEG``-derived log-probs of about -1.0e4,
        where the float32 ULP is 9.77e-4, so a 1e-4 absolute tolerance over all
        65 columns could never pass — it failed deterministically on a real
        export. Comparing ``slice_emissions(...)`` -> [32,27] measures exactly
        what ``CtcEmissions.sliceFromHead`` feeds the Kotlin beam.
  * #16 --workdir / argparse pathing (was hardcoded).
  * torch 2.9 flips torch.onnx.export to the dynamo exporter; ``dynamo=False``
    is passed explicitly so this script keeps producing the same graph.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_ceiling import slice_emissions  # noqa: E402
from futo_decoder_eval import load_layout  # noqa: E402
from model import MAX_KEYS, T_IN, encoder_from_checkpoint  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

PARITY_TRIALS = 100
#: Max |onnx - torch| accepted on the sliced contract view, over REAL traces on
#: the REAL layout (the operating distribution). Measured across Phase J
#: checkpoints: 0.8e-4 (ch 80), 1.5-7.6e-4 (ch 192), 1.5-2.6e-4 (ch 256) —
#: argmax 100/100 in every case. The residue is a property of the individual
#: weight matrix (accumulation order), NOT of the width, and it varies by ~5x
#: between checkpoints of the same architecture, so this bound is deliberately
#: loose; **argmax agreement is the gate that matters** and it is asserted
#: separately. NOTE the pre-Phase-J residues quoted in PHASE_I.md were taken on
#: a WHITE-NOISE probe and are not comparable to what this script prints today.
PARITY_TOL = 1e-3
#: Conv+BN folding is exact in exact arithmetic; this bounds the float32 residue.
FOLD_TOL = 2e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--out", default="ctc_swipe_encoder.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-fold-bn", dest="fold_bn", action="store_false",
                    help="export BatchNorm as its own node instead of folding "
                         "it into the preceding conv (dwsep trunks only)")
    ap.add_argument("--parity-tol", type=float, default=PARITY_TOL,
                    dest="parity_tol",
                    help="max sliced |onnx - torch| accepted on the real-trace "
                         "probe (default 1e-3; measured 0.8e-4..7.6e-4 across Phase-J "
                         "checkpoints, varying per weight matrix rather than "
                         "with width). Argmax agreement must be 100/100 on "
                         "BOTH probes regardless of this bound")
    ap.add_argument("--parity-features", default="cache/val.npz",
                    dest="parity_features",
                    help="npz whose 'features' array supplies REAL traces for "
                         "the asserted parity probe (default cache/val.npz). "
                         "The white-noise probe is diagnostic only — it neither "
                         "upper- nor lower-bounds the real-trace residue")
    args = ap.parse_args()

    ckpt_path = resolve(args.workdir, args.ckpt)
    out_path = resolve(args.workdir, args.out)
    letters, centers = load_layout(args.layout)
    num_letters = len(letters)

    # Real traces on the real layout = the operating distribution every numeric
    # bound below is asserted on (see the parity block). White noise is kept as
    # a diagnostic only: measured Phase J, it is NOT a conservative stand-in in
    # either direction (ch 256: noise 5e-3..1e-2 vs real 1.5e-4; ch 192: noise
    # 3e-5 vs real 1.6e-4), so it can both cry wolf and under-report.
    real_keys = torch.zeros(1, MAX_KEYS, 2)
    real_keys[0, :num_letters] = torch.tensor(np.asarray(centers, dtype=np.float32))
    real_feats = None
    fpath = resolve(args.workdir, args.parity_features)
    if fpath.exists():
        arr = np.load(fpath, allow_pickle=True)["features"][:PARITY_TRIALS]
        real_feats = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    else:
        print(f"⚠ parity features {fpath} missing — falling back to the "
              f"white-noise probe (its residue bound is NOT calibrated)")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = encoder_from_checkpoint(ck).eval()
    model.load_state_dict(ck["model"])
    print(f"arch: ch={model.ch} block={model.block} feat_v{model.feat_version} "
          f"dil={model.dilations} t_out={model.t_out}  params "
          f"{sum(p.numel() for p in model.parameters())}")
    if model.t_out != 32:
        print(f"⚠ t_out={model.t_out} is CONTRACT-BREAKING (shipped contract is "
              f"[1,32,·] outputs) — measurement artifact only")

    # BatchNorm in eval mode is a per-channel affine, so it folds exactly into the
    # preceding conv (Phase F). Verified numerically here rather than assumed: a
    # silent fold error would show up as a small accuracy drift nothing else checks.
    if args.fold_bn:
        pmask = torch.zeros(1, MAX_KEYS, dtype=torch.bool)
        pmask[:, :num_letters] = True
        if real_feats is not None:
            probe = (real_feats, real_keys.expand(len(real_feats), -1, -1),
                     pmask.expand(len(real_feats), -1))
        else:
            probe = (torch.rand(1, 2, T_IN), torch.rand(1, MAX_KEYS, 2), pmask)
        with torch.no_grad():
            before = model(*probe)[0]
        n_folded = model.fold_bn()
        if n_folded:
            with torch.no_grad():
                after = model(*probe)[0]
            # On the CONTRACT view only (audit fix #1): the 38 pad columns sit at
            # ~-1e4, where one float32 ULP is 9.77e-4, so a raw 65-wide max
            # reports that ULP as "drift" and hides the real number.
            sl = np.stack([slice_emissions(x, num_letters, MAX_KEYS)
                           for x in (after - before).numpy()])
            drift = float(np.abs(sl).max())
            print(f"folded {n_folded} Conv+BatchNorm pairs; max |Δlog_emissions| "
                  f"= {drift:.2e} (sliced contract view)")
            assert drift < FOLD_TOL, f"BN fold changed the output by {drift:.2e}"

    feats = torch.rand(1, 2, T_IN)
    keys = torch.rand(1, MAX_KEYS, 2)
    mask = torch.zeros(1, MAX_KEYS, dtype=torch.bool)
    mask[:, :num_letters] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (feats, keys, mask), str(out_path),
        input_names=["features", "layout_keys", "layout_mask"],
        output_names=["log_emissions", "coefficients", "lambda"],
        opset_version=args.opset,
        dynamic_axes=None,          # fully static: [1,2,64] / [1,64,2] / [1,64]
        do_constant_folding=True,
        dynamo=False,               # keep the TorchScript exporter across 2.9
    )

    # ── parity on the sliced contract view (audit fix #1) ───────────────────────
    # Two probes. The ASSERTED one feeds REAL swipe traces on the REAL layout
    # centers — the operating distribution the contract actually sees. The
    # legacy white-noise probe (uniform features AND uniform key positions) is
    # kept as a printed diagnostic only. Phase J measured that it tracks the
    # real-trace residue in neither direction: ch 256 white-noise 5e-3..1e-2 vs
    # 1.5e-4 real (gating on it fails a graph that is exact to 2e-4 where it is
    # used), while ch 192 white-noise 3.4e-5 vs 1.6e-4 real (it under-reports by
    # 5x). Only the operating distribution is asserted on magnitude; argmax
    # agreement is asserted on BOTH probes.
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    def _probe(feat_fn, key_fn):
        """(worst sliced |Δ|, worst raw |Δ|, argmax agreements) over PARITY_TRIALS."""
        worst_s = worst_f = 0.0
        ok = 0
        for i in range(PARITY_TRIALS):
            f, k = feat_fn(i), key_fn()
            with torch.no_grad():
                ref, _, _ = model(f, k, mask)
            out = sess.run(["log_emissions"],
                           {"features": f.numpy(), "layout_keys": k.numpy(),
                            "layout_mask": mask.numpy()})[0]
            ref_lp = slice_emissions(ref.numpy()[0], num_letters, MAX_KEYS)  # [32,27]
            out_lp = slice_emissions(out[0], num_letters, MAX_KEYS)          # [32,27]
            worst_s = max(worst_s, float(np.abs(out_lp - ref_lp).max()))
            worst_f = max(worst_f, float(np.abs(out - ref.numpy()).max()))
            ok += int((out_lp.argmax(-1) == ref_lp.argmax(-1)).all())
        return worst_s, worst_f, ok

    noise = (lambda i: torch.rand(1, 2, T_IN), lambda: torch.rand(1, MAX_KEYS, 2))
    if real_feats is not None:
        probe_desc = f"real traces ({fpath.name}) on {args.layout.name}"
        worst_sliced, worst_full, agree = _probe(
            lambda i: real_feats[i % len(real_feats)][None], lambda: real_keys)
        rnd_sliced, _, rnd_agree = _probe(*noise)
    else:
        probe_desc = "white noise (no real features available)"
        worst_sliced, worst_full, agree = _probe(*noise)
        rnd_sliced, rnd_agree = worst_sliced, agree
    print(f"sliced [{model.t_out},{num_letters + 1}] max |onnx - torch| = "
          f"{worst_sliced:.2e}   argmax agreement {agree}/{PARITY_TRIALS}   "
          f"({probe_desc})")
    print(f"(white-noise diagnostic probe: sliced {rnd_sliced:.2e}, argmax "
          f"{rnd_agree}/{PARITY_TRIALS} — NOT asserted on magnitude, "
          f"out-of-distribution)")
    print(f"(raw 65-wide head max abs = {worst_full:.2e}; the pad columns sit at "
          f"~{-1e4:.0e} where the float32 ULP is ~9.8e-4, so they are excluded "
          f"by design — see audit fix #1)")
    assert worst_sliced < args.parity_tol and agree == PARITY_TRIALS, \
        "export parity FAILED"
    assert rnd_agree == PARITY_TRIALS, \
        "export parity FAILED: random-layout probe flipped an argmax"
    print(f"exported {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
