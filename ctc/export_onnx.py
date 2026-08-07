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
from model import MAX_KEYS, T_IN, CtcSwipeEncoder  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

PARITY_TRIALS = 100
PARITY_TOL = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--out", default="ctc_swipe_encoder.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = resolve(args.workdir, args.ckpt)
    out_path = resolve(args.workdir, args.out)
    letters, _ = load_layout(args.layout)
    num_letters = len(letters)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = CtcSwipeEncoder(ch=ck.get("ch", 96),
                            embed_hid=ck.get("embed_hid", 96)).eval()
    model.load_state_dict(ck["model"])

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
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    worst_sliced = worst_full = 0.0
    agree = 0
    for _ in range(PARITY_TRIALS):
        f = torch.rand(1, 2, T_IN)
        k = torch.rand(1, MAX_KEYS, 2)
        with torch.no_grad():
            ref, _, _ = model(f, k, mask)
        out = sess.run(["log_emissions"],
                       {"features": f.numpy(), "layout_keys": k.numpy(),
                        "layout_mask": mask.numpy()})[0]
        ref_lp = slice_emissions(ref.numpy()[0], num_letters, MAX_KEYS)   # [32,27]
        out_lp = slice_emissions(out[0], num_letters, MAX_KEYS)           # [32,27]
        worst_sliced = max(worst_sliced, float(np.abs(out_lp - ref_lp).max()))
        worst_full = max(worst_full, float(np.abs(out - ref.numpy()).max()))
        agree += int((out_lp.argmax(-1) == ref_lp.argmax(-1)).all())
    print(f"sliced [32,{num_letters + 1}] max |onnx - torch| = {worst_sliced:.2e}   "
          f"argmax agreement {agree}/{PARITY_TRIALS}")
    print(f"(raw 65-wide head max abs = {worst_full:.2e}; the pad columns sit at "
          f"~{-1e4:.0e} where the float32 ULP is ~9.8e-4, so they are excluded "
          f"by design — see audit fix #1)")
    assert worst_sliced < PARITY_TOL and agree == PARITY_TRIALS, "export parity FAILED"
    print(f"exported {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
