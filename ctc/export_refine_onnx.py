#!/usr/bin/env python3
"""Export the phase-2 refinement head -> ctc_refine_head.onnx (guide §11 step 5).

Second ONNX alongside the encoder, with the fixed signature the ceiling harness
and the Kotlin side already expect:

    decoder_input      [1, 32, 92]  float32   concat(sliced[27] | coeff[64] | lambda[1])
    refined_log_probs  [1, 32, 27]  float32   log-softmaxed; blank at column 26

Fixed shapes, batch 1, opset 17, ``dynamo=False`` — same conventions as
``export_onnx.py``. Parity is asserted directly on the ``[32,27]`` output, which
here IS the contract view (no pad columns exist to distort the tolerance, unlike
the encoder's 65-wide head — see audit fix #1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import load_layout  # noqa: E402
from model import CtcRefineHead, T_OUT, refine_input_dim  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

PARITY_TRIALS = 100
PARITY_TOL = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--ckpt", default="ckpt/r2-refine/best.pt")
    ap.add_argument("--out", default="ctc_refine_head.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = resolve(args.workdir, args.ckpt)
    out_path = resolve(args.workdir, args.out)
    letters, _ = load_layout(args.layout)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    num_letters = int(ck.get("num_letters", len(letters)))
    if num_letters != len(letters):
        raise SystemExit(f"{ckpt_path}: head trained for {num_letters} letters but "
                         f"{args.layout} has {len(letters)}")
    head = CtcRefineHead(num_letters=num_letters,
                         hidden=ck.get("hidden", 128)).eval()
    head.load_state_dict(ck["head"])
    in_dim = refine_input_dim(num_letters)

    # The head's real input is a log-prob block concatenated with coefficients and
    # a softplus gate, so exercise parity on a similarly scaled sample rather than
    # uniform noise: sliced log-probs are negative, coefficients O(1), lambda > 0.
    def sample() -> torch.Tensor:
        sl = torch.log_softmax(torch.randn(1, T_OUT, num_letters + 1) * 4.0, dim=-1)
        co = torch.randn(1, T_OUT, in_dim - (num_letters + 1) - 1)
        la = torch.nn.functional.softplus(torch.randn(1, T_OUT, 1))
        return torch.cat([sl, co, la], dim=-1)

    dummy = sample()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        head, (dummy,), str(out_path),
        input_names=["decoder_input"],
        output_names=["refined_log_probs"],
        opset_version=args.opset,
        dynamic_axes=None,          # fully static: [1,32,92] -> [1,32,27]
        do_constant_folding=True,
        dynamo=False,
    )

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    agree = 0
    for _ in range(PARITY_TRIALS):
        x = sample()
        with torch.no_grad():
            ref = head(x).numpy()[0]
        got = sess.run(["refined_log_probs"], {"decoder_input": x.numpy()})[0][0]
        worst = max(worst, float(np.abs(got - ref).max()))
        agree += int((got.argmax(-1) == ref.argmax(-1)).all())
    print(f"[{T_OUT},{num_letters + 1}] max |onnx - torch| = {worst:.2e}   "
          f"argmax agreement {agree}/{PARITY_TRIALS}")
    assert worst < PARITY_TOL and agree == PARITY_TRIALS, "refine export parity FAILED"
    print(f"exported {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
