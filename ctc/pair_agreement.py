#!/usr/bin/env python3
"""Label-free pair-compatibility gate: per-frame argmax agreement of two encoders.

The mechanism Phase K isolated and then confirmed prospectively (PHASE_K.md
§4.3, §8.5): CTC never pins *which* alignment a model concentrates on, so two
models' emissions may only be averaged before the beam if they share the
alignment gauge — and per-frame argmax agreement ≥ ~95 % on unlabelled traces
predicts that they do. Agreement < 95 % predicts the broken band (ensemble
greedy 9–20 %, top-1 several points below either member).

**This computes the gate from the EXPORTED graphs, using no labels**, so it can
be run — and its verdict committed — before any beam decode of the mix. That
ordering is the whole point of the blind protocol; Phase K ran the measurement
ad hoc, and this script exists so Phase L (and anyone after it) runs the same
measurement the same way, reproducibly.

Reported, matching the three numbers PHASE_K §8.5 committed:
  * ``agreement``      — all frames, all rows;
  * ``blank_pattern``  — agreement on "is this frame blank?" alone;
  * ``letters_both``   — agreement on the letter identity, over frames where
                         BOTH models emit a non-blank.

Usage:
  python3 pair_agreement.py --onnx a.onnx,b.onnx --rows 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import load_layout  # noqa: E402
from model import MAX_KEYS  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

#: The Phase-K threshold, validated prospectively in §8.5.
GATE = 0.95


def emissions(sess: ort.InferenceSession, feats: np.ndarray,
              keys: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """-> ``[N,T',65]`` log-emissions.

    One row per call: the shipped graph is exported with a fixed batch of 1
    (the app's contract), exactly as `eval_beam.OnnxEncoder` calls it.
    """
    kp, mp = keys[None], mask[None]
    out: List[np.ndarray] = [
        sess.run(["log_emissions"],
                 {"features": f[None], "layout_keys": kp, "layout_mask": mp})[0][0]
        for f in feats]
    return np.stack(out)


def agreement_stats(la: np.ndarray, lb: np.ndarray, blank: int) -> Dict[str, float]:
    """Per-frame argmax agreement, split the way PHASE_K §8.5 reported it."""
    aa, ab = la.argmax(-1), lb.argmax(-1)
    same = aa == ab
    blk_a, blk_b = aa == blank, ab == blank
    both_letters = (~blk_a) & (~blk_b)
    return {
        "frames": int(aa.size),
        "agreement": float(same.mean()),
        "blank_pattern": float((blk_a == blk_b).mean()),
        "letters_both": float(same[both_letters].mean())
        if both_letters.any() else float("nan"),
        "letter_frames": int(both_letters.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--onnx", required=True,
                    help="exactly two comma-separated exported encoders")
    ap.add_argument("--features", default="cache/val.npz",
                    help="unlabelled feature source; labels are never read")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--json-out", default="", dest="json_out")
    args = ap.parse_args()

    paths = [resolve(args.workdir, p.strip()) for p in args.onnx.split(",")
             if p.strip()]
    if len(paths) != 2:
        raise SystemExit(f"--onnx needs exactly two graphs, got {len(paths)}")
    if "test" in Path(args.features).name:
        raise SystemExit("refusing to read test features: test-2400 is sealed")

    letters, centers = load_layout(args.layout)
    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:len(letters)] = np.asarray(centers, np.float32)[:len(letters)]
    mask = np.zeros((MAX_KEYS,), bool)
    mask[:len(letters)] = True

    with np.load(resolve(args.workdir, args.features)) as d:
        feats = np.asarray(d["features"][:args.rows], np.float32)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    la, lb = (emissions(ort.InferenceSession(
        str(p), so, providers=["CPUExecutionProvider"]), feats, keys, mask)
        for p in paths)
    if la.shape != lb.shape:
        raise SystemExit(f"emission shape mismatch {la.shape} vs {lb.shape}")

    st = agreement_stats(la, lb, blank=MAX_KEYS)
    st["rows"] = int(len(feats))
    st["onnx"] = [p.name for p in paths]
    st["gate"] = GATE
    st["verdict"] = "PASS" if st["agreement"] >= GATE else "FAIL"
    print(json.dumps(st, indent=1))
    print(f"per-frame argmax agreement = {st['agreement'] * 100:.1f}%  "
          f"(blank-pattern {st['blank_pattern'] * 100:.1f}%, "
          f"letters-where-both-emit {st['letters_both'] * 100:.1f}%, "
          f"first {st['rows']} rows, labels unused) -> {st['verdict']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(st, indent=1))
    return 0 if st["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
