#!/usr/bin/env python3
"""Freeze model-backed golden cases for the Kotlin parity test.

Synthetic paths -> features -> ONNX emissions -> harness greedy + beam. The
output matches the ``ctc_golden.json`` ``"beam"`` case schema that CleverKeys'
``CtcParityTest`` reads, plus the raw ``points -> features -> emissions`` pairs a
future ONNX-backed ``CtcEmissionModel`` parity test needs.

Emissions are stored as the sliced ``[32,27]`` contract view (blank relocated
from full-head column 64 to column 26), which is exactly what
``CtcEmissions.sliceFromHead`` hands ``CtcBeamDecoder``.

Audit fix #16: --workdir pathing; layout defaults to the script's directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import featurize, greedy_ctc, load_layout, LexTrie  # noqa: E402
from futo_decoder_ceiling import (ENC_BETA, ENC_BETA_PRUNE, ENC_GAMMA,  # noqa: E402
                                  ENC_GAMMA_PRUNE, ENC_LAMBDA, futo_viterbi_beam,
                                  slice_emissions)
from model import MAX_KEYS, T_OUT  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402

LEXICON = [("cat", 150.0), ("car", 180.0), ("cart", 120.0), ("care", 140.0),
           ("the", 250.0), ("hello", 160.0), ("keyboard", 110.0)]
WORDS = ["cat", "the", "hello", "keyboard"]
BEAM_WIDTH = 32
TOP_K = 4


def ideal_path(by_letter, word: str, pts_per_seg: int = 12
               ) -> Tuple[List[float], List[float], List[float]]:
    """Deterministic straight-line path through the word's key centers, 60 Hz stamps."""
    cs = [by_letter[c] for c in word]
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(cs[:-1], cs[1:]):
        for j in range(pts_per_seg):
            f = j / pts_per_seg
            xs.append(float(a[0] + f * (b[0] - a[0])))
            ys.append(float(a[1] + f * (b[1] - a[1])))
    xs.append(float(cs[-1][0]))
    ys.append(float(cs[-1][1]))
    ts = [i * (1000.0 / 60.0) for i in range(len(xs))]
    return xs, ys, ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--onnx", default="ctc_swipe_encoder.onnx")
    ap.add_argument("--out", default="ctc_model_golden.json")
    args = ap.parse_args()

    letters, centers = load_layout(args.layout)
    num_letters = len(letters)
    by_letter = {l: centers[i] for i, l in enumerate(letters)}
    sess = ort.InferenceSession(str(resolve(args.workdir, args.onnx)),
                                providers=["CPUExecutionProvider"])

    trie = LexTrie()
    for w, f in LEXICON:
        trie.insert(w, f)

    keys = np.zeros((MAX_KEYS, 2), np.float32)
    keys[:num_letters] = centers
    mask = np.zeros((MAX_KEYS,), bool)
    mask[:num_letters] = True

    cases = []
    for word in WORDS:
        xs, ys, ts = ideal_path(by_letter, word)
        feats = featurize(xs, ys, ts)                                 # [2,64]
        full = sess.run(["log_emissions"],
                        {"features": feats[None], "layout_keys": keys[None],
                         "layout_mask": mask[None]})[0][0]            # [32,65]
        lp = slice_emissions(full, num_letters, MAX_KEYS)             # [32,27]
        greedy = greedy_ctc(lp, letters, num_letters)
        topk = futo_viterbi_beam(lp, letters, num_letters, trie, BEAM_WIDTH, TOP_K,
                                 ENC_GAMMA, ENC_LAMBDA, ENC_BETA,
                                 ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)
        cases.append({
            "kind": "beam", "name": f"model_{word}",
            "alphabet": "".join(letters), "frames": T_OUT,
            "numClasses": num_letters + 1,
            "points": {"x": xs, "y": ys, "t": ts},
            "features": [float(v) for v in feats.reshape(-1)],        # [128] x-row then y-row
            "emissions": [[float(v) for v in row] for row in lp],
            "lexicon": [[w, f] for w, f in LEXICON],
            "params": {"gamma": ENC_GAMMA, "lambda": ENC_LAMBDA, "beta": ENC_BETA,
                       "alpha": 0.0, "gammaPrune": ENC_GAMMA_PRUNE,
                       "betaPrune": ENC_BETA_PRUNE, "beamWidth": BEAM_WIDTH,
                       "topK": TOP_K},
            "greedy": greedy,
            "topk": [[w, s] for w, s in topk],
        })
    out_path = resolve(args.workdir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"cases": cases}, indent=1))
    print(f"wrote {out_path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
