#!/usr/bin/env python3
"""Beam-eval a trained checkpoint (or exported ONNX) through the SAME harness used
for every committed FUTO-comparison number.

Reuses ``futo_viterbi_beam`` from ``futo_decoder_ceiling.py`` (the beam the Kotlin
``CtcBeamDecoder`` is golden-parity-tested against) and the trie/lexicon loaders
from ``futo_decoder_eval.py``, so the printed number is directly comparable to the
published baselines and is what the Kotlin decoder reproduces on-device.

Usage:
  python eval_beam.py --ckpt ckpt/base/best.pt --test data/val_hwsfuto.jsonl
  python eval_beam.py --onnx ctc_swipe_encoder.onnx --test data/test_hwsfuto.jsonl

Audit fixes: #15 (--out per-trace JSONL dump), #16 (--workdir pathing, layout
defaults to the script's directory).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futo_decoder_eval import (featurize, greedy_ctc, len_stratum,  # noqa: E402
                               load_combined_vocab, load_layout, load_test,
                               rank_of, Tally)
from futo_decoder_ceiling import (ENC_BETA, ENC_BETA_PRUNE, ENC_GAMMA,  # noqa: E402
                                  ENC_GAMMA_PRUNE, ENC_LAMBDA, futo_viterbi_beam,
                                  slice_emissions)
from model import MAX_KEYS, CtcSwipeEncoder  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402


def _pad(keys: np.ndarray, mask: np.ndarray):
    """Pad ``[K,2]`` centers / ``[K]`` mask up to the export-time ``MAX_KEYS``."""
    k = keys.shape[0]
    kp = np.zeros((MAX_KEYS, 2), np.float32)
    kp[:k] = keys
    mp = np.zeros((MAX_KEYS,), bool)
    mp[:k] = mask
    return kp, mp


class TorchEncoder:
    """Run the torch checkpoint directly (reference path)."""

    def __init__(self, ckpt_path: Path, device: str = "cpu") -> None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model = CtcSwipeEncoder(ch=ck.get("ch", 96),
                                     embed_hid=ck.get("embed_hid", 96)).eval().to(device)
        self.model.load_state_dict(ck["model"])
        self.device = device

    @torch.no_grad()
    def forward(self, feats: np.ndarray, keys: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kp, mp = _pad(keys, mask)
        log_e, _, _ = self.model(
            torch.from_numpy(feats[None]).to(self.device),
            torch.from_numpy(kp[None]).to(self.device),
            torch.from_numpy(mp[None]).to(self.device))
        return log_e.cpu().numpy()[0]                    # [32, 65]


class OnnxEncoder:
    """Run the exported ONNX graph (the on-device path)."""

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])

    def forward(self, feats: np.ndarray, keys: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kp, mp = _pad(keys, mask)
        out = self.sess.run(["log_emissions"],
                            {"features": feats[None], "layout_keys": kp[None],
                             "layout_mask": mp[None]})
        return out[0][0]                                 # [32, 65]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--vocab", default="data/futo_en_wordlist.combined")
    ap.add_argument("--test", default="data/val_hwsfuto.jsonl")
    ap.add_argument("--out", default="", help="per-trace JSONL dump (audit fix #15)")
    ap.add_argument("--device", default="cpu", help="torch device for --ckpt")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=200)
    args = ap.parse_args()
    if bool(args.ckpt) == bool(args.onnx):
        raise SystemExit("pass exactly one of --ckpt / --onnx")

    letters, key_centers = load_layout(args.layout)
    num_letters = len(letters)
    trie = load_combined_vocab(resolve(args.workdir, args.vocab))
    print(f"trie: {trie.num_words} words")
    enc = (TorchEncoder(resolve(args.workdir, args.ckpt), args.device) if args.ckpt
           else OnnxEncoder(resolve(args.workdir, args.onnx)))
    rows = load_test(resolve(args.workdir, args.test))
    if args.limit:
        rows = rows[: args.limit]
    mask = np.ones((num_letters,), bool)
    out_f = open(resolve(args.workdir, args.out), "w") if args.out else None

    g_tal, b_tal = Tally(), Tally()
    strat = {"<=3": Tally(), "4+": Tally()}
    t0 = time.time()
    for i, (word, xs, ys, ts) in enumerate(rows):
        target = word.lower()
        feats = featurize(xs, ys, ts)
        full = enc.forward(feats, key_centers, mask)          # [32, 65]
        lp = slice_emissions(full, num_letters, MAX_KEYS)     # [32, 27], blank -> 26
        greedy = greedy_ctc(lp, letters, num_letters)
        beam = futo_viterbi_beam(lp, letters, num_letters, trie,
                                 args.beam_width, args.top_k,
                                 ENC_GAMMA, ENC_LAMBDA, ENC_BETA,
                                 ENC_GAMMA_PRUNE, ENC_BETA_PRUNE)
        words = [w for w, _ in beam]
        g_tal.add(0 if greedy == target else -1)
        r = rank_of(target, words)
        b_tal.add(r)
        strat[len_stratum(target)].add(r)
        if out_f is not None:
            out_f.write(json.dumps({"idx": i, "word": word, "greedy": greedy,
                                    "topk": [[w, float(s)] for w, s in beam],
                                    "rank": r}) + "\n")
        if (i + 1) % args.progress == 0:
            if out_f is not None:
                out_f.flush()
            print(f"  [{i + 1}/{len(rows)}] beam {b_tal.row()}  "
                  f"({(i + 1) / (time.time() - t0):.1f} tr/s)", flush=True)
    if out_f is not None:
        out_f.close()
    print("=" * 70)
    print(f"n={b_tal.n}  GREEDY t1 {g_tal.t1 / max(g_tal.n, 1) * 100:.2f}%")
    print(f"BEAM top-1/3/5   {b_tal.row()}")
    for s in ("<=3", "4+"):
        print(f"  {s:<4} n={strat[s].n:<5} {strat[s].row()}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
