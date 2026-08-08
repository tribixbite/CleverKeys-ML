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
from futo_decoder_ceiling import (DEC_BETA, DEC_BETA_PRUNE, DEC_GAMMA,  # noqa: E402
                                  DEC_GAMMA_PRUNE, DEC_LAMBDA, ENC_BETA,
                                  ENC_BETA_PRUNE, ENC_GAMMA, ENC_GAMMA_PRUNE,
                                  ENC_LAMBDA, build_decoder_input,
                                  futo_viterbi_beam, slice_emissions)
from model import MAX_KEYS, CtcRefineHead, encoder_from_checkpoint  # noqa: E402
from paths import DEFAULT_LAYOUT, DEFAULT_WORKDIR, resolve  # noqa: E402
import seal  # noqa: E402

#: scoring.json presets: encoder-only vs encoder+refinement (guide §7 / §11 step 6).
SCORING = {
    "enc": (ENC_GAMMA, ENC_LAMBDA, ENC_BETA, ENC_GAMMA_PRUNE, ENC_BETA_PRUNE),
    "dec": (DEC_GAMMA, DEC_LAMBDA, DEC_BETA, DEC_GAMMA_PRUNE, DEC_BETA_PRUNE),
}


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
        self.model = encoder_from_checkpoint(ck).eval().to(device)
        self.model.load_state_dict(ck["model"])
        self.device = device

    @torch.no_grad()
    def forward(self, feats: np.ndarray, keys: np.ndarray, mask: np.ndarray):
        """-> ``(log_emissions[32,65], coefficients[32,64], lambda[32])``."""
        kp, mp = _pad(keys, mask)
        log_e, coeff, lam = self.model(
            torch.from_numpy(feats[None]).to(self.device),
            torch.from_numpy(kp[None]).to(self.device),
            torch.from_numpy(mp[None]).to(self.device))
        return (log_e.cpu().numpy()[0], coeff.cpu().numpy()[0],
                lam.cpu().numpy()[0].reshape(-1))


class OnnxEncoder:
    """Run the exported ONNX graph (the on-device path)."""

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])

    def forward(self, feats: np.ndarray, keys: np.ndarray, mask: np.ndarray):
        """-> ``(log_emissions[32,65], coefficients[32,64], lambda[32])``."""
        kp, mp = _pad(keys, mask)
        out = self.sess.run(["log_emissions", "coefficients", "lambda"],
                            {"features": feats[None], "layout_keys": kp[None],
                             "layout_mask": mp[None]})
        return out[0][0], out[1][0], out[2][0].reshape(-1)


class TorchRefiner:
    """Phase-2 refinement head: ``[32,92]`` -> refined ``[32,27]`` (guide §11)."""

    def __init__(self, ckpt_path: Path, device: str = "cpu") -> None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.head = CtcRefineHead(num_letters=ck.get("num_letters", 26),
                                  hidden=ck.get("hidden", 128)).eval().to(device)
        self.head.load_state_dict(ck["head"])
        self.device = device

    @torch.no_grad()
    def forward(self, din: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(din[None]).to(self.device)
        return self.head(x).cpu().numpy()[0]             # [32, 27]


class OnnxRefiner:
    """Phase-2 refinement head loaded from its exported ONNX graph."""

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])

    def forward(self, din: np.ndarray) -> np.ndarray:
        return self.sess.run(["refined_log_probs"],
                             {"decoder_input": din[None]})[0][0]   # [32, 27]


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
    ap.add_argument("--refine-ckpt", default="", dest="refine_ckpt",
                    help="phase-2 refinement head checkpoint (guide §11); its "
                         "refined [32,27] output REPLACES the sliced emissions")
    ap.add_argument("--refine-onnx", default="", dest="refine_onnx",
                    help="phase-2 refinement head as exported ONNX")
    ap.add_argument("--scoring", choices=["auto", "enc", "dec"], default="auto",
                    help="scoring.json preset; auto = dec when refining, else enc")
    ap.add_argument("--preset", default="",
                    help="override --scoring with explicit "
                         "'gamma,lambda,beta,gammaPrune,betaPrune'")
    ap.add_argument("--device", default="cpu", help="torch device for --ckpt")
    ap.add_argument("--beam-width", type=int, default=100, dest="beam_width")
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=200)
    seal.add_argument(ap)
    args = ap.parse_args()
    if bool(args.ckpt) == bool(args.onnx):
        raise SystemExit("pass exactly one of --ckpt / --onnx")
    if args.refine_ckpt and args.refine_onnx:
        raise SystemExit("pass at most one of --refine-ckpt / --refine-onnx")

    letters, key_centers = load_layout(args.layout)
    num_letters = len(letters)
    trie = load_combined_vocab(resolve(args.workdir, args.vocab))
    print(f"trie: {trie.num_words} words")
    enc = (TorchEncoder(resolve(args.workdir, args.ckpt), args.device) if args.ckpt
           else OnnxEncoder(resolve(args.workdir, args.onnx)))
    refiner = None
    if args.refine_ckpt:
        refiner = TorchRefiner(resolve(args.workdir, args.refine_ckpt), args.device)
    elif args.refine_onnx:
        refiner = OnnxRefiner(resolve(args.workdir, args.refine_onnx))
    preset = args.scoring if args.scoring != "auto" else ("dec" if refiner else "enc")
    if args.preset:
        # Explicit params win over the named presets. This is the path that
        # cross-checks a re-tuned preset through the REAL per-row decoder rather
        # than through sweep_scoring's analytic re-scoring (Phase E, E1).
        vals = [float(v) for v in args.preset.split(",")]
        if len(vals) != 5:
            raise SystemExit("--preset needs 5 floats: "
                             "gamma,lambda,beta,gammaPrune,betaPrune")
        SCORING["custom"] = (vals[0], vals[1], vals[2], vals[3], vals[4])
        preset = "custom"
    gamma, lam_w, beta, gamma_prune, beta_prune = SCORING[preset]
    print(f"scoring preset '{preset}': gamma={gamma} lambda={lam_w} beta={beta} "
          f"gammaPrune={gamma_prune} betaPrune={beta_prune}"
          + ("  [refined emissions]" if refiner else ""))
    rows = load_test(resolve(args.workdir, args.test))
    seal.check(rows, args.unseal_test, what="eval_beam.py")
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
        full, coeff, lam = enc.forward(feats, key_centers, mask)   # [32,65] [32,64] [32]
        lp = slice_emissions(full, num_letters, MAX_KEYS)      # [32, 27], blank -> 26
        if refiner is not None:
            # Refined log-probs REPLACE the emissions before the beam (guide §11).
            lp = refiner.forward(build_decoder_input(lp, coeff, lam))   # [32, 27]
        greedy = greedy_ctc(lp, letters, num_letters)
        beam = futo_viterbi_beam(lp, letters, num_letters, trie,
                                 args.beam_width, args.top_k,
                                 gamma, lam_w, beta, gamma_prune, beta_prune)
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
