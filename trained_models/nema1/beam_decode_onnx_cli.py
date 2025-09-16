#!/usr/bin/env python3
"""
beam_decode_onnx_cli.py

Desktop CLI for ONNX-based lexicon beam search.
Runs exported models (encoder_int8_qdq.onnx, rnnt_step_fp32.onnx)
with features.npy, words.txt, and runtime_meta.json.
"""

import argparse
import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Any


def load_meta(path: str) -> Tuple[int, int, Dict[str, int]]:
    """Load runtime metadata from JSON file"""
    with open(path, "r") as f:
        j = json.load(f)
    return j["blank_id"], j["unk_id"], j["char_to_id"]


def normalize(w: str) -> str:
    """Normalize word: lowercase and replace curly with straight apostrophe"""
    return w.lower().replace("'", "'")


def build_trie(words: List[str], char_to_id: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    """
    Build trie from word list using character-to-ID mapping.
    Returns (root_node, kept_count)
    """
    root = {"ch": {}, "is": False, "wid": -1}
    kept = 0

    for wid, w0 in enumerate(words):
        w = normalize(w0)
        if any(ch not in char_to_id for ch in w):
            continue

        cur = root
        for ch in w:
            cid = char_to_id[ch]
            cur = cur["ch"].setdefault(cid, {"ch": {}, "is": False, "wid": -1})
        cur["is"] = True
        cur["wid"] = wid
        kept += 1

    print(f"Trie built: {len(words)} words -> {kept} kept ({kept/len(words)*100:.1f}% coverage)")
    return root, kept


def slice_lc(x: np.ndarray, i: int, L: int, H: int) -> np.ndarray:
    """Slice LSTM state: x: (L,N,H) -> (L,1,H) picking index i"""
    out = np.empty((L, 1, H), np.float32)
    out[:, 0, :] = x[:, i, :]
    return out


def rnnt_word_beam(
    encoder_sess: ort.InferenceSession,
    step_sess: ort.InferenceSession,
    feats_bft: np.ndarray,
    F: int, T: int, L: int, H: int, D: int,
    blank_id: int,
    words: List[str],
    trie_root: Dict[str, Any],
    priors: Optional[np.ndarray] = None,
    beam: int = 16,
    prune: int = 6,
    max_sym: int = 20,
    lm_lambda: float = 0.4,
    topk: int = 5
) -> List[Tuple[str, float, float]]:
    """
    RNNT word-level beam search with lexicon constraints.

    Returns list of (word, total_score, rnnt_logp) tuples.
    """
    # Run encoder
    enc_out = encoder_sess.run(None, {
        "features_bft": feats_bft.astype(np.float32),
        "lengths": np.array([T], np.int32),
    })
    enc_btf = enc_out[0][0]  # (T_out, D)
    T_out, Denc = enc_btf.shape
    assert Denc == D, f"Encoder output dim mismatch: {Denc} != {D}"

    # Initialize beam
    beams = [{
        "y": blank_id,
        "h": np.zeros((L, 1, H), np.float32),
        "c": np.zeros((L, 1, H), np.float32),
        "tr": trie_root,
        "lp": 0.0,
        "chars": []
    }]

    # Main beam search loop
    for t in range(T_out):
        for s in range(max_sym):
            beams.sort(key=lambda b: b["lp"], reverse=True)
            act = beams[:beam]
            N = len(act)

            # Prepare batch inputs
            yprev = np.array([b["y"] for b in act], np.int64)
            h0 = np.concatenate([b["h"] for b in act], axis=1)  # (L, N, H)
            c0 = np.concatenate([b["c"] for b in act], axis=1)
            enc_t = np.repeat(enc_btf[t][None, :], N, axis=0)  # (N, D)

            # Run step
            logits, h1, c1 = step_sess.run(None, {
                "y_prev": yprev,
                "h0": h0,
                "c0": c0,
                "enc_t": enc_t
            })
            V = logits.shape[1]

            # Expand beams
            nxt = []
            for i, b in enumerate(act):
                # Blank transition
                lp_blank = float(logits[i, blank_id])
                nxt.append({
                    "y": blank_id,
                    "h": slice_lc(h1, i, L, H),
                    "c": slice_lc(c1, i, L, H),
                    "tr": b["tr"],
                    "lp": b["lp"] + lp_blank,
                    "chars": b["chars"][:]
                })

                # Character transitions (trie-constrained)
                allowed = list(b["tr"]["ch"].keys())
                if allowed:
                    allowed.sort(key=lambda cid: float(logits[i, cid]), reverse=True)
                    for cid in allowed[:min(prune, len(allowed))]:
                        child = b["tr"]["ch"][cid]
                        nxt.append({
                            "y": cid,
                            "h": slice_lc(h1, i, L, H),
                            "c": slice_lc(c1, i, L, H),
                            "tr": child,
                            "lp": b["lp"] + float(logits[i, cid]),
                            "chars": b["chars"] + [cid]
                        })

            nxt.sort(key=lambda b: b["lp"], reverse=True)
            beams = nxt[:beam]

            # Early stopping if best beam emits blank
            if beams and beams[0]["y"] == blank_id:
                break

    # Collect word candidates
    cands = []
    for b in beams:
        if b["tr"]["is"] and b["tr"]["wid"] >= 0:
            wid = b["tr"]["wid"]
            lm = 0.0 if priors is None else float(priors[wid])
            total_score = b["lp"] + lm_lambda * lm
            cands.append((wid, total_score, b["lp"]))

    cands.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicates and format output
    out = []
    seen = set()
    for wid, score, rp in cands:
        if wid in seen:
            continue
        seen.add(wid)
        out.append((words[wid], score, rp))
        if len(out) >= topk:
            break

    return out


def main():
    ap = argparse.ArgumentParser(description="ONNX-based lexicon beam search CLI")
    ap.add_argument("--encoder", required=True, help="Path to encoder ONNX model")
    ap.add_argument("--step", required=True, help="Path to step ONNX model")
    ap.add_argument("--meta", required=True, help="Path to runtime_meta.json")
    ap.add_argument("--words", required=True, help="Path to words.txt")
    ap.add_argument("--priors", default=None, help="Optional path to word_priors.f32")
    ap.add_argument("--features", required=True, help="Path to features.npy (shape [1,F,T])")
    ap.add_argument("--F", type=int, default=37, help="Feature dimension")
    ap.add_argument("--L", type=int, default=2, help="LSTM layers")
    ap.add_argument("--H", type=int, default=320, help="Hidden dimension")
    ap.add_argument("--D", type=int, required=True, help="Encoder output dimension")
    ap.add_argument("--beam", type=int, default=16, help="Beam size")
    ap.add_argument("--prune", type=int, default=6, help="Prune per beam")
    ap.add_argument("--max-sym", type=int, default=20, help="Max symbols per frame")
    ap.add_argument("--lm-lambda", type=float, default=0.4, help="Language model weight")
    ap.add_argument("--topk", type=int, default=5, help="Return top K results")
    args = ap.parse_args()

    # Load metadata and lexicon
    blank_id, _, char_to_id = load_meta(args.meta)
    words = [w for w in open(args.words, "r", encoding="utf-8").read().splitlines() if w]
    trie, kept = build_trie(words, char_to_id)

    priors = None
    if args.priors:
        priors = np.fromfile(args.priors, dtype="<f4")  # little-endian float32
        print(f"Loaded {len(priors)} word priors")

    # Load models
    so = ort.SessionOptions()
    enc = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"], sess_options=so)
    step = ort.InferenceSession(args.step, providers=["CPUExecutionProvider"], sess_options=so)

    # Load features
    feats = np.load(args.features)  # (1, F, T)
    B, F, T = feats.shape
    assert B == 1 and F == args.F, f"Feature shape mismatch: {feats.shape} expected (1, {args.F}, T)"
    feats_bft = feats.astype(np.float32)

    print(f"Running beam search on features shape {feats.shape}")
    print(f"Beam size: {args.beam}, Prune: {args.prune}, Max symbols: {args.max_sym}")
    print(f"LM lambda: {args.lm_lambda}, Top-K: {args.topk}")

    # Run beam search
    top = rnnt_word_beam(
        enc, step, feats_bft, F, T, args.L, args.H, args.D,
        blank_id, words, trie, priors,
        beam=args.beam, prune=args.prune, max_sym=args.max_sym,
        lm_lambda=args.lm_lambda, topk=args.topk
    )

    # Output results
    print("\nTop predictions:")
    for i, (w, score, rp) in enumerate(top, 1):
        print(f"{i:2d}. {w:<15} score={score:7.3f} (model={rp:7.3f})")


if __name__ == "__main__":
    main()

# Usage example:
# python beam_decode_onnx_cli.py \
#   --encoder encoder_int8_qdq.onnx \
#   --step rnnt_step_fp32.onnx \
#   --meta runtime_meta.json \
#   --words words.txt \
#   --priors word_priors.f32 \
#   --features sample_features.npy \
#   --D 256