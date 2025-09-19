#!/usr/bin/env python3
"""Test with corrected vocabulary size"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Any

def load_meta_fixed(path: str) -> Tuple[int, int, Dict[str, int], int]:
    """Load runtime metadata with vocab size fix"""
    with open(path, "r") as f:
        j = json.load(f)

    # Model actually uses vocab_size=30 but metadata only has 29
    # Token 29 appears to be an additional blank or padding token
    actual_vocab_size = 30

    return j["blank_id"], j["unk_id"], j["char_to_id"], actual_vocab_size

def build_trie(words: List[str], char_to_id: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    """Build trie from word list"""
    root = {"ch": {}, "is": False, "wid": -1}
    kept = 0

    for wid, w0 in enumerate(words):
        w = w0.lower().replace("'", "'")
        if any(ch not in char_to_id for ch in w):
            continue

        cur = root
        for ch in w:
            cid = char_to_id[ch]
            cur = cur["ch"].setdefault(cid, {"ch": {}, "is": False, "wid": -1})
        cur["is"] = True
        cur["wid"] = wid
        kept += 1

    return root, kept

# Load models
encoder_sess = ort.InferenceSession("encoder_fresh.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

# Load metadata with fix
blank_id, _, char_to_id, vocab_size = load_meta_fixed("../trained_models/nema1/runtime_meta.json")
print(f"Blank ID: {blank_id}, Vocab size: {vocab_size}")

# Load words
with open("../trained_models/nema1/words.txt", "r") as f:
    words = [w.strip() for w in f if w.strip()]

trie, kept = build_trie(words, char_to_id)
print(f"Trie: {kept}/{len(words)} words")

# Load test features
features = np.load("test_features.npy")
T = features.shape[2]

# Run encoder
enc_out = encoder_sess.run(None, {
    "features_bft": features.astype(np.float32),
    "lengths": np.array([T], np.int32)
})
enc_btf = enc_out[0][0].T  # (T_out, D)
T_out = enc_btf.shape[0]
print(f"Encoder: {T} -> {T_out} frames")

# Simple beam search treating token 29 as non-blank
L, H = 2, 320
beam_size = 16
max_sym = 20

beams = [{
    "y": blank_id,
    "h": np.zeros((L, 1, H), np.float32),
    "c": np.zeros((L, 1, H), np.float32),
    "tr": trie,
    "lp": 0.0,
    "chars": []
}]

print("\nBeam search with vocab_size=30:")
for t in range(T_out):
    for s in range(max_sym):
        beams.sort(key=lambda b: b["lp"], reverse=True)
        act = beams[:beam_size]

        if not act:
            break

        N = len(act)

        # Batch process
        yprev = np.array([b["y"] for b in act], np.int64)
        h0 = np.concatenate([b["h"] for b in act], axis=1)
        c0 = np.concatenate([b["c"] for b in act], axis=1)
        enc_t = np.repeat(enc_btf[t][None, :], N, axis=0)

        outputs = step_sess.run(None, {
            "y_prev": yprev,
            "h0": h0,
            "c0": c0,
            "enc_t": enc_t
        })

        logits = outputs[0]
        h1 = outputs[1]
        c1 = outputs[2]

        # Fix logits shape
        if len(logits.shape) > 2:
            logits = logits.squeeze()
            if len(logits.shape) == 1:
                logits = logits.reshape(1, -1)

        nxt = []
        for i, b in enumerate(act):
            # Get h and c for this beam
            h_i = h1[:, i:i+1, :]
            c_i = c1[:, i:i+1, :]

            # Blank transition (only token 0)
            lp_blank = float(logits[i, blank_id])
            nxt.append({
                "y": blank_id,
                "h": h_i,
                "c": c_i,
                "tr": b["tr"],
                "lp": b["lp"] + lp_blank,
                "chars": b["chars"][:]
            })

            # Character transitions - SKIP token 29
            allowed = list(b["tr"]["ch"].keys())
            if allowed:
                allowed.sort(key=lambda cid: float(logits[i, cid]), reverse=True)
                for cid in allowed[:6]:  # Top 6
                    if cid == 29:  # Skip token 29
                        continue
                    child = b["tr"]["ch"][cid]
                    nxt.append({
                        "y": cid,
                        "h": h_i,
                        "c": c_i,
                        "tr": child,
                        "lp": b["lp"] + float(logits[i, cid]),
                        "chars": b["chars"] + [cid]
                    })

        nxt.sort(key=lambda b: b["lp"], reverse=True)
        beams = nxt[:beam_size]

        if beams and beams[0]["y"] == blank_id:
            break

    if t % 5 == 0:
        print(f"  t={t}: best beam has {len(beams[0]['chars'])} chars")

# Get results
print("\nTop words from beam search:")
results = []
for b in beams[:10]:
    if b["tr"]["is"] and b["tr"]["wid"] >= 0:
        wid = b["tr"]["wid"]
        word = words[wid]
        score = b["lp"]
        results.append((word, score))

results.sort(key=lambda x: x[1], reverse=True)
for i, (word, score) in enumerate(results[:5], 1):
    print(f"  {i}. {word:15} (score={score:.2f})")

print(f"\nExpected: 'is'")