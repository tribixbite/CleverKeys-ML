#!/usr/bin/env python3
"""Corrected beam search ignoring token 29"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Any

def load_meta(path: str) -> Tuple[int, int, Dict[str, int]]:
    """Load runtime metadata"""
    with open(path, "r") as f:
        j = json.load(f)
    return j["blank_id"], j["unk_id"], j["char_to_id"]

def normalize(w: str) -> str:
    """Normalize word"""
    return w.lower().replace("'", "'")

def build_trie(words: List[str], char_to_id: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    """Build trie from word list"""
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

    return root, kept

def slice_lc(x: np.ndarray, i: int, L: int, H: int) -> np.ndarray:
    """Slice LSTM state"""
    out = np.empty((L, 1, H), np.float32)
    out[:, 0, :] = x[:, i, :]
    return out

# Load models
encoder_sess = ort.InferenceSession("encoder_fresh.onnx", providers=["CPUExecutionProvider"])
step_sess = ort.InferenceSession("rnnt_step_fresh.onnx", providers=["CPUExecutionProvider"])

# Load metadata
blank_id, _, char_to_id = load_meta("../trained_models/nema1/runtime_meta.json")
with open("../trained_models/nema1/words.txt", "r") as f:
    words = [w.strip() for w in f if w.strip()]
trie, kept = build_trie(words, char_to_id)

print(f"Blank ID: {blank_id}")
print(f"Trie: {kept}/{len(words)} words")

# Load test features
features = np.load("test_features.npy")
T = features.shape[2]

# Run encoder
enc_out = encoder_sess.run(None, {
    "features_bft": features.astype(np.float32),
    "lengths": np.array([T], np.int32)
})
enc_bdt = enc_out[0]
enc_btf = enc_bdt[0].T
T_out = enc_btf.shape[0]
print(f"Encoder: {T} -> {T_out} frames")

# Beam search parameters
L, H, D = 2, 320, 256
beam_size = 32
prune = 8
max_sym = 30

# Initialize beam
beams = [{
    "y": blank_id,
    "h": np.zeros((L, 1, H), np.float32),
    "c": np.zeros((L, 1, H), np.float32),
    "tr": trie,
    "lp": 0.0,
    "chars": [],
    "word": ""
}]

print("\nBeam search (ignoring token 29):")
for t in range(T_out):
    for s in range(max_sym):
        beams.sort(key=lambda b: b["lp"], reverse=True)
        act = beams[:beam_size]
        N = len(act)

        # Prepare batch
        yprev = np.array([b["y"] for b in act], np.int64)
        h0 = np.concatenate([b["h"] for b in act], axis=1)
        c0 = np.concatenate([b["c"] for b in act], axis=1)
        enc_t = np.repeat(enc_btf[t][None, :], N, axis=0)

        # Run step
        outputs = step_sess.run(None, {
            "y_prev": yprev,
            "h0": h0,
            "c0": c0,
            "enc_t": enc_t
        })

        logits = outputs[0]
        h1 = outputs[1]
        c1 = outputs[2]

        # Fix shape
        if len(logits.shape) > 2:
            logits = logits.squeeze()
            if len(logits.shape) == 1:
                logits = logits.reshape(1, -1)

        # IMPORTANT: Mask out token 29 (set to very negative value)
        logits[:, 29] = -1000.0

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
                "chars": b["chars"][:],
                "word": b["word"]
            })

            # Character transitions
            allowed = list(b["tr"]["ch"].keys())
            if allowed:
                # Sort by logit score
                allowed.sort(key=lambda cid: float(logits[i, cid]), reverse=True)
                for cid in allowed[:prune]:
                    if cid == 29:  # Skip token 29
                        continue
                    child = b["tr"]["ch"][cid]
                    char = [ch for ch, id in char_to_id.items() if id == cid][0]
                    nxt.append({
                        "y": cid,
                        "h": slice_lc(h1, i, L, H),
                        "c": slice_lc(c1, i, L, H),
                        "tr": child,
                        "lp": b["lp"] + float(logits[i, cid]),
                        "chars": b["chars"] + [cid],
                        "word": b["word"] + char
                    })

        nxt.sort(key=lambda b: b["lp"], reverse=True)
        beams = nxt[:beam_size]

        # Early stop if best beam is blank
        if beams and beams[0]["y"] == blank_id:
            break

    # Progress
    if t % 5 == 0:
        best_word = beams[0]["word"] if beams else ""
        print(f"  t={t}: best partial word: '{best_word}'")

print("\nFinal beam search results:")
cands = []
for b in beams:
    if b["tr"]["is"] and b["tr"]["wid"] >= 0:
        wid = b["tr"]["wid"]
        word = words[wid]
        score = b["lp"]
        cands.append((word, score))

cands.sort(key=lambda x: x[1], reverse=True)

print("Top predictions:")
for i, (word, score) in enumerate(cands[:5], 1):
    print(f"  {i}. {word:15} (score={score:.2f})")

print(f"\nExpected: 'is'")