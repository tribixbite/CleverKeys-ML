#!/usr/bin/env python3
"""Test with correctly exported models and metadata"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Any

def load_metadata(meta_path: str) -> Tuple[int, List[str], Dict[str, int]]:
    """Load runtime metadata with correct blank_id"""
    with open(meta_path, "r") as f:
        meta = json.load(f)

    blank_id = meta["blank_id"]
    tokens = meta["tokens"]

    # Build char_to_id mapping
    # NeMo puts blank at position 29, so regular chars are at 0-27
    char_to_id = {}
    for i, token in enumerate(tokens):
        if token == "<blank>":
            continue  # Skip blank, it's at position 29
        elif token == "<unk>":
            char_to_id[token] = 28
        elif token == "'":
            char_to_id[token] = 0
        else:
            # a-z are at positions 1-26
            char_to_id[token] = ord(token) - ord('a') + 1

    return blank_id, tokens, char_to_id

def build_trie(words: List[str], char_to_id: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    """Build trie for word constraints"""
    root = {"ch": {}, "is": False, "wid": -1}
    kept = 0

    for wid, word in enumerate(words):
        word = word.lower().replace("'", "'")
        if any(ch not in char_to_id for ch in word):
            continue

        cur = root
        for ch in word:
            cid = char_to_id[ch]
            cur = cur["ch"].setdefault(cid, {"ch": {}, "is": False, "wid": -1})
        cur["is"] = True
        cur["wid"] = wid
        kept += 1

    return root, kept

# Load models
print("Loading models...")
encoder_sess = ort.InferenceSession(
    "../trained_models/nema1/onnx_rare_words_epoch80/encoder.onnx",
    providers=["CPUExecutionProvider"]
)
decoder_sess = ort.InferenceSession(
    "../trained_models/nema1/rnnt_step_correct_v2.onnx",
    providers=["CPUExecutionProvider"]
)

# Load metadata
blank_id, tokens, char_to_id = load_metadata("../trained_models/nema1/runtime_meta_correct.json")
print(f"Blank ID: {blank_id}")
print(f"Vocab size: {len(tokens)}")

# Load words
with open("../trained_models/nema1/words.txt", "r") as f:
    words = [w.strip() for w in f if w.strip()]

trie, kept = build_trie(words, char_to_id)
print(f"Trie: {kept}/{len(words)} words")

# Load test features (for word "is")
features = np.load("test_features.npy")
T = features.shape[2]

# Run encoder
enc_out = encoder_sess.run(None, {
    "features": features.astype(np.float32),
    "features_length": np.array([T], np.int64)
})
enc_bdt = enc_out[0]
enc_btf = enc_bdt.T if len(enc_bdt.shape) == 2 else enc_bdt[0].T
T_out = enc_btf.shape[0]

print(f"\nEncoder: {T} frames -> {T_out} frames")

# Greedy decode first
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = blank_id
decoded = []

print("\nGreedy decoding:")
for t in range(T_out):
    enc_t = enc_btf[t:t+1]

    outputs = decoder_sess.run(None, {
        "y_prev": np.array([y_prev], np.int64),
        "h0": h,
        "c0": c,
        "enc_t": enc_t
    })

    logits = outputs[0]
    h = outputs[1]
    c = outputs[2]

    if len(logits.shape) > 2:
        logits = logits.squeeze()
    if len(logits.shape) == 0:
        logits = logits.reshape(-1)

    y_pred = np.argmax(logits)

    if y_pred != blank_id:
        if y_pred < len(tokens) - 1:  # -1 because blank is at end
            if y_pred == 0:
                char = "'"
            elif 1 <= y_pred <= 26:
                char = chr(ord('a') + y_pred - 1)
            elif y_pred == 28:
                char = "<unk>"
            else:
                char = f"?{y_pred}"
            decoded.append(char)
            if len(decoded) <= 5:
                print(f"  t={t:2d}: '{char}' (id={y_pred})")
        y_prev = y_pred
    else:
        if t == 0 or len(decoded) == 0:
            print(f"  t={t:2d}: <blank>")
        y_prev = blank_id

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: 'is'")

# Now run beam search with lexicon constraints
print("\n" + "=" * 60)
print("Running beam search with lexicon constraints...")

def slice_lc(x: np.ndarray, i: int) -> np.ndarray:
    """Slice LSTM state"""
    L, B, H = x.shape
    out = np.empty((L, 1, H), np.float32)
    out[:, 0, :] = x[:, i, :]
    return out

beam_size = 16
prune = 6
max_sym = 20

beams = [{
    "y": blank_id,
    "h": np.zeros((L, 1, H), np.float32),
    "c": np.zeros((L, 1, H), np.float32),
    "tr": trie,
    "lp": 0.0,
    "chars": []
}]

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

        outputs = decoder_sess.run(None, {
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

        # Expand beams
        nxt = []
        for i, b in enumerate(act):
            # Blank transition
            lp_blank = float(logits[i, blank_id])
            nxt.append({
                "y": blank_id,
                "h": slice_lc(h1, i),
                "c": slice_lc(c1, i),
                "tr": b["tr"],
                "lp": b["lp"] + lp_blank,
                "chars": b["chars"][:]
            })

            # Character transitions
            allowed = list(b["tr"]["ch"].keys())
            if allowed:
                allowed.sort(key=lambda cid: float(logits[i, cid]), reverse=True)
                for cid in allowed[:prune]:
                    child = b["tr"]["ch"][cid]
                    nxt.append({
                        "y": cid,
                        "h": slice_lc(h1, i),
                        "c": slice_lc(c1, i),
                        "tr": child,
                        "lp": b["lp"] + float(logits[i, cid]),
                        "chars": b["chars"] + [cid]
                    })

        nxt.sort(key=lambda b: b["lp"], reverse=True)
        beams = nxt[:beam_size]

        if beams and beams[0]["y"] == blank_id:
            break

# Get results
results = []
seen = set()
for b in beams:
    if b["tr"]["is"] and b["tr"]["wid"] >= 0:
        wid = b["tr"]["wid"]
        if wid not in seen:
            seen.add(wid)
            results.append((words[wid], b["lp"]))

results.sort(key=lambda x: x[1], reverse=True)

print("\nTop predictions from beam search:")
for i, (word, score) in enumerate(results[:5], 1):
    print(f"  {i}. {word:15} (score={score:.2f})")

print(f"\nExpected: 'is'")