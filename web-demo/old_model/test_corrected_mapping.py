#!/usr/bin/env python3
"""Test with properly corrected character mapping"""

import json
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Any

def load_metadata_corrected(meta_path: str) -> Tuple[int, List[str], Dict[str, int], Dict[int, str]]:
    """Load runtime metadata and build correct mappings"""
    with open(meta_path, "r") as f:
        meta = json.load(f)

    blank_id = meta["blank_id"]
    tokens = meta["tokens"]

    # The tokens list shows the logical order, but model indices are:
    # 0: ' (apostrophe)
    # 1-26: a-z
    # 27: unused (model outputs 30 dims but we have 29 tokens)
    # 28: <unk>
    # 29: <blank>

    char_to_id = {}
    id_to_char = {}

    # Map based on model's actual output indices
    id_to_char[29] = "<blank>"
    char_to_id["<blank>"] = 29

    id_to_char[28] = "<unk>"
    char_to_id["<unk>"] = 28

    id_to_char[0] = "'"
    char_to_id["'"] = 0

    for i in range(26):
        char = chr(ord('a') + i)
        id_to_char[i + 1] = char
        char_to_id[char] = i + 1

    # Index 27 is unused
    id_to_char[27] = "<?27>"

    return blank_id, tokens, char_to_id, id_to_char

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
    "../trained_models/nema1/rnnt_step_final.onnx",
    providers=["CPUExecutionProvider"]
)

# Load metadata with corrected mapping
blank_id, tokens, char_to_id, id_to_char = load_metadata_corrected(
    "../trained_models/nema1/runtime_meta_final.json"
)

print(f"Blank ID: {blank_id}")
print(f"Vocab tokens: {len(tokens)}")
print(f"Model output dims: 30 (indices 0-29)")
print()
print("Character mapping:")
for i in range(30):
    char = id_to_char.get(i, f"?{i}")
    print(f"  {i:2d}: '{char}'")
print()

# Verify specific mappings
print("Key mappings:")
print(f"  'i' -> {char_to_id.get('i', 'NOT FOUND')}")
print(f"  's' -> {char_to_id.get('s', 'NOT FOUND')}")
print(f"  blank -> {blank_id}")

# Load words
with open("../trained_models/nema1/words.txt", "r") as f:
    words = [w.strip() for w in f if w.strip()]

trie, kept = build_trie(words, char_to_id)
print(f"\nTrie: {kept}/{len(words)} words")

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

# Greedy decode
L, H = 2, 320
h = np.zeros((L, 1, H), np.float32)
c = np.zeros((L, 1, H), np.float32)
y_prev = blank_id
decoded = []
all_predictions = []

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

    # Check shape
    if t == 0:
        print(f"  Logits shape: {logits.shape}")

    y_pred = np.argmax(logits)
    all_predictions.append(y_pred)

    if y_pred != blank_id:
        char = id_to_char.get(y_pred, f"?{y_pred}")
        decoded.append(char if char not in ["<blank>", "<unk>", "<?27>"] else f"[{char}]")
        if len(decoded) <= 10:
            print(f"  t={t:2d}: pred={y_pred:2d} -> '{char}'")
        y_prev = y_pred
    else:
        if t == 0 or len(decoded) == 0:
            print(f"  t={t:2d}: <blank>")
        y_prev = blank_id

    # Show scores for 'i' and 's'
    if t < 5:
        i_id = char_to_id.get('i')
        s_id = char_to_id.get('s')
        i_score = float(logits[i_id])
        s_score = float(logits[s_id])
        blank_score = float(logits[blank_id])
        print(f"       scores: i({i_id})={i_score:.2f}, s({s_id})={s_score:.2f}, blank({blank_id})={blank_score:.2f}")

print(f"\nDecoded: '{''.join(decoded)}'")
print(f"Expected: 'is'")
print(f"\nAll predictions: {all_predictions[:10]}...")

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
            char_seq = ''.join([id_to_char.get(cid, f"?{cid}") for cid in b["chars"]])
            results.append((words[wid], char_seq, b["lp"]))

results.sort(key=lambda x: x[2], reverse=True)

print("\nTop predictions from beam search:")
for i, (word, chars, score) in enumerate(results[:10], 1):
    print(f"  {i}. {word:15} (chars: '{chars}', score={score:.2f})")

print(f"\nExpected: 'is'")