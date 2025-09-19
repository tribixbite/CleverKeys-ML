#!/usr/bin/env python3
"""Complete decoder test with proper beam search"""

import json
import numpy as np
from typing import List
from test_with_resampling import ResamplingDecoder

class CompleteDecoder(ResamplingDecoder):
    """Extended decoder with beam search"""

    def decode(self, raw_points: List[dict], beam_size: int = 16, top_k: int = 5) -> List[tuple]:
        """Decode with beam search"""
        # Preprocess
        processed = self.preprocess_points(raw_points)

        # Compute features
        features = self.compute_features_batch(processed)
        features_bft = features.T[np.newaxis, :, :]
        T = features_bft.shape[2]

        # Run encoder
        enc_out = self.encoder_session.run(None, {
            "features_bft": features_bft.astype(np.float32),
            "lengths": np.array([T], np.int32)
        })
        enc_bdt = enc_out[0]
        enc_btf = enc_bdt[0].T
        T_out = enc_btf.shape[0]

        # Beam search
        L, H = 2, 320
        beams = [{
            "y": self.blank_id,
            "h": np.zeros((L, 1, H), np.float32),
            "c": np.zeros((L, 1, H), np.float32),
            "tr": self.trie_root,
            "lp": 0.0,
            "chars": []
        }]

        for t in range(T_out):
            for s in range(20):  # max symbols per frame
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

                outputs = self.decoder_session.run(None, {
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

                # Mask token 29 (ONNX export artifact)
                if logits.shape[1] > 29:
                    logits[:, 29] = -1000.0

                # Expand beams
                nxt = []
                for i, b in enumerate(act):
                    # Extract state
                    h_i = h1[:, i:i+1, :]
                    c_i = c1[:, i:i+1, :]

                    # Blank
                    lp_blank = float(logits[i, self.blank_id])
                    nxt.append({
                        "y": self.blank_id,
                        "h": h_i,
                        "c": c_i,
                        "tr": b["tr"],
                        "lp": b["lp"] + lp_blank,
                        "chars": b["chars"][:]
                    })

                    # Characters
                    allowed = list(b["tr"]["children"].keys())
                    if allowed:
                        allowed.sort(key=lambda cid: float(logits[i, cid]), reverse=True)
                        for cid in allowed[:8]:
                            if cid >= 29:
                                continue
                            child = b["tr"]["children"][cid]
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

                if beams and beams[0]["y"] == self.blank_id:
                    break

        # Collect results
        results = []
        seen = set()
        for b in beams:
            if b["tr"]["is_word"] and b["tr"]["word_id"] >= 0:
                wid = b["tr"]["word_id"]
                if wid not in seen:
                    seen.add(wid)
                    results.append((self.words[wid], b["lp"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Test with validation data
decoder = CompleteDecoder(
    encoder_path='encoder_fresh.onnx',
    decoder_path='rnnt_step_fresh.onnx',
    runtime_meta_path='../trained_models/nema1/runtime_meta.json',
    words_path='../trained_models/nema1/words.txt'
)

val_file = '../trained_models/nema1/personalized_tuning/20250918_105357/val_short_common.jsonl'

print("Testing validation samples with complete decoder:")
print("=" * 60)

correct_count = 0
total_count = 0

with open(val_file, 'r') as f:
    for idx, line in enumerate(f):
        if idx >= 10:
            break

        data = json.loads(line)
        word = data['word']
        points = data['points']

        print(f"\nSample {idx + 1}: '{word}' ({len(points)} points)")

        results = decoder.decode(points, beam_size=32, top_k=5)

        if results:
            for i, (pred_word, score) in enumerate(results[:3], 1):
                match = "✓✓✓" if pred_word == word else ""
                print(f"  {i}. {pred_word:15} (score={score:8.2f}) {match}")

            if results[0][0] == word:
                correct_count += 1
                print(f"  ✓ CORRECT!")
        else:
            print(f"  No predictions")

        total_count += 1

print(f"\n{'=' * 60}")
print(f"Accuracy: {correct_count}/{total_count} = {correct_count/total_count*100:.1f}%")