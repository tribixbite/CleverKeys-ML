#!/usr/bin/env python3
"""
Verify PyTorch (NeMo) decoding vs the ONNX preprocessing pipeline.

Loads a .ckpt/.nemo via the personalized trainer’s model class, runs the same
normalize -> adaptive resample -> 37D features pipeline used by
decode_onnx_stateful.py, then decodes with NeMo (transcribe / rnnt decode).

If PyTorch predicts correctly but ONNX does not, the issue is ONNX export or
ONNX inference loop. If PyTorch also fails, the checkpoint likely cannot
decode that swipe reliably.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from new.train_transducer_personalized import (
    PersonalizedRNNTModel,
    determine_resample_target,
    PersonalizedSwipeFeaturizer,
    CONFIG,
)
from new.decode_onnx_stateful import read_word_trace, read_line_trace


def normalize_points(points):
    out = []
    if not points:
        return out
    start_t = float(points[0].get('t', 0.0))
    for idx, pt in enumerate(points):
        rx = float(pt.get('x', 0.5)); ry = float(pt.get('y', 0.5))
        cx = max(-1.0, min(1.0, rx * 2.0 - 1.0))
        cy = max(-1.0, min(1.0, ry * 2.0 - 1.0))
        rt = float(pt.get('t', idx * 10.0))
        out.append({'x': cx, 'y': cy, 't': max(0.0, rt - start_t)})
    return out


def resample_points(points, target_count):
    if target_count <= 0 or len(points) == 0:
        return []
    if len(points) == target_count:
        return [dict(p) for p in points]
    resampled = []
    first_time = points[0]['t']; last_time = points[-1]['t']
    duration = max(last_time - first_time, 1.0)
    step = duration / max(target_count - 1, 1)
    src_idx = 0
    for i in range(target_count):
        target_time = last_time if i == target_count - 1 else first_time + step * i
        while src_idx < len(points) - 2 and points[src_idx + 1]['t'] < target_time:
            src_idx += 1
        p1 = points[src_idx]
        p2 = points[min(src_idx + 1, len(points) - 1)]
        span = max(p2['t'] - p1['t'], 1.0)
        alpha = max(0.0, min(1.0, (target_time - p1['t']) / span))
        x = p1['x'] + (p2['x'] - p1['x']) * alpha
        y = p1['y'] + (p2['y'] - p1['y']) * alpha
        resampled.append({'x': x, 'y': y, 't': target_time})
    return resampled


def main() -> None:
    ap = argparse.ArgumentParser(description='Verify PyTorch NeMo decode matches ONNX preprocessing')
    ap.add_argument('--checkpoint', required=True, help='Path to .ckpt or .nemo')
    ap.add_argument('--dataset', default='data/train_final_train.jsonl')
    ap.add_argument('--word', help='Word to decode')
    ap.add_argument('--line', type=int, help='Line number (1-based) to decode')
    ap.add_argument('--dump-features', help='Optional path to save [1,F,T] features for comparison')
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if args.word:
        rec = read_word_trace(dataset_path, args.word)
        if rec is None:
            raise RuntimeError(f"Word '{args.word}' not found in {dataset_path}")
        label = args.word
    elif args.line:
        rec = read_line_trace(dataset_path, int(args.line))
        if rec is None:
            raise RuntimeError(f"Line {args.line} not found or invalid JSON")
        label = rec.get('word', '')
    else:
        raise SystemExit('Specify --word or --line')

    # Preprocess: normalize -> adaptive resample -> 37D features
    norm = normalize_points(rec['points'])
    target = determine_resample_target(len(norm), dict(CONFIG.get('preprocess', {})))
    proc = resample_points(norm, target)
    feats_tf = PersonalizedSwipeFeaturizer()(proc)  # [T,F]
    print(f"Features: T={feats_tf.shape[0]} F={feats_tf.shape[1]}")

    # [1,F,T] for ONNX; for NeMo transcribe we pass [T,F]
    feats_bft = feats_tf.astype(np.float32).T.reshape(1, feats_tf.shape[1], feats_tf.shape[0])
    if args.dump_features:
        np.save(args.dump_features, feats_bft)

    # Load NeMo model
    print(f"Loading PyTorch model from {args.checkpoint}...")
    model = PersonalizedRNNTModel.load_from_checkpoint(args.checkpoint, map_location='cpu') if args.checkpoint.endswith('.ckpt') else PersonalizedRNNTModel.restore_from(args.checkpoint, map_location='cpu')
    model.eval(); model.freeze()

    # Decode using NeMo high-level API
    features = torch.from_numpy(feats_tf).float()  # [T,F]
    with torch.no_grad():
        try:
            hypotheses = model.transcribe([features], batch_size=1)
            pred = hypotheses[0] if isinstance(hypotheses, list) else str(hypotheses)
        except Exception:
            # Fallback: manual encode + rnnt_decoder_predictions_tensor
            feats_bft_t = torch.from_numpy(feats_bft)
            lens = torch.tensor([feats_tf.shape[0]], dtype=torch.int32)
            encoded, enc_len = model.encoder(audio_signal=feats_bft_t, length=lens)
            preds = model.decoding.rnnt_decoder_predictions_tensor(encoded, enc_len)
            def hyp_to_text(h):
                if isinstance(h, list) and h:
                    h = h[0]
                return h.text if hasattr(h, 'text') else str(h)
            pred = hyp_to_text(preds[0])

    print('-' * 60)
    print(f"PyTorch Model Prediction: '{pred}'")
    print(f"Label:                   '{label}'")
    print('-' * 60)


if __name__ == '__main__':
    main()

