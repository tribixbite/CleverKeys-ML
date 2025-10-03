#!/usr/bin/env python3
"""
Reference RNNT greedy decode using the training-time featurizer + NeMo decoder.

- Loads a .nemo (or .ckpt) via the canonical loader
- Reads a swipe trace from the dataset JSONL by word or line number
- Runs normalize -> resample -> feature extraction
- Feeds features into the model encoder and uses NeMo's rnnt_decoder_predictions_tensor
  to produce a reference greedy decode that we can compare with the web decoder.

Usage examples:

  python new/decode_reference.py \
    --checkpoint 9292025script/20251002/rnnt_checkpoints_short_common_20251002_233024/conformer_rnnt_final.nemo \
    --dataset data/train_final_train.jsonl \
    --word companion

  python new/decode_reference.py --checkpoint <.nemo> --dataset data/train_final_train.jsonl --line 123

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Reuse shared loader + featurizer
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
from new.train_transducer_personalized import (
    CONFIG,
    determine_resample_target,
    PersonalizedSwipeFeaturizer,
)

def normalize_points(points):
    norm = []
    if not points:
        return norm
    start_t = float(points[0].get('t', 0.0))
    for idx, pt in enumerate(points):
        rx = float(pt.get('x', 0.5)); ry = float(pt.get('y', 0.5))
        cx = max(-1.0, min(1.0, rx * 2.0 - 1.0))
        cy = max(-1.0, min(1.0, ry * 2.0 - 1.0))
        rt = float(pt.get('t', idx * 10.0))
        norm.append({'x': cx, 'y': cy, 't': max(0.0, rt - start_t)})
    return norm

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


def read_word_trace(dataset_path: Path, word: str) -> Optional[Dict[str, Any]]:
    word_lc = word.lower()
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("word", "").lower() == word_lc and isinstance(rec.get("points"), list):
                return rec
    return None


def read_line_trace(dataset_path: Path, line_num: int) -> Optional[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if i == line_num:
                try:
                    return json.loads(line)
                except Exception:
                    return None
    return None


def to_encoder_inputs(features_t: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """features_t: shape [T, 37] float32 -> encoder inputs (B,F,T) and lengths.
    Nemo ConformerEncoder (as exported) accepts audio_signal=[B,F,T], length=[B].
    """
    if features_t.size == 0:
        feats_bft = torch.zeros((1, 37, 0), dtype=torch.float32)
        lens = torch.tensor([0], dtype=torch.int32)
        return feats_bft, lens
    T = int(features_t.shape[0])
    F = int(features_t.shape[1])
    feats_bft = torch.from_numpy(features_t.astype(np.float32)).transpose(0, 1).unsqueeze(0)  # [1,F,T]
    lens = torch.tensor([T], dtype=torch.int32)
    return feats_bft, lens


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference RNNT greedy decode using training featurizer")
    ap.add_argument("--checkpoint", required=True, help="Path to .nemo or .ckpt")
    ap.add_argument("--dataset", default="data/train_final_train.jsonl", help="Path to dataset JSONL")
    ap.add_argument("--word", help="Word to search for in dataset")
    ap.add_argument("--line", type=int, help="Line number to read from dataset (1-based)")
    ap.add_argument("--max-symbols", type=int, default=15, help="Max symbols per frame")
    ap.add_argument("--beam", type=int, default=0, help="Beam size (0 = greedy)")
    ap.add_argument("--scan", action='store_true', help="Scan dataset for first matching word that decodes correctly")
    ap.add_argument("--scan-limit", type=int, default=5000, help="Max lines to scan when --scan")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    record: Optional[Dict[str, Any]] = None
    if args.scan and args.word:
        # Scan dataset for first instance of the word that decodes correctly
        print(f"Scanning for decodable sample of '{args.word}' (limit={args.scan_limit})...")
        found = None
        count = 0
        with dataset_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                count += 1
                if count > int(args.scan_limit):
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get('word','').lower() != args.word.lower() or not isinstance(rec.get('points'), list):
                    continue
                # Featurize and try quick greedy decode
                norm = normalize_points(rec['points'])
                target = determine_resample_target(len(norm), dict(CONFIG.get('preprocess', {})))
                proc = resample_points(norm, target)
                feats = PersonalizedSwipeFeaturizer()(proc)
                feats_bft, lens = to_encoder_inputs(feats)
                with torch.no_grad():
                    encoded, enc_len = model.encoder(audio_signal=feats_bft, length=lens)
                    preds = model.decoding.rnnt_decoder_predictions_tensor(encoded, enc_len)
                def hyp_to_text(h):
                    if isinstance(h, list) and h:
                        h = h[0]
                    if hasattr(h, 'text'):
                        return h.text
                    return str(h)
                texts = [hyp_to_text(p) for p in preds]
                if texts and texts[0] and isinstance(texts[0], str) and texts[0].lower() == args.word.lower():
                    record = rec
                    print(f"Found matching decode at line {count}")
                    break
        if record is None:
            raise RuntimeError(f"No decodable sample of '{args.word}' found within first {args.scan_limit} lines")
    else:
        if args.word:
            record = read_word_trace(dataset_path, args.word)
            if record is None:
                raise RuntimeError(f"Word '{args.word}' not found in {dataset_path}")
        elif args.line:
            record = read_line_trace(dataset_path, int(args.line))
            if record is None:
                raise RuntimeError(f"Line {args.line} not found or invalid JSON in {dataset_path}")
        else:
            raise SystemExit("Specify --word or --line or --scan --word")

    points = record["points"]
    word = record["word"]
    print(f"Loaded trace for '{word}' with {len(points)} points")

    # Featurize using the same path as training
    norm = normalize_points(points)
    target = determine_resample_target(len(norm), dict(CONFIG.get('preprocess', {})))
    proc = resample_points(norm, target)
    feats = PersonalizedSwipeFeaturizer()(proc)  # [T,37]
    print(f"Features: T={feats.shape[0]} F={feats.shape[1]}")

    # Load trained model (.nemo or .ckpt) using NeMo
    ckpt_path = str(args.checkpoint)
    model = None
    try:
        if ckpt_path.endswith('.ckpt'):
            model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(ckpt_path, map_location='cpu')
        else:
            model = nemo_asr.models.EncDecRNNTModel.restore_from(ckpt_path, map_location='cpu')
    except Exception as e:
        print('Primary load failed, trying restore_from:', e)
        model = nemo_asr.models.EncDecRNNTModel.restore_from(ckpt_path, map_location='cpu')
    model.eval()

    # Configure decoding strategy
    try:
        if int(args.beam) > 0:
            # Switch to beam decoding
            beam_cfg = DictConfig({
                'strategy': 'beam',
                'beam': {
                    'beam_size': int(args.beam),
                    'return_best_hypothesis': True,
                    'score_norm': True,
                    'softmax_temperature': 1.0,
                    'nbest': int(max(1, min(args.beam, 20))),
                },
            })
            if hasattr(model, 'change_decoding_strategy'):
                model.change_decoding_strategy(beam_cfg)
            else:
                # Fallback: mutate cfg and recreate
                if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
                    model.cfg.decoding.strategy = 'beam'
                    model.cfg.decoding.beam = beam_cfg.beam
        else:
            # Greedy
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
                model.cfg.decoding.strategy = 'greedy_batch'
                if hasattr(model.cfg.decoding, 'greedy_batch'):
                    model.cfg.decoding.greedy_batch.max_symbols = int(args.max_symbols)
    except Exception as e:
        print("Warning: failed to configure decoding strategy:", e)

    # Encoder forward (bypass audio preprocessor)
    with torch.no_grad():
        feats_bft, lens = to_encoder_inputs(feats)
        encoded, enc_len = model.encoder(audio_signal=feats_bft, length=lens)
        # NeMo greedy decode on encoded frames
        preds = model.decoding.rnnt_decoder_predictions_tensor(encoded, enc_len)

    # preds may be list[Hypothesis] or list[list[Hypothesis]] depending on version
    def hyp_to_text(h):
        if isinstance(h, list) and h:
            h = h[0]
        if hasattr(h, 'text'):
            return h.text
        if isinstance(h, str):
            return h
        # Fallback: try to map tokens via model.decoding
        try:
            tokens = getattr(h, 'y_sequence', None)
            if tokens is not None:
                ids = [int(t) for t in tokens]
                return model.decoding.decode_tokens_to_str(ids)
        except Exception:
            pass
        return str(h)

    texts = [hyp_to_text(p) for p in preds]
    print("Predictions:", texts)
    if texts:
        print(f"Top prediction: '{texts[0]}' vs label '{word}'")


if __name__ == "__main__":
    main()
