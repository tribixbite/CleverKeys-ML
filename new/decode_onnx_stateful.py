#!/usr/bin/env python3
"""
Stateful RNN-T ONNX decoder (encoder + decoder_joint) on dataset swipes.

Implements the correct RNNT step loop with explicit LSTM state management:
 - Runs encoder.onnx once on features [1, F, T]
 - Initializes predictor states h/c as zeros [L, 1, H] using runtime_meta.json
 - For each encoder time step t, repeatedly calls decoder_joint.onnx with:
     { encoder_outputs[:, :, t:t+1], last_token, state_h, state_c }
   until a blank is predicted; feeds output_states_* into the next call.

Usage:
  python new/decode_onnx_stateful.py \
    --model-dir web-demo/models/best_latest \
    --dataset data/train_final_train.jsonl --word companion

  python new/decode_onnx_stateful.py \
    --model-dir web-demo/models/best_latest \
    --dataset data/train_final_train.jsonl --line 123
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort

# Import featurizer pipeline from the trainer to ensure parity
from new.train_transducer_personalized import (
    determine_resample_target,
    PersonalizedSwipeFeaturizer,
    CONFIG,
)


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


def read_word_trace(dataset_path: Path, word: str) -> Optional[Dict[str, Any]]:
    target = word.lower()
    with dataset_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('word', '').lower() == target and isinstance(rec.get('points'), list):
                return rec
    return None


def read_line_trace(dataset_path: Path, line_num: int) -> Optional[Dict[str, Any]]:
    with dataset_path.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh, 1):
            if i == line_num:
                try:
                    return json.loads(line)
                except Exception:
                    return None
    return None


def to_bft(features_tf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[T,F] -> ([1,F,T], [1])"""
    if features_tf.size == 0:
        return np.zeros((1, 37, 0), np.float32), np.array([0], dtype=np.int64)
    T, F = features_tf.shape
    bft = features_tf.astype(np.float32).T.reshape(1, F, T)
    return bft, np.array([T], dtype=np.int64)


def pick(name_list, preferred):
    for k in preferred:
        if k in name_list:
            return k
    return name_list[0] if name_list else None


def run_encoder(sess: ort.InferenceSession, feats_bft: np.ndarray, lengths: np.ndarray):
    in_names = list(sess.get_inputs())
    out_names = list(sess.get_outputs())
    in0 = pick([x.name for x in in_names], ['audio_signal', 'features_bft', 'input'])
    in1 = pick([x.name for x in in_names], ['length', 'lengths'])
    out0 = pick([x.name for x in out_names], ['outputs', 'encoded_btf', 'encoded'])
    out1 = pick([x.name for x in out_names], ['encoded_lengths', 'lengths'])
    feeds = {}
    feeds[in0] = feats_bft
    # Length type
    exp_dtype = [x for x in in_names if x.name == in1][0].type
    if exp_dtype == 'tensor(int32)':
        feeds[in1] = lengths.astype(np.int32)
    else:
        feeds[in1] = lengths.astype(np.int64)
    out = sess.run([out0, out1], feeds)
    encoded = out[0]  # expect [1, D, T'] or [1, T', D]
    enc_len = int(out[1].reshape(-1)[0])
    return encoded, enc_len


def run_decoder_joint(
    sess: ort.InferenceSession,
    encoder_frame: np.ndarray,  # [1, D, 1]
    target_seq: np.ndarray,  # [1, U] int
    state_h: np.ndarray,  # [L,1,H]
    state_c: np.ndarray,  # [L,1,H]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = list(sess.get_inputs())
    in_names = [x.name for x in inputs]
    out_names = [x.name for x in sess.get_outputs()]
    n_encoder = pick(in_names, ['encoder_outputs', 'encoder_output'])
    n_targets = pick(in_names, ['targets', 'y_prev'])
    n_tlen = pick(in_names, ['target_length', 'tlen'])
    n_h = pick(in_names, ['input_states_1', 'h0'])
    n_c = pick(in_names, ['input_states_2', 'c0'])
    # Determine expected integer dtypes
    def exp_dtype(name: str) -> str:
        for inp in inputs:
            if inp.name == name:
                return inp.type
        return 'tensor(int32)'
    tgt_dtype = exp_dtype(n_targets)
    len_dtype = exp_dtype(n_tlen)
    # Cast provided sequence to expected dtype
    tgt_arr = target_seq.astype(np.int64 if tgt_dtype == 'tensor(int64)' else np.int32)
    tlen_arr = np.array([tgt_arr.shape[1]], dtype=np.int64 if len_dtype == 'tensor(int64)' else np.int32)
    feeds = {
        n_encoder: encoder_frame.astype(np.float32),
        n_targets: tgt_arr,
        n_tlen: tlen_arr,
        n_h: state_h.astype(np.float32),
        n_c: state_c.astype(np.float32),
    }
    # Outputs: logits + new states
    n_logits = pick(out_names, ['outputs', 'logits'])
    n_ho = pick(out_names, ['output_states_1', 'h1'])
    n_co = pick(out_names, ['output_states_2', 'c1'])
    outs = sess.run([n_logits, n_ho, n_co], feeds)
    return outs[0], outs[1], outs[2]


def main() -> None:
    ap = argparse.ArgumentParser(description='Stateful RNNT ONNX decoder on dataset swipes')
    ap.add_argument('--model-dir', required=True, help='Directory with encoder.onnx, decoder_joint.onnx, runtime_meta.json')
    ap.add_argument('--dataset', default='data/train_final_train.jsonl')
    ap.add_argument('--word', help='Word to decode from dataset')
    ap.add_argument('--line', type=int, help='Line number (1-based) to decode from dataset')
    ap.add_argument('--max-symbols', type=int, default=15, help='Max symbols per encoder frame')
    ap.add_argument('--targets', choices=['cumulative','last'], default='cumulative', help='Predictor targets feeding mode')
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    encoder_path = model_dir / 'encoder.onnx'
    decoder_path = model_dir / 'decoder_joint.onnx'
    meta_path = model_dir / 'runtime_meta.json'
    if not (encoder_path.exists() and decoder_path.exists() and meta_path.exists()):
        raise FileNotFoundError('Missing encoder/decoder_joint/runtime_meta in model dir')

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    tokens = meta.get('tokens')
    id_to_char = meta.get('id_to_char') or {}
    joint_blank_id = int(meta.get('blank_id', 0))
    dec_cfg = (meta.get('decoder_config') or {})
    num_layers = int(dec_cfg.get('num_layers', 1))
    hidden_size = int(dec_cfg.get('hidden_size', 256))
    # Predictor mapping: joint-space (includes RNNT blank) -> predictor-space (blankless)
    pred = meta.get('predictor') or {}
    joint2pred = None
    if pred and 'label_map' in pred and 'joint2pred' in pred['label_map']:
        joint2pred = pred['label_map']['joint2pred']
    # BOS in predictor space defaults to index of '<blank>' token (pad) if available, else 0
    bos_pred_id = 0
    if isinstance(tokens, list) and '<blank>' in tokens:
        bos_pred_id = int(tokens.index('<blank>'))

    # Read dataset record
    dataset_path = Path(args.dataset)
    if args.word:
        rec = read_word_trace(dataset_path, args.word)
        if rec is None:
            raise RuntimeError(f"Word '{args.word}' not found")
    elif args.line:
        rec = read_line_trace(dataset_path, int(args.line))
        if rec is None:
            raise RuntimeError(f"Line {args.line} not found or invalid JSON")
    else:
        raise SystemExit('Specify --word or --line')

    # Featurize
    norm = normalize_points(rec['points'])
    target = determine_resample_target(len(norm), dict(CONFIG.get('preprocess', {})))
    proc = resample_points(norm, target)
    feats_tf = PersonalizedSwipeFeaturizer()(proc)  # [T,F]
    print(f"Features: T={feats_tf.shape[0]} F={feats_tf.shape[1]}")
    feats_bft, lengths = to_bft(feats_tf)

    # Create sessions
    sess_opts = ort.SessionOptions()
    encoder_sess = ort.InferenceSession(str(encoder_path), sess_opts, providers=['CPUExecutionProvider'])
    decoder_sess = ort.InferenceSession(str(decoder_path), sess_opts, providers=['CPUExecutionProvider'])

    # Run encoder once
    encoded, enc_len = run_encoder(encoder_sess, feats_bft, lengths)
    # Normalize encoder layout to [1, D, T]
    if encoded.ndim == 3 and encoded.shape[1] < encoded.shape[2]:
        # [1,T,D] -> [1,D,T]
        encoded = np.transpose(encoded, (0, 2, 1))

    D = encoded.shape[1]
    Tprime = min(enc_len, encoded.shape[2])

    # Initialize predictor state
    h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    last_joint_token: Optional[int] = None  # last predicted in joint space; None means BOS
    # Predictor-space cumulative sequence (start with BOS)
    pred_seq = [bos_pred_id]
    hyp: list[int] = []

    # RNNT decode loop
    for t in range(Tprime):
        frame = encoded[:, :, t:t+1].astype(np.float32)
        emitted = 0
        while emitted < int(args.max_symbols):
            # Build predictor targets per mode
            if args.targets == 'cumulative':
                target_seq = np.array([pred_seq], dtype=np.int32)
            else:
                # last-only
                last_pid = pred_seq[-1] if pred_seq else bos_pred_id
                target_seq = np.array([[last_pid]], dtype=np.int32)

            logits, h_new, c_new = run_decoder_joint(decoder_sess, frame, target_seq, h, c)
            # logits may be [V] or [1,V]
            logv = logits.reshape(-1)
            pred_id = int(np.argmax(logv))
            h, c = h_new, c_new
            if pred_id == joint_blank_id:
                break
            hyp.append(pred_id)
            last_joint_token = pred_id
            # Append to predictor-space cumulative sequence (skip joint blank)
            if joint2pred is not None:
                if 0 <= pred_id < len(joint2pred):
                    pid = int(joint2pred[pred_id])
                    if pid >= 0:
                        pred_seq.append(pid)
            else:
                # derive mapping by skipping joint blank index
                if pred_id != joint_blank_id:
                    pid = pred_id if pred_id < joint_blank_id else pred_id - 1
                    pred_seq.append(pid)
            emitted += 1

    # Map tokens to text
    if tokens and isinstance(tokens, list):
        text = ''.join(tokens[i] if 0 <= i < len(tokens) else '' for i in hyp)
    else:
        text = ''.join(id_to_char.get(str(i), '') for i in hyp)

    target_word = rec.get('word', '')
    print(f"Prediction: '{text}' vs label '{target_word}'")


if __name__ == '__main__':
    main()
