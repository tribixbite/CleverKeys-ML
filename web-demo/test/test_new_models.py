#!/usr/bin/env python3
"""
Test new ONNX models with companion and other words
"""

import json
import numpy as np
import onnxruntime as ort
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'new'))
from train_transducer_personalized import PersonalizedSwipeFeaturizer, determine_resample_target, resample_points


def load_test_words(word_list, max_search=50000):
    """Load specific words from training set"""
    results = {}
    data_path = '../../data/train_final_train.jsonl'

    with open(data_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if i > max_search:
                break
            data = json.loads(line)
            if data['word'] in word_list:
                results[data['word']] = {
                    'line': i,
                    'points': data['points']
                }
                print(f"Found '{data['word']}' at line {i}")
                if len(results) == len(word_list):
                    break

    return results


def process_swipe(points, featurizer, preprocess_cfg):
    """Process swipe points through full pipeline"""
    # 1. Transform coordinates from [0, 1] to [-1, 1]
    transformed_points = []
    for pt in points:
        transformed_points.append({
            'x': pt['x'] * 2.0 - 1.0,
            'y': pt['y'] * 2.0 - 1.0,
            't': pt['t']
        })

    # 2. Determine resample target
    target_len = determine_resample_target(len(transformed_points), preprocess_cfg)

    # 3. Resample points
    resampled_points = resample_points(transformed_points, target_len)

    # 4. Extract features
    features = featurizer(resampled_points)

    return features, len(transformed_points), target_len


def run_inference(features, encoder, decoder, meta):
    """Run greedy inference on features"""
    # Prepare encoder input - transpose to [batch, feat_dim, seq_len]
    encoder_input = features.T[np.newaxis, :, :]  # [1, 37, num_frames]
    length = np.array([features.shape[0]], dtype=np.int64)

    # Run encoder
    encoder_outputs = encoder.run(
        None,
        {
            'audio_signal': encoder_input.astype(np.float32),
            'length': length
        }
    )

    encoded = encoder_outputs[0]  # [1, encoder_dim, time]

    # Simple greedy decode
    blank_id = meta.get('blank_id', 29)
    max_symbols = 24

    decoded_tokens = []

    # Initialize decoder states
    pred_layers = 1
    pred_hidden = 320  # Updated from metadata if different
    state_h = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
    state_c = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)

    last_token = 0  # Start with <blank>

    # Process each encoder frame
    for t in range(encoded.shape[2]):
        encoder_frame = encoded[:, :, t:t+1]  # [1, encoder_dim, 1]

        symbols_emitted = 0
        max_symbols_per_frame = 6

        while symbols_emitted < max_symbols_per_frame and len(decoded_tokens) < max_symbols:
            # Run decoder
            try:
                decoder_outputs = decoder.run(
                    None,
                    {
                        'encoder_outputs': encoder_frame,
                        'targets': np.array([[last_token]], dtype=np.int32),
                        'target_length': np.array([1], dtype=np.int32),
                        'input_states_1': state_h,
                        'input_states_2': state_c
                    }
                )
            except Exception as e:
                print(f"Decoder error: {e}")
                print(f"Encoder frame shape: {encoder_frame.shape}")
                print(f"State shapes: h={state_h.shape}, c={state_c.shape}")
                raise

            logits = decoder_outputs[0]  # [batch, time, decoder_time, vocab]
            state_h = decoder_outputs[2]
            state_c = decoder_outputs[3]

            # Get prediction
            if len(logits.shape) == 4:
                probs = logits[0, 0, 0, :]  # Get first time step
            else:
                probs = logits[0, 0, :]

            predicted_token = np.argmax(probs)

            if predicted_token == blank_id:
                # Blank - move to next frame
                break
            else:
                decoded_tokens.append(predicted_token)
                last_token = predicted_token
                symbols_emitted += 1

    return decoded_tokens


def tokens_to_text(tokens, meta):
    """Convert token IDs to text"""
    return ''.join([meta['id_to_char'][str(t)] for t in tokens if str(t) in meta['id_to_char']])


def main():
    # Words to test - include 'companion' as requested
    test_words = ['companion', 'hello', 'person', 'the', 'and', 'you', 'world', 'test', 'good', 'time']

    print("Loading test data...")
    word_data = load_test_words(test_words)
    print(f"\nFound {len(word_data)}/{len(test_words)} words from dataset")

    # Initialize featurizer
    featurizer = PersonalizedSwipeFeaturizer()

    # Preprocessing config
    preprocess_cfg = {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    }

    # Load NEW ONNX models from rnnt_new_latest
    model_dir = '../models/rnnt_new_latest'
    print(f"\nLoading models from: {model_dir}")
    encoder = ort.InferenceSession(os.path.join(model_dir, 'encoder.onnx'))
    decoder = ort.InferenceSession(os.path.join(model_dir, 'decoder_joint.onnx'))

    # Load runtime meta
    with open(os.path.join(model_dir, 'runtime_meta.json'), 'r') as f:
        meta = json.load(f)

    print(f"Vocab size: {meta.get('vocab_size', 'unknown')}")
    print(f"Blank ID: {meta.get('blank_id', 'unknown')}")
    print(f"Tokens: {meta.get('tokens', [])[:5]}...")

    # Check decoder hidden size from metadata
    if 'pred_hidden' in meta:
        pred_hidden = meta['pred_hidden']
        print(f"Decoder hidden size: {pred_hidden}")

    print("\n" + "="*70)
    print("TESTING NEW MODELS")
    print("="*70)

    results = []

    for word, data in word_data.items():
        print(f"\nTesting: '{word}' (line {data['line']})")
        print("-" * 40)

        points = data['points']
        print(f"Original points: {len(points)}")

        # Check coordinate ranges
        x_vals = [p['x'] for p in points]
        y_vals = [p['y'] for p in points]
        print(f"X range: [{min(x_vals):.3f}, {max(x_vals):.3f}]")
        print(f"Y range: [{min(y_vals):.3f}, {max(y_vals):.3f}]")

        # Process through pipeline
        features, orig_len, resample_len = process_swipe(points, featurizer, preprocess_cfg)
        print(f"Resampled: {orig_len} → {resample_len} points")
        print(f"Features shape: {features.shape}")

        # Run inference
        try:
            tokens = run_inference(features, encoder, decoder, meta)
            predicted = tokens_to_text(tokens, meta)
        except Exception as e:
            print(f"ERROR during inference: {e}")
            predicted = "ERROR"
            tokens = []

        # Check results
        success = predicted == word
        results.append({
            'word': word,
            'predicted': predicted,
            'success': success,
            'tokens': tokens
        })

        print(f"Predicted: '{predicted}'")
        print(f"Tokens: {tokens[:10]}")
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")

    print("\nDetailed results:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} '{r['word']}' → '{r['predicted']}'")

    # Special check for companion
    if 'companion' in [r['word'] for r in results]:
        companion_result = next(r for r in results if r['word'] == 'companion')
        print(f"\n{'='*70}")
        print("COMPANION TEST:")
        print(f"{'='*70}")
        if companion_result['success']:
            print(f"✅ SUCCESS: 'companion' correctly predicted!")
        else:
            print(f"❌ FAILED: 'companion' → '{companion_result['predicted']}'")
            print(f"Tokens: {companion_result['tokens']}")


if __name__ == '__main__':
    main()