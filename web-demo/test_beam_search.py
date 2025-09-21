#!/usr/bin/env python
"""Test beam search inference with the exported stateful models."""

import json
import numpy as np
import onnxruntime as ort
import sys
sys.path.append('../new')

from swipe_data_utils import KeyboardGrid, SwipeFeaturizer

class BeamHypothesis:
    def __init__(self, tokens, score, h_state, c_state, last_token):
        self.tokens = tokens
        self.score = score
        self.h_state = h_state
        self.c_state = c_state
        self.last_token = last_token

def run_beam_search_inference(features, beam_width=5):
    """Run beam search inference with ONNX models."""
    # Load models
    encoder_session = ort.InferenceSession('models/encoder.onnx')
    decoder_session = ort.InferenceSession('models/decoder.onnx')
    joint_session = ort.InferenceSession('models/joint.onnx')

    # Load metadata
    with open('models/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    blank_id = meta['blank_id']
    num_layers = meta['decoder_config']['num_layers']
    hidden_size = meta['decoder_config']['hidden_size']
    vocab_size = meta['vocab_size']

    # Prepare encoder input
    batch_size = 1
    time_steps = features.shape[0]
    feat_dim = features.shape[1]

    # Reshape to [batch, features, time]
    audio_signal = features.T.reshape(1, feat_dim, time_steps).astype(np.float32)
    length = np.array([time_steps], dtype=np.int64)

    # Run encoder
    encoder_outputs = encoder_session.run(None, {'audio_signal': audio_signal, 'length': length})
    encoded = encoder_outputs[0]
    encoded_len = encoder_outputs[1][0]

    # Transpose if needed
    if encoded.shape[1] > encoded.shape[2]:
        encoded = np.transpose(encoded, (0, 2, 1))

    # Initialize beam with single empty hypothesis
    h_init = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    c_init = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    beam = [BeamHypothesis([], 0.0, h_init, c_init, blank_id)]

    # Beam search
    for t in range(min(encoded_len, 30)):  # Limit steps
        encoder_frame = encoded[:, t:t+1, :]
        all_candidates = []

        for hyp in beam:
            # Run decoder for this hypothesis
            input_tokens = np.array([[hyp.last_token]], dtype=np.int64)
            decoder_outputs = decoder_session.run(
                None,
                {
                    'input_tokens': input_tokens,
                    'h_in': hyp.h_state,
                    'c_in': hyp.c_state
                }
            )
            decoder_out = decoder_outputs[0]
            h_next = decoder_outputs[1]
            c_next = decoder_outputs[2]

            # Run joint network
            joint_outputs = joint_session.run(
                None,
                {
                    'encoder_output': encoder_frame,
                    'decoder_output': decoder_out
                }
            )
            logits = joint_outputs[0][0, 0, :]

            # Apply log softmax
            logits_exp = np.exp(logits - np.max(logits))
            log_probs = logits - np.max(logits) - np.log(np.sum(logits_exp))

            # Consider top-k tokens
            top_k = min(10, vocab_size)
            top_indices = np.argsort(log_probs)[-top_k:]

            for token_id in top_indices:
                log_prob = log_probs[token_id]

                if token_id == blank_id:
                    # Continue with same tokens
                    new_hyp = BeamHypothesis(
                        hyp.tokens,
                        hyp.score + log_prob,
                        hyp.h_state,  # Keep same state for blank
                        hyp.c_state,
                        hyp.last_token
                    )
                else:
                    # Add new token
                    new_hyp = BeamHypothesis(
                        hyp.tokens + [token_id],
                        hyp.score + log_prob,
                        h_next,
                        c_next,
                        token_id
                    )
                all_candidates.append(new_hyp)

        # Prune beam
        all_candidates.sort(key=lambda h: h.score, reverse=True)
        beam = all_candidates[:beam_width]

        # Early stop if all hypotheses are long enough
        if all(len(h.tokens) >= 20 for h in beam):
            break

    # Get best hypothesis
    best_hyp = max(beam, key=lambda h: h.score / max(len(h.tokens), 1))
    return best_hyp.tokens, meta

def test_beam_search():
    """Test beam search on validation samples."""
    # Load samples
    samples = []
    with open('../data/train_final_val.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Test first 10
                break
            samples.append(json.loads(line))

    grid = KeyboardGrid()
    featurizer = SwipeFeaturizer(grid)

    correct = 0
    for i, sample in enumerate(samples):
        print(f"\n=== Sample {i+1}: '{sample['word']}' ===")

        # Preprocess
        features = featurizer(sample['points'])

        # Run beam search
        tokens, meta = run_beam_search_inference(features, beam_width=5)

        # Convert to text
        predicted = ''.join([meta['tokens'][t] for t in tokens if t < len(meta['tokens'])])

        print(f"Expected: '{sample['word']}'")
        print(f"Predicted: '{predicted}'")

        if predicted == sample['word']:
            print("✓ Correct!")
            correct += 1
        else:
            print("✗ Incorrect")

    print(f"\n=== Beam Search Results ===")
    print(f"Accuracy: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")

if __name__ == "__main__":
    test_beam_search()