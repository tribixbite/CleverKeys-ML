#!/usr/bin/env python3
"""Check model parameters and training status"""

import sys
sys.path.append('../trained_models/nema1')

import torch
from export_common import load_trained_model

# Load the model
print("Loading checkpoint...")
model = load_trained_model("../trained_models/nema1/last.ckpt")

# Check model configuration
print("\nModel configuration:")
if hasattr(model, 'cfg'):
    cfg = model.cfg
    print(f"  Vocabulary size: {cfg.labels if hasattr(cfg, 'labels') else 'N/A'}")
    print(f"  Blank as pad: {cfg.blank_as_pad if hasattr(cfg, 'blank_as_pad') else 'N/A'}")

    if hasattr(cfg, 'encoder'):
        print(f"\nEncoder config:")
        print(f"  Type: {cfg.encoder.get('_target_', 'N/A')}")
        print(f"  Feature dim: {cfg.encoder.get('feat_in', 'N/A')}")
        print(f"  Hidden dim: {cfg.encoder.get('d_model', 'N/A')}")

    if hasattr(cfg, 'decoder'):
        print(f"\nDecoder config:")
        print(f"  Type: {cfg.decoder.get('_target_', 'N/A')}")
        print(f"  Hidden dim: {cfg.decoder.get('pred_hidden', 'N/A')}")
        print(f"  Num layers: {cfg.decoder.get('pred_rnn_layers', 'N/A')}")

# Check vocabulary
if hasattr(model, 'joint') and hasattr(model.joint, 'vocabulary'):
    vocab = model.joint.vocabulary
    print(f"\nVocabulary ({len(vocab)} tokens):")
    for i, token in enumerate(list(vocab)[:35]):
        print(f"  {i}: '{token}'")

# Check blank index
if hasattr(model, 'decoder') and hasattr(model.decoder, 'blank_idx'):
    print(f"\nBlank index: {model.decoder.blank_idx}")

# Check if model is in training or eval mode
print(f"\nModel training: {model.training}")

# Check a forward pass with dummy data
print("\nTesting forward pass...")
try:
    batch_size = 1
    seq_len = 56
    feat_dim = 37

    # Dummy input
    audio_signal = torch.randn(batch_size, feat_dim, seq_len)
    audio_len = torch.tensor([seq_len])

    # Run encoder
    with torch.no_grad():
        enc_out, enc_len = model.encoder(audio_signal=audio_signal, length=audio_len)

    print(f"  Encoder input: {audio_signal.shape}")
    print(f"  Encoder output: {enc_out.shape}")
    print(f"  Output length: {enc_len.item()}")

    # Check decoder
    blank_idx = model.decoder.blank_idx if hasattr(model.decoder, 'blank_idx') else 0
    y_prev = torch.tensor([blank_idx])

    # Get initial states
    batch = enc_out.shape[0]
    states = model.decoder.initialize_state(enc_out)

    # Single decode step
    with torch.no_grad():
        # Extract one frame from encoder output
        enc_frame = enc_out[:, :, 0:1]  # (B, D, 1)

        # Run decoder prediction
        pred_out, new_states = model.decoder.predict(
            y_prev,
            state=states,
            add_sos=False,
            batch_size=batch
        )

        # Run joint
        joint_out = model.joint(
            encoder_outputs=enc_frame,
            decoder_outputs=pred_out.unsqueeze(2)  # Add time dimension
        )

    print(f"\n  Decoder input (y_prev): {y_prev.shape}")
    print(f"  Decoder output: {pred_out.shape}")
    print(f"  Joint output: {joint_out.shape}")
    print(f"  Joint output values (first 5): {joint_out[0, :5, 0].tolist()}")

    # Get probabilities
    probs = torch.softmax(joint_out[0, :, 0], dim=0)
    top5_probs, top5_idx = torch.topk(probs, 5)

    print(f"\n  Top 5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_probs)):
        token = list(vocab)[idx] if idx < len(vocab) else f"OOV_{idx}"
        print(f"    {i+1}. Token {idx} ('{token}'): {prob:.4f}")

except Exception as e:
    print(f"  Error in forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")