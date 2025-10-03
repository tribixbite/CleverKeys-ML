#!/usr/bin/env python3
"""
Inspect ONNX models to understand their actual input/output dimensions
"""

import onnxruntime as ort
import json


def inspect_model(model_path):
    """Inspect an ONNX model's inputs and outputs"""
    session = ort.InferenceSession(model_path)

    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    print("\nINPUTS:")
    for input_info in session.get_inputs():
        print(f"  {input_info.name}:")
        print(f"    - Shape: {input_info.shape}")
        print(f"    - Type: {input_info.type}")

    print("\nOUTPUTS:")
    for output_info in session.get_outputs():
        print(f"  {output_info.name}:")
        print(f"    - Shape: {output_info.shape}")
        print(f"    - Type: {output_info.type}")

    return session


def test_decoder_with_different_configs(decoder_session, encoder_dim):
    """Try different state configurations to see what works"""
    import numpy as np

    # Create dummy encoder output
    encoder_frame = np.random.randn(1, encoder_dim, 1).astype(np.float32)
    targets = np.array([[0]], dtype=np.int32)
    target_length = np.array([1], dtype=np.int32)

    # Try different state configurations
    configs = [
        (1, 192),  # Mobile preset from docs
        (1, 320),  # Common LSTM hidden size
        (2, 192),  # 2 layers, smaller hidden
        (2, 320),  # 2 layers, larger hidden
        (1, 256),  # Another common size
        (2, 256),  # 2 layers of 256
    ]

    print("\n" + "="*60)
    print("Testing different decoder state configurations:")
    print("="*60)

    for num_layers, hidden_size in configs:
        try:
            state_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
            state_c = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

            outputs = decoder_session.run(
                None,
                {
                    'encoder_outputs': encoder_frame,
                    'targets': targets,
                    'target_length': target_length,
                    'input_states_1': state_h,
                    'input_states_2': state_c
                }
            )

            print(f"\n✅ SUCCESS with layers={num_layers}, hidden={hidden_size}")
            print(f"   Output shapes:")
            for i, output in enumerate(outputs):
                print(f"     Output {i}: {output.shape}")

            # This is the working configuration
            return num_layers, hidden_size

        except Exception as e:
            print(f"❌ Failed with layers={num_layers}, hidden={hidden_size}: {str(e)[:100]}")

    return None, None


def main():
    print("ONNX MODEL INSPECTION")
    print("="*60)

    # Inspect auto_best models
    print("\n### AUTO_BEST MODELS ###")
    encoder_session = inspect_model('../models/auto_best/encoder.onnx')
    decoder_session = inspect_model('../models/auto_best/decoder_joint.onnx')

    # Load runtime meta
    with open('../models/auto_best/runtime_meta.json', 'r') as f:
        meta = json.load(f)

    print("\n" + "="*60)
    print("RUNTIME METADATA:")
    print("="*60)
    print(f"vocab_size: {meta.get('vocab_size')}")
    print(f"blank_id: {meta.get('blank_id')}")
    if 'decoder_config' in meta:
        print(f"decoder_config: {meta['decoder_config']}")
    else:
        print("decoder_config: NOT FOUND")

    # Test decoder with different configs
    encoder_dim = 144  # From encoder output shape
    num_layers, hidden_size = test_decoder_with_different_configs(decoder_session, encoder_dim)

    if num_layers and hidden_size:
        print("\n" + "="*60)
        print("DISCOVERED CONFIGURATION:")
        print("="*60)
        print(f"✅ Decoder LSTM configuration:")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Encoder dimension: {encoder_dim}")

        # Create updated runtime_meta
        updated_meta = meta.copy()
        updated_meta['decoder_config'] = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'encoder_dim': encoder_dim
        }

        # Save it
        output_path = '../models/auto_best/runtime_meta_fixed.json'
        with open(output_path, 'w') as f:
            json.dump(updated_meta, f, indent=2)

        print(f"\n✅ Saved fixed runtime_meta to: {output_path}")

    # Also check rnnt_new_latest if it exists
    print("\n\n### RNNT_NEW_LATEST MODELS ###")
    try:
        encoder_session2 = inspect_model('../models/rnnt_new_latest/encoder.onnx')
        decoder_session2 = inspect_model('../models/rnnt_new_latest/decoder_joint.onnx')

        # Test this one too
        encoder_dim2 = 256  # Likely from encoder output
        num_layers2, hidden_size2 = test_decoder_with_different_configs(decoder_session2, encoder_dim2)

        if num_layers2 and hidden_size2:
            print(f"\n✅ rnnt_new_latest decoder config: layers={num_layers2}, hidden={hidden_size2}")
    except Exception as e:
        print(f"Could not inspect rnnt_new_latest: {e}")


if __name__ == '__main__':
    main()