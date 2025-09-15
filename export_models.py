#!/usr/bin/env python3
"""
Export trained model to ONNX and prepare for PTE conversion
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from train_production import ImprovedTransformerModel, RobustFeaturizer, CONFIG

def export_to_onnx(checkpoint_path="checkpoints_production/best_model.pth"):
    """Export model to ONNX format with quantization"""
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', CONFIG)
    
    # Initialize model
    featurizer = RobustFeaturizer()
    input_dim = 15 + featurizer.num_keys
    model = ImprovedTransformerModel(input_dim, config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dummy input
    batch_size = 1
    seq_length = 100
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    dummy_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    
    # Export to ONNX
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    onnx_path = export_dir / "swipe_model.onnx"
    
    print(f"Exporting to {onnx_path}...")
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        onnx_path,
        input_names=['features', 'mask'],
        output_names=['log_probs'],
        dynamic_axes={
            'features': {0: 'batch', 1: 'sequence'},
            'mask': {0: 'batch', 1: 'sequence'},
            'log_probs': {0: 'sequence', 1: 'batch'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"ONNX model exported to {onnx_path}")
    
    # Dynamic quantization for smaller model
    print("Applying dynamic quantization...")
    quantized_path = export_dir / "swipe_model_quantized.onnx"
    
    # Use onnxruntime for quantization
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized model saved to {quantized_path}")
        
        # Verify quantized model
        session = ort.InferenceSession(str(quantized_path))
        outputs = session.run(
            None,
            {
                'features': dummy_input.numpy(),
                'mask': dummy_mask.numpy()
            }
        )
        print(f"Quantized model verified. Output shape: {outputs[0].shape}")
        
    except ImportError:
        print("Install onnxruntime-tools for quantization: pip install onnxruntime onnxruntime-tools")
    
    return onnx_path, quantized_path

def export_to_torchscript(checkpoint_path="checkpoints_production/best_model.pth"):
    """Export model to TorchScript for PTE conversion"""
    
    print("Loading checkpoint for TorchScript...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', CONFIG)
    
    # Initialize model
    featurizer = RobustFeaturizer()
    input_dim = 15 + featurizer.num_keys
    model = ImprovedTransformerModel(input_dim, config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Trace the model
    dummy_input = torch.randn(1, 100, input_dim)
    dummy_mask = torch.zeros(1, 100, dtype=torch.bool)
    
    print("Tracing model...")
    traced_model = torch.jit.trace(model, (dummy_input, dummy_mask))
    
    # Save TorchScript model
    export_dir = Path("exports")
    torchscript_path = export_dir / "swipe_model.pt"
    traced_model.save(str(torchscript_path))
    
    print(f"TorchScript model saved to {torchscript_path}")
    
    # Create PTE conversion script
    pte_script = """#!/bin/bash
# Convert TorchScript to PTE format
# Requires ExecuTorch SDK

echo "Converting to PTE format..."
python -m executorch.exir.capture \\
    --model_path exports/swipe_model.pt \\
    --output_path exports/swipe_model.pte

echo "PTE conversion complete!"
"""
    
    pte_script_path = export_dir / "convert_to_pte.sh"
    pte_script_path.write_text(pte_script)
    pte_script_path.chmod(0o755)
    
    print(f"PTE conversion script saved to {pte_script_path}")
    
    return torchscript_path

def export_vocabulary_and_config():
    """Export vocabulary and configuration for web deployment"""
    
    export_dir = Path("exports")
    
    # Export character mapping
    char_to_idx = {char: idx + 1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz'")}
    char_to_idx['<blank>'] = 0
    
    vocab_data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {v: k for k, v in char_to_idx.items()},
        "blank_idx": 0
    }
    
    vocab_path = export_dir / "vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"Vocabulary exported to {vocab_path}")
    
    # Export keyboard layout
    featurizer = RobustFeaturizer()
    keyboard_data = {
        "layout": {k: list(v) for k, v in featurizer.keyboard_layout.items()},
        "num_keys": featurizer.num_keys
    }
    
    keyboard_path = export_dir / "keyboard_layout.json"
    with open(keyboard_path, 'w') as f:
        json.dump(keyboard_data, f, indent=2)
    
    print(f"Keyboard layout exported to {keyboard_path}")
    
    # Export model config
    model_config = {
        "input_dim": 15 + featurizer.num_keys,
        "d_model": CONFIG["d_model"],
        "vocab_size": 28,
        "max_seq_length": CONFIG["max_seq_length"]
    }
    
    config_path = export_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model config exported to {config_path}")

def main():
    """Export all model formats"""
    
    print("="*50)
    print("Model Export Pipeline")
    print("="*50)
    
    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints_production/best_model.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Using latest checkpoint instead...")
        checkpoints = list(Path("checkpoints_production").glob("*.pth"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found. Train the model first!")
            return
    
    # Export to ONNX
    onnx_path, quantized_path = export_to_onnx(str(checkpoint_path))
    
    # Export to TorchScript
    torchscript_path = export_to_torchscript(str(checkpoint_path))
    
    # Export vocabulary and config
    export_vocabulary_and_config()
    
    print("\n" + "="*50)
    print("Export Complete!")
    print("="*50)
    print(f"ONNX Model: {onnx_path}")
    print(f"Quantized ONNX: {quantized_path}")
    print(f"TorchScript: {torchscript_path}")
    print("\nFor PTE conversion, run: ./exports/convert_to_pte.sh")

if __name__ == "__main__":
    main()