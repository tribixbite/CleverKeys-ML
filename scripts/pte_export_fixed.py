#!/usr/bin/env python3
"""
Export trained model to PTE (PyTorch Edge) format for Android deployment
"""

import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from train_production import ImprovedTransformerModel, RobustFeaturizer, CONFIG

def export_to_pte(checkpoint_path="checkpoints_production/best_model.pth"):
    """Export model to PTE format for Android"""
    
    print("Loading checkpoint for PTE export...")
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create export directory
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    # Export to TorchScript first
    print("Converting to TorchScript...")
    batch_size = 1
    seq_length = 100
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    dummy_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    
    # Trace the model
    traced_model = torch.jit.trace(model, (dummy_input, dummy_mask))
    
    # Save TorchScript model
    torchscript_path = export_dir / "swipe_model.pt"
    traced_model.save(str(torchscript_path))
    print(f"TorchScript model saved to {torchscript_path}")
    
    # For actual PTE export, you would use ExecuTorch toolchain
    # This requires the executorch package which needs separate installation
    print("\nTo complete PTE export for Android:")
    print("1. Install ExecuTorch: pip install executorch")
    print("2. Run: python -m executorch.export swipe_model.pt --output swipe_model.pte")
    print("\nAlternatively, use the TorchScript model directly in Android with PyTorch Mobile")
    
    # Save model info for Android integration
    model_info = {
        "input_dim": input_dim,
        "vocab_size": config["vocab_size"],
        "max_sequence_length": 100,
        "model_params": sum(p.numel() for p in model.parameters()),
        "config": config
    }
    
    import json
    with open(export_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel info saved to {export_dir}/model_info.json")
    print(f"Model size: {torchscript_path.stat().st_size / 1e6:.1f} MB")
    
    return torchscript_path

if __name__ == "__main__":
    export_to_pte()