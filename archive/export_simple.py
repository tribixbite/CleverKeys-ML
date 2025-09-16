#!/usr/bin/env python3
"""
Simple export of trained model to ONNX format
"""

import torch
from pathlib import Path
from train_production import ImprovedTransformerModel, RobustFeaturizer, CONFIG

def main():
    print("="*50)
    print("Model Export")
    print("="*50)
    
    # Load checkpoint
    checkpoint_path = "checkpoints_production/best_model.pth"
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', CONFIG)
    
    # Initialize model
    featurizer = RobustFeaturizer()
    input_dim = 15 + featurizer.num_keys
    model = ImprovedTransformerModel(input_dim, config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # Save as regular PyTorch model
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    # Save model for PyTorch usage
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_dim': input_dim,
        'vocab_size': config.get('vocab_size', 28)
    }, export_dir / "swipe_model_final.pth")
    
    print(f"✓ PyTorch model saved to exports/swipe_model_final.pth")
    
    # Export vocabulary
    import json
    vocab = {
        'chars': list('abcdefghijklmnopqrstuvwxyz '),
        'input_dim': input_dim,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    with open(export_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    print(f"✓ Vocabulary saved to exports/vocab.json")
    print("\nExport complete!")
    print(f"Total model size: {(export_dir / 'swipe_model_final.pth').stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()