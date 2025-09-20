#!/usr/bin/env python3
"""
Advanced export script for NeMo RNN-T models.

Handles exporting to ONNX and PyTorch ExecuTorch (.pte) formats, with
support for INT8 dynamic quantization. It also generates the necessary
runtime_meta.json file for web/mobile deployments.

Usage:
    # Export to FP32 ONNX
    uv run python new/export_advanced.py --checkpoint <path.ckpt> --output_dir new/ --format onnx

    # Export to INT8 Quantized ONNX
    uv run python new/export_advanced.py --checkpoint <path.ckpt> --output_dir new/ --format onnx --quantize int8

    # Export to INT8 Quantized PTE for ExecuTorch
    uv run python new/export_advanced.py --checkpoint <path.ckpt> --output_dir new/ --format pte --quantize int8
"""

import argparse
import json
from pathlib import Path
import sys
import torch
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).parent.absolute()))

from train_transducer_personalized import PersonalizedRNNTModel, CONFIG, build_model_config, load_vocab

def export_onnx(model, output_dir, quantize_int8=False):
    """Exports the model to ONNX, with optional INT8 quantization."""
    encoder_path = output_dir / "encoder.onnx"
    decoder_path = output_dir / "decoder_joint.onnx"

    print(f"Exporting ONNX model to {output_dir}...")
    model.export(str(output_dir / "model.onnx")) # NeMo creates multiple files

    # Rename for clarity
    if (output_dir / "encoder-model.onnx").exists():
        (output_dir / "encoder-model.onnx").rename(encoder_path)
    if (output_dir / "decoder_joint-model.onnx").exists():
        (output_dir / "decoder_joint-model.onnx").rename(decoder_path)

    if quantize_int8:
        print("Performing INT8 dynamic quantization...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(encoder_path, encoder_path.with_suffix('.int8.onnx'), weight_type=QuantType.QInt8)
        quantize_dynamic(decoder_path, decoder_path.with_suffix('.int8.onnx'), weight_type=QuantType.QInt8)
        print("INT8 quantization complete.")

def export_pte(model, output_dir, quantize_int8=False):
    """Exports the model to PyTorch ExecuTorch (.pte) format."""
    print("Scripting model for PTE export...")
    # ExecuTorch requires a JIT-scripted model
    scripted_encoder = torch.jit.script(model.encoder)
    scripted_decoder = torch.jit.script(model.decoder)
    scripted_joint = torch.jit.script(model.joint)

    if quantize_int8:
        print("Performing INT8 dynamic quantization for PTE...")
        # For PTE, quantization is done on the scripted model before export
        scripted_encoder = torch.quantization.quantize_dynamic(scripted_encoder, {torch.nn.Linear}, dtype=torch.qint8)
        scripted_decoder = torch.quantization.quantize_dynamic(scripted_decoder, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
        scripted_joint = torch.quantization.quantize_dynamic(scripted_joint, {torch.nn.Linear}, dtype=torch.qint8)

    print("Exporting to .pte files...")
    from torch.export import export
    from executorch.exir import to_edge

    # You need example inputs to trace the model for ExecuTorch
    # These shapes should match what the model expects
    enc_input = (torch.randn(1, 37, 96), torch.tensor([96]))
    dec_input = (torch.randint(0, 29, (1, 20)), torch.tensor([20]))

    # Convert to Edge IR and then to .pte
    encoder_edge = to_edge(export(scripted_encoder, enc_input))
    decoder_edge = to_edge(export(scripted_decoder, dec_input))

    with open(output_dir / "encoder.pte", "wb") as f:
        f.write(encoder_edge.get_buffer())
    with open(output_dir / "decoder.pte", "wb") as f:
        f.write(decoder_edge.get_buffer())
    print("PTE export complete.")

def generate_runtime_meta(model, vocab, output_dir):
    """Generates the runtime_meta.json file needed by frontends."""
    print(f"Generating runtime_meta.json in {output_dir}...")
    meta = {
        "vocab_size": len(vocab),
        "blank_id": model.decoder.blank_idx, # Critical for decoding
        "tokens": list(vocab.keys()),
        "char_to_id": vocab,
        "id_to_char": {v: k for k, v in vocab.items()},
    }
    with open(output_dir / "runtime_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    print("runtime_meta.json generated.")

def main():
    parser = argparse.ArgumentParser(description="Advanced NeMo Model Exporter")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .ckpt file")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save exported files")
    parser.add_argument("--format", required=True, choices=['onnx', 'pte'], help="Export format")
    parser.add_argument("--quantize", choices=['int8', 'none'], default='none', help="Quantization type")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Building model from configuration...")
    cfg = DictConfig(CONFIG)
    vocab = load_vocab(cfg.data.vocab_path)
    labels = list(vocab.keys())
    nemo_cfg = build_model_config(cfg, labels)
    model = PersonalizedRNNTModel(cfg=nemo_cfg)

    print(f"Loading weights from checkpoint: {args.checkpoint}")
    import omegaconf
    with torch.serialization.safe_globals([omegaconf.dictconfig.DictConfig]):
        model.load_state_dict(torch.load(args.checkpoint, weights_only=False)['state_dict'])
    model.eval()

    if args.format == 'onnx':
        export_onnx(model, output_dir, quantize_int8=(args.quantize == 'int8'))
    elif args.format == 'pte':
        export_pte(model, output_dir, quantize_int8=(args.quantize == 'int8'))

    generate_runtime_meta(model, vocab, output_dir)

if __name__ == "__main__":
    main()
