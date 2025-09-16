#!/usr/bin/env python3
# export_pte_ultra.py
# Ultra-optimized quantized ExecuTorch .pte encoder for blazing fast Android performance

import argparse
import logging
import torch
from model_class import GestureRNNTModel, get_default_config

# ExecuTorch PT2E quantization
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("export_pte_ultra")

class OptimizedEncoderWrapper(torch.nn.Module):
    """Ultra-optimized encoder wrapper for maximum Android performance"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length):
        # Direct forward without extra processing for maximum speed
        return self.encoder(audio_signal=audio_signal, length=length)

def create_calibration_data(model, num_samples=32):
    """Create calibration data for quantization"""
    calibration_data = []

    # Generate diverse calibration samples
    for i in range(num_samples):
        # Vary sequence lengths for robustness
        T = torch.randint(50, 200, (1,)).item()
        B, F = 1, 37

        # Create realistic gesture-like patterns
        audio_signal = torch.randn(B, F, T) * 0.5 + 0.1
        length = torch.tensor([T], dtype=torch.int32)

        calibration_data.append((audio_signal, length))

    return calibration_data

def main():
    parser = argparse.ArgumentParser(description="Export ultra-optimized quantized PTE encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--output", default="encoder_ultra_quant.pte", help="Output PTE file")
    parser.add_argument("--max_length", type=int, default=200, help="Max sequence length for export")
    args = parser.parse_args()

    log.info(f"Loading checkpoint: {args.checkpoint}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if 'hyper_parameters' in ckpt:
        cfg = ckpt['hyper_parameters']['cfg']
    else:
        cfg = get_default_config()

    model = GestureRNNTModel(cfg).eval()

    # Clean torch.compile keys
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder._orig_mod."):
            new_key = key.replace("encoder._orig_mod.", "encoder.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)

    # Create optimized wrapper
    encoder_wrapper = OptimizedEncoderWrapper(model.encoder).eval()

    log.info("Creating calibration data...")
    calibration_data = create_calibration_data(model, num_samples=16)

    log.info("Exporting to torch.export format...")
    # Use first calibration sample for export
    example_input = calibration_data[0]
    exported_program = torch.export.export(encoder_wrapper, example_input)

    log.info("Setting up XNNPACK quantizer...")
    quantizer = XNNPACKQuantizer()
    # Ultra-aggressive quantization for maximum performance
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

    log.info("Preparing quantization...")
    prepared_program = prepare_pt2e(exported_program, quantizer)

    log.info("Calibrating with representative data...")
    # Run calibration
    with torch.no_grad():
        for i, (audio_signal, length) in enumerate(calibration_data[:8]):  # Use subset for speed
            try:
                _ = prepared_program(audio_signal, length)
                if (i + 1) % 4 == 0:
                    log.info(f"  Calibrated {i+1}/8 samples")
            except Exception as e:
                log.warning(f"Calibration sample {i} failed: {e}")
                continue

    log.info("Converting to quantized model...")
    quantized_program = convert_pt2e(prepared_program)

    log.info("Lowering to ExecuTorch...")
    edge = to_edge(quantized_program)

    # Ultra-optimized XNNPACK partitioning
    edge = edge.to_backend(XnnpackPartitioner())

    # Generate final ExecuTorch program
    executorch_program = edge.to_executorch()

    # Save to file
    with open(args.output, "wb") as f:
        f.write(executorch_program.buffer)

    log.info(f"✓ Ultra-optimized quantized PTE saved: {args.output}")

    # Print optimization summary
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    log.info(f"Final model size: {size_mb:.1f} MB")
    log.info("Optimizations applied:")
    log.info("  - INT8 symmetric per-channel quantization")
    log.info("  - XNNPACK backend partitioning")
    log.info("  - Optimized for Android ARM/NEON")

if __name__ == "__main__":
    main()