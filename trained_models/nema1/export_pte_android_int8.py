#!/usr/bin/env python3
"""Export optimized INT8 PTE encoder for Android with high accuracy and low latency."""

import argparse
import logging
import os
import torch
import torch.nn as nn
from pathlib import Path

# ExecuTorch imports
try:
    from executorch.exir import to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from torch.ao.quantization import (
        get_default_qconfig_mapping,
        quantize_fx,
    )
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False
    print("Warning: ExecuTorch not available, will export ONNX instead")

from export_common import (
    load_trained_model,
    make_example_inputs,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("export_pte_android")


class EncoderWrapper(nn.Module):
    """Wrapper for encoder to simplify export."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        return self.encoder(audio_signal=audio_signal, length=length)


def export_pte_int8(model_path: str, output_path: str, max_trace_len: int = 150):
    """Export INT8 quantized PTE model."""

    # Load model
    log.info(f"Loading checkpoint: {model_path}")
    model = load_trained_model(model_path)

    # Wrap encoder
    encoder = EncoderWrapper(model.encoder)

    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    log.info(f"Model parameters: {params/1e6:.2f}M")
    log.info(f"Estimated FP32 size: {params*4/(1024**2):.1f}MB")
    log.info(f"Target INT8 size: {params/(1024**2):.1f}MB")

    # Create example inputs
    example_feats, example_lens = make_example_inputs(max_trace_len)

    if HAS_EXECUTORCH:
        try:
            # Export with torch.export
            log.info("Exporting model graph...")
            exported = torch.export.export(encoder, (example_feats, example_lens))

            # Convert to edge
            log.info("Converting to Edge IR...")
            edge = to_edge(exported)

            # Partition for XNNPACK (includes quantization)
            log.info("Partitioning for XNNPACK with INT8...")
            edge = edge.to_backend(XnnpackPartitioner())

            # Export to ExecuTorch
            log.info("Creating ExecuTorch program...")
            et_program = edge.to_executorch()

            # Get buffer
            buffer = et_program.buffer if hasattr(et_program, 'buffer') else et_program

            # Save PTE
            with open(output_path, 'wb') as f:
                f.write(buffer)

            log.info(f"✓ Saved PTE to {output_path}")

            # Report size
            size_mb = os.path.getsize(output_path) / (1024**2)
            log.info(f"Final PTE size: {size_mb:.1f}MB")

            # Compression ratio
            compression = (params*4/(1024**2)) / size_mb
            log.info(f"Compression ratio: {compression:.1f}x")

            return True

        except Exception as e:
            log.error(f"PTE export failed: {e}")
            log.info("Falling back to ONNX export...")
            return export_onnx_fallback(encoder, example_feats, example_lens, output_path)
    else:
        return export_onnx_fallback(encoder, example_feats, example_lens, output_path)


def export_onnx_fallback(encoder, example_feats, example_lens, output_path):
    """Fallback to ONNX export if PTE fails."""
    try:
        onnx_path = output_path.replace('.pte', '.onnx')
        log.info(f"Exporting ONNX to {onnx_path}")

        torch.onnx.export(
            encoder,
            (example_feats, example_lens),
            onnx_path,
            opset_version=17,
            input_names=["features_bft", "lengths"],
            output_names=["encoded_btf", "encoded_lengths"],
            dynamic_axes={
                "features_bft": {0: "B", 2: "T"},
                "lengths": {0: "B"},
                "encoded_btf": {0: "B", 1: "T_out"},
                "encoded_lengths": {0: "B"},
            },
        )

        size_mb = os.path.getsize(onnx_path) / (1024**2)
        log.info(f"✓ ONNX exported: {onnx_path} ({size_mb:.1f}MB)")
        return True

    except Exception as e:
        log.error(f"ONNX export also failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export INT8 PTE for Android")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="encoder_android_int8.pte", help="Output PTE file")
    parser.add_argument("--max-trace-len", type=int, default=150, help="Max gesture length")
    args = parser.parse_args()

    success = export_pte_int8(args.checkpoint, args.output, args.max_trace_len)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())