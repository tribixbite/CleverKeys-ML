#!/usr/bin/env python3
"""Create optimized FP32 ExecuTorch encoder for Android."""

import argparse
import logging
import os

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge

from export_common import load_trained_model, make_example_inputs, package_artifacts

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("export_pte_optimized")


class AndroidOptimizedEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        return self.encoder(audio_signal=audio_signal, length=length)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export optimized (non-quant) ExecuTorch encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt or .nemo model")
    parser.add_argument("--output", default="encoder_android_optimized.pte", help="Output ExecuTorch file")
    parser.add_argument("--max-trace-len", type=int, default=200, help="Max gesture length for export inputs")
    parser.add_argument("--lexicon", help="Optional lexicon to include when packaging")
    parser.add_argument("--package-dir", help="Copy exported file (and assets) into this directory")
    parser.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional assets to include when packaging",
    )
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    encoder_wrapper = AndroidOptimizedEncoderWrapper(model.encoder)

    audio_signal, length = make_example_inputs(args.max_trace_len)

    log.info("Exporting encoder to ExecuTorch")
    exported_program = torch.export.export(encoder_wrapper, (audio_signal, length))
    edge_program = to_edge(exported_program)
    edge_program = edge_program.to_backend(XnnpackPartitioner())
    executorch_program = edge_program.to_executorch()

    buffer = getattr(executorch_program, "buffer", None)
    if buffer is None and hasattr(executorch_program, "to_buffer"):
        buffer = executorch_program.to_buffer()
    if buffer is None:
        raise RuntimeError("Unable to extract ExecuTorch buffer")

    with open(args.output, "wb") as handle:
        handle.write(buffer)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    log.info(f"✓ Android-optimized ExecuTorch saved: {args.output} ({size_mb:.1f} MB)")
    log.info("Android optimizations applied:")
    log.info("  ✓ XNNPACK backend for ARM acceleration")
    log.info("  ✓ Graph optimizations for mobile")
    log.info("  ✓ Memory layout optimizations")
    log.info("  ✓ Operator fusion for efficiency")

    if args.package_dir:
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts([args.output], args.package_dir, extra_files=extras)


if __name__ == "__main__":
    main()
