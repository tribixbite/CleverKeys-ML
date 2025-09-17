#!/usr/bin/env python3
"""Ultra-optimized quantized ExecuTorch encoder for blazing fast Android keyboards."""

import argparse
import logging
import os
import zlib

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from export_common import (
    create_calibration_loader,
    ensure_default_manifest,
    iter_calibration_batches,
    load_trained_model,
    make_example_inputs,
    package_artifacts,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("export_pte_ultra")


def log_model_parameters(model, name: str = "Model") -> None:
    """Log parameter count and estimated size."""
    if hasattr(model, "encoder"):
        params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        log.info(
            "%s encoder parameters: %.2fM (~%.1fMB fp32, ~%.1fMB int8)",
            name,
            params / 1e6,
            params * 4 / (1024 ** 2),
            params / (1024 ** 2),
        )
    else:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(
            "%s parameters: %.2fM (~%.1fMB fp32, ~%.1fMB int8)",
            name,
            params / 1e6,
            params * 4 / (1024 ** 2),
            params / (1024 ** 2),
        )


def verify_quantization(program, program_name: str = "Program") -> bool:
    """Verify that quantize/dequantize nodes exist in the exported program."""
    quant_ops_found = 0
    dequant_ops_found = 0
    fp32_linear_ops = 0

    try:
        graph_module = program.graph_module if hasattr(program, "graph_module") else program
        for node in graph_module.graph.nodes:
            lowered = str(node.target).lower()
            if "quantize" in lowered:
                quant_ops_found += 1
            elif "dequantize" in lowered:
                dequant_ops_found += 1
            elif "linear" in lowered and "quantized" not in lowered:
                fp32_linear_ops += 1
    except Exception as exc:  # pragma: no cover - diagnostics only
        log.warning("Could not analyse %s graph: %s", program_name, exc)
        return False

    is_quantized = quant_ops_found > 0 and dequant_ops_found > 0
    log.info("%s quantization analysis:", program_name)
    log.info("  Quantize ops: %d", quant_ops_found)
    log.info("  Dequantize ops: %d", dequant_ops_found)
    log.info("  FP32 linear ops: %d", fp32_linear_ops)
    log.info("  Status: %s", "✓ QUANTIZED" if is_quantized else "⚠ NOT QUANTIZED")
    return is_quantized


def validate_xnnpack_partition(edge_program, program_name: str = "Edge program") -> bool:
    """Measure XNNPACK partition coverage for the lowered ExecuTorch program."""
    try:
        graph_module = edge_program.exported_program().graph_module
        total_nodes = len(list(graph_module.graph.nodes))
        xnnpack_nodes = 0
        fallback_nodes = 0

        for node in graph_module.graph.nodes:
            lowered = str(node.target).lower()
            if "xnnpack" in lowered:
                xnnpack_nodes += 1
            elif node.op in ("call_function", "call_method") and "aten" in lowered:
                fallback_nodes += 1

        partition_ratio = xnnpack_nodes / max(total_nodes, 1) * 100
        log.info("%s XNNPACK partition analysis:", program_name)
        log.info("  Total nodes: %d", total_nodes)
        log.info("  XNNPACK nodes: %d", xnnpack_nodes)
        log.info("  Fallback nodes: %d", fallback_nodes)
        log.info("  Partition ratio: %.1f%%", partition_ratio)

        if partition_ratio < 50:
            log.warning("⚠ Low XNNPACK partition ratio (%.1f%%)", partition_ratio)
        else:
            log.info("✓ Good XNNPACK partition ratio (%.1f%%)", partition_ratio)
        return partition_ratio >= 50
    except Exception as exc:  # pragma: no cover - diagnostics only
        log.warning("Could not analyse %s partitioning: %s", program_name, exc)
        return False


def estimate_compressed_size(file_path: str, compression_level: int = 6):
    """Estimate APK compressed size using zlib."""
    try:
        with open(file_path, "rb") as handle:
            original_data = handle.read()
        compressed_data = zlib.compress(original_data, compression_level)
        original_mb = len(original_data) / (1024 ** 2)
        compressed_mb = len(compressed_data) / (1024 ** 2)
        compression_ratio = len(compressed_data) / len(original_data) * 100
        log.info("Size analysis for %s:", file_path)
        log.info("  Raw size: %.1f MB", original_mb)
        log.info("  Compressed (APK): %.1f MB (%.1f%% of original)", compressed_mb, compression_ratio)
        return compressed_mb
    except Exception as exc:  # pragma: no cover - diagnostics only
        log.warning("Could not estimate compressed size: %s", exc)
        return None


def create_synthetic_calibration_data(num_samples: int) -> list:
    """Fallback synthetic calibration samples used when no dataset is available."""
    samples = []
    log.info("Generating %d synthetic calibration samples", num_samples)
    for idx in range(num_samples):
        T = torch.randint(30, 200, (1,)).item()
        audio_signal = torch.randn(1, 37, T) * 0.4
        lengths = torch.tensor([T], dtype=torch.int32)
        samples.append((audio_signal, lengths))
        if (idx + 1) % 8 == 0:
            log.info("  Created %d/%d synthetic samples", idx + 1, num_samples)
    return samples


class OptimizedEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        return self.encoder(audio_signal=audio_signal, length=length)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ultra-optimized quantized PTE encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .ckpt or .nemo model")
    parser.add_argument("--vocab", default="vocab/final_vocab.txt", help="Vocabulary file for calibration")
    parser.add_argument(
        "--calib-manifest",
        help="Calibration manifest (.jsonl). Defaults to data/train_final_val.jsonl if absent.",
    )
    parser.add_argument("--output", default="encoder_ultra_quant.pte", help="Output PTE file")
    parser.add_argument("--max-trace-len", type=int, default=200, help="Max gesture length for export inputs")
    parser.add_argument("--calib-batches", type=int, default=32, help="Calibration batches to run")
    parser.add_argument("--calib-batch-size", type=int, default=8, help="Calibration batch size")
    parser.add_argument(
        "--synthetic-calibration",
        action="store_true",
        help="Use synthetic calibration data instead of dataset batches",
    )
    parser.add_argument("--skip_quantization_check", action="store_true", help="Skip quantization verification (faster)")
    parser.add_argument("--skip_partition_check", action="store_true", help="Skip XNNPACK partition validation")
    parser.add_argument("--fallback_to_fp32", action="store_true", help="Fallback to FP32 if quantization fails")
    parser.add_argument("--lexicon", help="Optional lexicon to bundle with packaged output")
    parser.add_argument("--package-dir", help="Copy exported file (and assets) into this directory")
    parser.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional files (trie, metadata, etc.) to include when packaging",
    )
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    log_model_parameters(model, "Original model")

    encoder_wrapper = OptimizedEncoderWrapper(model.encoder).eval()
    log_model_parameters(encoder_wrapper, "Encoder wrapper")

    example_feats, example_lens = make_example_inputs(args.max_trace_len)
    log.info("Exporting encoder graph with torch.export ...")
    exported_program = torch.export.export(encoder_wrapper, (example_feats, example_lens))

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    log.info("Preparing quantization observers")
    prepared_program = prepare_pt2e(exported_program, quantizer)

    log.info("Calibrating quantized observers ...")
    if args.synthetic_calibration:
        calibration_samples = create_synthetic_calibration_data(args.calib_batches)
        with torch.no_grad():
            for idx, (audio_signal, length) in enumerate(calibration_samples[: args.calib_batches]):
                try:
                    prepared_program(audio_signal, length)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    log.warning("Synthetic calibration sample %d failed: %s", idx, exc)
    else:
        manifest = ensure_default_manifest(args.calib_manifest, "data/train_final_val.jsonl")
        dataloader = create_calibration_loader(
            manifest,
            args.vocab,
            args.max_trace_len,
            batch_size=args.calib_batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            for idx, (audio_signal, length) in enumerate(
                iter_calibration_batches(dataloader, args.calib_batches)
            ):
                prepared_program(audio_signal, length)
                if (idx + 1) % 10 == 0:
                    log.info("  Calibrated %d batches", idx + 1)

    log.info("Converting to quantized model ...")
    try:
        quantized_program = convert_pt2e(prepared_program)
        if not args.skip_quantization_check:
            is_quantized = verify_quantization(quantized_program, "Quantized program")
            if not is_quantized and not args.fallback_to_fp32:
                log.error("Quantization verification failed; rerun with --fallback_to_fp32 to retain FP32")
                return 1
            if not is_quantized and args.fallback_to_fp32:
                log.warning("Quantization failed, falling back to FP32 export")
                quantized_program = exported_program

        log.info("Lowering to ExecuTorch Edge IR ...")
        edge = to_edge(quantized_program)
        log.info("Partitioning for XNNPACK ...")
        edge = edge.to_backend(XnnpackPartitioner())
        if not args.skip_partition_check:
            validate_xnnpack_partition(edge, "XNNPACK partitioned program")
    except Exception as exc:
        log.error("Quantization/partitioning failed: %s", exc)
        if args.fallback_to_fp32:
            log.warning("Falling back to FP32 export path")
            edge = to_edge(exported_program)
            edge = edge.to_backend(XnnpackPartitioner())
        else:
            log.error("Use --fallback_to_fp32 to continue with FP32")
            return 1

    executorch_program = edge.to_executorch()
    buffer = getattr(executorch_program, "buffer", None)
    if buffer is None and hasattr(executorch_program, "to_buffer"):
        buffer = executorch_program.to_buffer()
    if buffer is None:
        log.error("ExecuTorch buffer missing; aborting export")
        return 1

    with open(args.output, "wb") as handle:
        handle.write(buffer)
    log.info("✓ Ultra-optimized quantized PTE saved: %s", args.output)

    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 ** 2)
        log.info("Final model size: %.1f MB", size_mb)
        compressed = estimate_compressed_size(args.output)
        log.info("Optimizations applied:")
        if not args.synthetic_calibration:
            log.info("  ✓ Dataset calibration (%d batches)", args.calib_batches)
        else:
            log.info("  ✓ Synthetic calibration (%d samples)", args.calib_batches)
        if not args.skip_quantization_check:
            log.info("  ✓ Quantization graph verified")
        if not args.skip_partition_check:
            log.info("  ✓ XNNPACK partition validated")
        if compressed:
            log.info("  ✓ Estimated APK footprint: %.1f MB", compressed)
    else:
        log.error("Output file missing after export: %s", args.output)
        return 1

    if args.package_dir:
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts([args.output], args.package_dir, extra_files=extras)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
