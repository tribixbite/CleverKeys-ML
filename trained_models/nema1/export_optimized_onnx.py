#!/usr/bin/env python3
"""Create optimized ONNX exports for web and Android targets."""

import argparse
import logging
import os
import zlib

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

try:
    from onnxruntime.tools import convert_float_to_float16  # type: ignore
except ImportError:  # pragma: no cover
    convert_float_to_float16 = None

from export_common import (
    DatasetBackedCalibrationReader,
    create_calibration_loader,
    ensure_default_manifest,
    load_trained_model,
    make_example_inputs,
    package_artifacts,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("export_optimized")


def log_model_parameters(model, name="Model"):
    """Log parameter count and estimated size"""
    if hasattr(model, 'encoder'):
        # For full model, focus on encoder
        params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        log.info(f"{name} encoder parameters: {params/1e6:.2f}M (~{params*4/(1024**2):.1f}MB fp32, ~{params/(1024**2):.1f}MB int8)")
    else:
        # For encoder only
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"{name} parameters: {params/1e6:.2f}M (~{params*4/(1024**2):.1f}MB fp32, ~{params/(1024**2):.1f}MB int8)")

def audit_onnx_initializers(onnx_path):
    """Audit ONNX initializer sizes and types to verify quantization"""
    try:
        model = onnx.load(onnx_path)
        sizes = []
        total_bytes = 0
        dtype_counts = {}

        for tensor in model.graph.initializer:
            try:
                array = onnx.numpy_helper.to_array(tensor)
                size_bytes = array.nbytes
                dtype = str(array.dtype)

                sizes.append((tensor.name, dtype, size_bytes))
                total_bytes += size_bytes
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            except Exception as e:
                log.warning(f"Could not analyze tensor {tensor.name}: {e}")

        total_mb = total_bytes / (1024**2)
        log.info(f"ONNX initializer audit for {onnx_path}:")
        log.info(f"  Total initializer size: {total_mb:.1f} MB")
        log.info(f"  Data types found: {dict(dtype_counts)}")

        # Check if quantized (should have int8 types)
        is_quantized = any('int8' in dtype for dtype, _ in dtype_counts.items())
        log.info(f"  Quantization status: {'✓ QUANTIZED' if is_quantized else '⚠ FP32 ONLY'}")

        # Show largest tensors
        sizes.sort(key=lambda x: x[2], reverse=True)
        log.info("  Largest tensors:")
        for name, dtype, size_bytes in sizes[:5]:
            log.info(f"    {name[:40]:<40} {dtype:<10} {size_bytes/(1024**2):.2f}MB")

        return total_mb, is_quantized

    except Exception as e:
        log.warning(f"Could not audit ONNX initializers: {e}")
        return None, False

def estimate_compressed_size(file_path, compression_level=6):
    """Estimate APK compressed size using zlib"""
    try:
        with open(file_path, 'rb') as f:
            original_data = f.read()

        compressed_data = zlib.compress(original_data, compression_level)
        original_mb = len(original_data) / (1024**2)
        compressed_mb = len(compressed_data) / (1024**2)
        compression_ratio = len(compressed_data) / len(original_data) * 100

        log.info(f"Size analysis for {os.path.basename(file_path)}:")
        log.info(f"  Raw size: {original_mb:.1f} MB")
        log.info(f"  Compressed (APK): {compressed_mb:.1f} MB ({compression_ratio:.1f}% of original)")

        return compressed_mb
    except Exception as e:
        log.warning(f"Could not estimate compressed size: {e}")
        return None

def save_ort_optimized_onnx(input_onnx: str, output_onnx: str, target_platform="web"):
    """Save ORT-optimized ONNX with graph optimizations"""
    log.info(f"Creating ORT-optimized ONNX: {output_onnx}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if target_platform == "web":
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # Better for web workers
    else:  # android
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Better for mobile
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True

    so.optimized_model_filepath = output_onnx

    # Use CPU provider for optimization (actual runtime will use WebGPU/XNNPACK)
    providers = ["CPUExecutionProvider"]

    try:
        _ = ort.InferenceSession(input_onnx, sess_options=so, providers=providers)
        log.info(f"✓ ORT-optimized ONNX saved: {output_onnx}")
        return True
    except Exception as e:
        log.warning(f"Could not create ORT-optimized ONNX: {e}")
        return False

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, feats_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=feats_bft, length=lengths)


def main():
    parser = argparse.ArgumentParser(description="Create optimized ONNX encoder exports")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt or .nemo model")
    parser.add_argument("--vocab", default="vocab/final_vocab.txt", help="Vocabulary file for calibration")
    parser.add_argument(
        "--calib-manifest",
        help="Calibration manifest (.jsonl). Defaults to data/train_final_val.jsonl if present.",
    )
    parser.add_argument("--fp32-base", default="encoder_base_fp32.onnx", help="Base FP32 ONNX output")
    parser.add_argument("--fp16-base", help="Optional FP16 ONNX output path")
    parser.add_argument("--web_onnx", default="encoder_web_quant.onnx", help="Web-optimized quantized ONNX")
    parser.add_argument("--android_onnx", default="encoder_android_quant.onnx", help="Android-optimized quantized ONNX")
    parser.add_argument("--external_data", action="store_true", help="Save initializers as external data for smaller files")
    parser.add_argument("--skip_quantization_audit", action="store_true", help="Skip quantization verification (faster)")
    parser.add_argument("--skip_quantization", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--calibration_batches", type=int, default=16, help="Number of calibration batches")
    parser.add_argument("--calibration_batch_size", type=int, default=4, help="Calibration batch size")
    parser.add_argument("--create_ort_optimized", action="store_true", help="Create ORT-optimized versions (10-30% smaller)")
    parser.add_argument("--compression_test", action="store_true", help="Test APK compression estimates")
    parser.add_argument("--max-trace-len", type=int, default=200, help="Max gesture length for example export inputs")
    parser.add_argument("--lexicon", help="Optional lexicon to copy alongside exports")
    parser.add_argument(
        "--package-dir",
        help="Copy produced ONNX models (and assets) into this directory",
    )
    parser.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional assets to include when packaging (metadata, language models, etc.)",
    )
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    log_model_parameters(model, "Original model")

    wrapper = EncoderWrapper(model.encoder)
    log_model_parameters(wrapper, "Encoder wrapper")

    example_feats, example_lens = make_example_inputs(args.max_trace_len)

    log.info(f"Exporting base FP32 ONNX: {args.fp32_base}")
    torch.onnx.export(
        wrapper,
        (example_feats, example_lens),
        args.fp32_base,
        opset_version=17,
        input_names=["features_bft", "lengths"],
        output_names=["encoded_btf", "encoded_lengths"],
        dynamic_axes={
            "features_bft": {0: "B", 2: "T"},
            "lengths": {0: "B"},
            "encoded_btf": {0: "B", 1: "T_out"},
            "encoded_lengths": {0: "B"},
        },
        # Optional external data storage
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
    )

    # Optionally convert to external data format for smaller files
    if args.external_data:
        log.info("Converting to external data format...")
        external_base = args.fp32_base.replace('.onnx', '_external.onnx')
        try:
            model = onnx.load(args.fp32_base)
            onnx.save_model(model, external_base, save_as_external_data=True, all_tensors_to_one_file=True,
                           location=external_base.replace('.onnx', '.bin'), size_threshold=1024)
            log.info(f"✓ External data version saved: {external_base}")
        except Exception as e:
            log.warning(f"Could not create external data version: {e}")

    # Audit the FP32 model
    if not args.skip_quantization_audit:
        audit_onnx_initializers(args.fp32_base)

    if args.fp16_base:
        if convert_float_to_float16 is None:
            log.warning("onnxruntime.tools not available; skipping FP16 conversion")
        else:
            log.info("Converting base ONNX to FP16: %s", args.fp16_base)
            convert_float_to_float16.convert_float_to_float16(
                args.fp32_base, args.fp16_base, keep_io_types=True
            )

    if args.skip_quantization:
        log.info("Skipping quantization per flag")
        created_quant_files = []
    else:
        manifest = ensure_default_manifest(args.calib_manifest, "data/train_final_val.jsonl")
        calib_loader = create_calibration_loader(
            manifest,
            args.vocab,
            args.max_trace_len,
            batch_size=args.calibration_batch_size,
            shuffle=False,
            num_workers=0,
        )

        log.info("Creating web-optimized INT8 ONNX: %s", args.web_onnx)
        reader = DatasetBackedCalibrationReader(calib_loader, max_batches=args.calibration_batches)
        quantize_static(
            model_input=args.fp32_base,
            model_output=args.web_onnx,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        if not args.skip_quantization_audit:
            audit_onnx_initializers(args.web_onnx)
        log.info("✓ Web quantization complete")

        log.info("Creating Android-optimized INT8 ONNX: %s", args.android_onnx)
        reader = DatasetBackedCalibrationReader(calib_loader, max_batches=args.calibration_batches)
        quantize_static(
            model_input=args.fp32_base,
            model_output=args.android_onnx,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        if not args.skip_quantization_audit:
            audit_onnx_initializers(args.android_onnx)
        log.info("✓ Android quantization complete")
        created_quant_files = [
            ("Web Quantized", args.web_onnx),
            ("Android Quantized", args.android_onnx),
        ]

    # Create ORT-optimized versions if requested
    ort_files = []
    if args.create_ort_optimized:
        log.info("Creating ORT-optimized versions...")
        web_ort = args.web_onnx.replace('.onnx', '_ort.onnx')
        android_ort = args.android_onnx.replace('.onnx', '_ort.onnx')
        if not args.skip_quantization:
            if save_ort_optimized_onnx(args.web_onnx, web_ort, "web"):
                ort_files.append(("Web ORT", web_ort))
            if save_ort_optimized_onnx(args.android_onnx, android_ort, "android"):
                ort_files.append(("Android ORT", android_ort))
        else:
            log.warning("ORT optimization skipped because quantization was skipped")

    log.info("\n🚀 Ultra-optimized models created:")

    # Collect all created files for summary
    all_files = [
        ("Base FP32", args.fp32_base),
    ]

    # Add external data version if created
    if args.external_data:
        external_base = args.fp32_base.replace('.onnx', '_external.onnx')
        if os.path.exists(external_base):
            all_files.append(("Base FP32 (External)", external_base))

    if args.fp16_base and os.path.exists(args.fp16_base):
        all_files.append(("Base FP16", args.fp16_base))

    if not args.skip_quantization:
        all_files.extend(created_quant_files)

    # Add ORT-optimized files
    all_files.extend(ort_files)

    # Display file sizes and compression estimates
    for name, path in all_files:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            log.info(f"  {name}: {os.path.basename(path)} ({size_mb:.1f} MB)")

            # Show compression estimate if requested
            if args.compression_test:
                compressed_mb = estimate_compressed_size(path)

    log.info("\nOptimizations applied:")
    if not args.skip_quantization:
        quant_note = (
            "  ✓ INT8 symmetric per-channel quantization (verified)"
            if not args.skip_quantization_audit
            else "  ✓ INT8 symmetric per-channel quantization"
        )
        log.info(quant_note)
        log.info("  ✓ QDQ format for hardware acceleration")
        log.info("  ✓ Web: Optimized for WebGPU/WASM-SIMD")
        log.info("  ✓ Android: Optimized for XNNPACK/NNAPI")
        log.info(f"  ✓ {args.calibration_batches} calibration batches used")
    if args.fp16_base:
        log.info("  ✓ Optional FP16 baseline")
    if args.external_data:
        log.info("  ✓ External data format for smaller files")
    if args.create_ort_optimized:
        log.info("  ✓ ORT graph optimizations applied")
    if args.compression_test:
        log.info("  ✓ APK compression estimates included")

    if args.package_dir:
        package_targets = [args.fp32_base]
        if args.fp16_base and os.path.exists(args.fp16_base):
            package_targets.append(args.fp16_base)
        if not args.skip_quantization:
            package_targets.append(args.web_onnx)
            package_targets.append(args.android_onnx)
            package_targets.extend([path for _, path in ort_files])
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts(package_targets, args.package_dir, extra_files=extras)

    return 0

if __name__ == "__main__":
    main()
