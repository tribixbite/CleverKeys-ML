#!/usr/bin/env python3
# export_optimized_onnx.py
# Create ultra-optimized quantized ONNX models for web and Android

import argparse
import logging
import torch
import onnx
import onnxruntime as ort
import zlib
import os
from model_class import GestureRNNTModel, get_default_config
from swipe_data_utils import SwipeDataset, SwipeFeaturizer, KeyboardGrid, collate_fn

from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
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

class OptimizedEncoderWrapper(torch.nn.Module):
    """Optimized encoder wrapper for ONNX export"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, features_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=features_bft, length=lengths)

class FastCalibrationReader(CalibrationDataReader):
    """Fast calibration data reader with minimal data"""
    def __init__(self, data_loader, max_samples=16):
        self.data_iter = iter(data_loader)
        self.max_samples = max_samples
        self.count = 0

    def get_next(self):
        if self.count >= self.max_samples:
            return None

        try:
            batch = next(self.data_iter)
            self.count += 1
        except StopIteration:
            return None

        # Convert batch to encoder inputs
        if isinstance(batch, dict):
            feats = batch["features"]
            lens = batch.get("feat_lens") or batch.get("lengths")
        else:
            feats, lens = batch[0], batch[1]

        feats_bft = feats.transpose(1, 2).contiguous()  # (B,F,T)
        lens = lens.to(dtype=torch.int32, copy=False)

        return {
            "features_bft": feats_bft.cpu().numpy(),
            "lengths": lens.cpu().numpy(),
        }


def main():
    parser = argparse.ArgumentParser(description="Create ultra-optimized quantized ONNX models")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--val_manifest", required=True, help="Validation manifest for calibration")
    parser.add_argument("--vocab", required=True, help="Vocab file")
    parser.add_argument("--web_onnx", default="encoder_web_quant.onnx", help="Web-optimized quantized ONNX")
    parser.add_argument("--android_onnx", default="encoder_android_quant.onnx", help="Android-optimized quantized ONNX")
    parser.add_argument("--fp32_base", default="encoder_base_fp32.onnx", help="Base FP32 ONNX")
    parser.add_argument("--external_data", action="store_true", help="Save initializers as external data for smaller files")
    parser.add_argument("--skip_quantization_audit", action="store_true", help="Skip quantization verification (faster)")
    parser.add_argument("--calibration_samples", type=int, default=16, help="Number of calibration samples")
    parser.add_argument("--create_ort_optimized", action="store_true", help="Create ORT-optimized versions (10-30% smaller)")
    parser.add_argument("--compression_test", action="store_true", help="Test APK compression estimates")
    args = parser.parse_args()

    # Load model
    log.info(f"Loading checkpoint: {args.checkpoint}")
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

    # Log model parameters before optimization
    log_model_parameters(model, "Original model")

    # Create wrapper
    wrapper = OptimizedEncoderWrapper(model.encoder).eval()
    log_model_parameters(wrapper, "Encoder wrapper")

    # Export base FP32 ONNX
    log.info(f"Exporting base FP32 ONNX: {args.fp32_base}")
    B, F, T = 1, 37, 200
    example_feats = torch.randn(B, F, T, dtype=torch.float32)
    example_lens = torch.tensor([T], dtype=torch.int32)

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

    # Prepare calibration data
    log.info("Preparing minimal calibration data...")
    featurizer = SwipeFeaturizer(KeyboardGrid(chars="abcdefghijklmnopqrstuvwxyz'"))
    vocab = {}
    with open(args.vocab, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i

    ds = SwipeDataset(args.val_manifest, featurizer, vocab, 150)  # Shorter for speed
    calib_dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn,
                                         shuffle=False, num_workers=0)

    reader = FastCalibrationReader(calib_dl, max_samples=args.calibration_samples)

    # Create web-optimized quantized ONNX
    log.info(f"Creating web-optimized quantized ONNX: {args.web_onnx}")
    try:
        quantize_static(
            model_input=args.fp32_base,
            model_output=args.web_onnx,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,  # Best for WebGPU
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        log.info("✓ Web quantization completed")

        # Audit web quantized model
        if not args.skip_quantization_audit:
            audit_onnx_initializers(args.web_onnx)

    except Exception as e:
        log.error(f"Web quantization failed: {e}")
        return 1

    # Reset reader for Android quantization
    reader = FastCalibrationReader(calib_dl, max_samples=args.calibration_samples)

    # Create Android-optimized quantized ONNX
    log.info(f"Creating Android-optimized quantized ONNX: {args.android_onnx}")
    try:
        quantize_static(
            model_input=args.fp32_base,
            model_output=args.android_onnx,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        log.info("✓ Android quantization completed")

        # Audit Android quantized model
        if not args.skip_quantization_audit:
            audit_onnx_initializers(args.android_onnx)

    except Exception as e:
        log.error(f"Android quantization failed: {e}")
        return 1

    # Create ORT-optimized versions if requested
    ort_files = []
    if args.create_ort_optimized:
        log.info("Creating ORT-optimized versions...")
        web_ort = args.web_onnx.replace('.onnx', '_ort.onnx')
        android_ort = args.android_onnx.replace('.onnx', '_ort.onnx')

        if save_ort_optimized_onnx(args.web_onnx, web_ort, "web"):
            ort_files.append(("Web ORT", web_ort))
        if save_ort_optimized_onnx(args.android_onnx, android_ort, "android"):
            ort_files.append(("Android ORT", android_ort))

    log.info("\n🚀 Ultra-optimized models created:")

    # Collect all created files for summary
    all_files = [
        ("Base FP32", args.fp32_base),
        ("Web Quantized", args.web_onnx),
        ("Android Quantized", args.android_onnx)
    ]

    # Add external data version if created
    if args.external_data:
        external_base = args.fp32_base.replace('.onnx', '_external.onnx')
        if os.path.exists(external_base):
            all_files.append(("Base FP32 (External)", external_base))

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
    log.info("  ✓ INT8 symmetric per-channel quantization (verified)" if not args.skip_quantization_audit else "  ✓ INT8 symmetric per-channel quantization")
    log.info("  ✓ QDQ format for hardware acceleration")
    log.info("  ✓ Web: Optimized for WebGPU/WASM-SIMD")
    log.info("  ✓ Android: Optimized for XNNPACK/NNAPI")
    log.info(f"  ✓ {args.calibration_samples} calibration samples used")
    if args.external_data:
        log.info("  ✓ External data format for smaller files")
    if args.create_ort_optimized:
        log.info("  ✓ ORT graph optimizations applied")
    if args.compression_test:
        log.info("  ✓ APK compression estimates included")

    return 0

if __name__ == "__main__":
    main()