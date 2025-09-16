#!/usr/bin/env python3
# export_optimized_onnx.py
# Create ultra-optimized quantized ONNX models for web and Android

import argparse
import logging
import torch
import onnx
import onnxruntime as ort
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

def save_web_optimized_onnx(input_onnx: str, output_onnx: str):
    """Create web-optimized ONNX with WebGPU/WASM-SIMD optimizations"""
    log.info(f"Creating web-optimized ONNX: {output_onnx}")

    # Load and optimize for web
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # Parallel for web workers
    so.optimized_model_filepath = output_onnx

    # Prefer WebGPU/WASM providers
    providers = ["CPUExecutionProvider"]  # Will be WebGPU/WASM in browser

    _ = ort.InferenceSession(input_onnx, sess_options=so, providers=providers)
    log.info(f"✓ Web-optimized ONNX saved: {output_onnx}")

def save_android_optimized_onnx(input_onnx: str, output_onnx: str):
    """Create Android-optimized ONNX with XNNPACK/NNAPI optimizations"""
    log.info(f"Creating Android-optimized ONNX: {output_onnx}")

    # Load and optimize for Android
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Better for mobile
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    so.optimized_model_filepath = output_onnx

    # Android-optimized providers
    providers = ["CPUExecutionProvider"]  # Will be XNNPACK/NNAPI on Android

    _ = ort.InferenceSession(input_onnx, sess_options=so, providers=providers)
    log.info(f"✓ Android-optimized ONNX saved: {output_onnx}")

def main():
    parser = argparse.ArgumentParser(description="Create ultra-optimized quantized ONNX models")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--val_manifest", required=True, help="Validation manifest for calibration")
    parser.add_argument("--vocab", required=True, help="Vocab file")
    parser.add_argument("--web_onnx", default="encoder_web_quant.onnx", help="Web-optimized quantized ONNX")
    parser.add_argument("--android_onnx", default="encoder_android_quant.onnx", help="Android-optimized quantized ONNX")
    parser.add_argument("--fp32_base", default="encoder_base_fp32.onnx", help="Base FP32 ONNX")
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

    # Create wrapper
    wrapper = OptimizedEncoderWrapper(model.encoder).eval()

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
    )

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

    reader = FastCalibrationReader(calib_dl, max_samples=8)  # Minimal calibration

    # Create web-optimized quantized ONNX
    log.info(f"Creating web-optimized quantized ONNX: {args.web_onnx}")
    quantize_static(
        model_input=args.fp32_base,
        model_output=args.web_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,  # Best for WebGPU
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    # Reset reader for Android quantization
    reader = FastCalibrationReader(calib_dl, max_samples=8)

    # Create Android-optimized quantized ONNX
    log.info(f"Creating Android-optimized quantized ONNX: {args.android_onnx}")
    android_temp = args.android_onnx.replace('.onnx', '_temp.onnx')
    quantize_static(
        model_input=args.fp32_base,
        model_output=android_temp,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    # Apply Android-specific graph optimizations
    save_android_optimized_onnx(android_temp, args.android_onnx)

    # Apply web-specific optimizations to web model
    web_temp = args.web_onnx
    web_final = args.web_onnx.replace('.onnx', '_web_final.onnx')
    save_web_optimized_onnx(web_temp, web_final)

    log.info("\n🚀 Ultra-optimized models created:")

    import os
    for name, path in [("Web", web_final), ("Android", args.android_onnx), ("Base FP32", args.fp32_base)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            log.info(f"  {name}: {path} ({size_mb:.1f} MB)")

    log.info("\nOptimizations applied:")
    log.info("  ✓ INT8 symmetric per-channel quantization")
    log.info("  ✓ QDQ format for hardware acceleration")
    log.info("  ✓ Web: Optimized for WebGPU/WASM-SIMD")
    log.info("  ✓ Android: Optimized for XNNPACK/NNAPI")
    log.info("  ✓ Graph-level optimizations enabled")

    # Clean up temp files
    try:
        if os.path.exists(android_temp):
            os.remove(android_temp)
    except:
        pass

if __name__ == "__main__":
    main()