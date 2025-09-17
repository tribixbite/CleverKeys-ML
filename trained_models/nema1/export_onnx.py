#!/usr/bin/env python3
"""Export RNNT encoder to ONNX (FP32 baseline, optional FP16 + INT8 QDQ)."""

import argparse
import logging
from pathlib import Path

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("export_onnx")


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder.eval()

    def forward(self, feats_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=feats_bft, length=lengths)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export RNNT encoder to ONNX formats")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .ckpt or .nemo artifact")
    parser.add_argument("--vocab", default="vocab/final_vocab.txt", help="Vocabulary file for calibration dataset")
    parser.add_argument(
        "--calib-manifest",
        help="Calibration manifest (.jsonl). Defaults to data/train_final_val.jsonl if present.",
    )
    parser.add_argument("--fp32-output", default="encoder_fp32.onnx", help="Baseline FP32 ONNX path")
    parser.add_argument("--int8-output", default="encoder_int8_qdq.onnx", help="INT8 QDQ ONNX output")
    parser.add_argument("--fp16-output", help="Optional FP16 ONNX output path")
    parser.add_argument("--skip-int8", action="store_true", help="Skip INT8 quantization step")
    parser.add_argument("--lexicon", help="Optional lexicon file to bundle with outputs")
    parser.add_argument(
        "--package-dir",
        help="Copy exported models (and optional assets) into this directory",
    )
    parser.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional files (metadata, trie, etc.) to include when packaging",
    )
    parser.add_argument("--calib-batch-size", type=int, default=16, help="Calibration batch size")
    parser.add_argument("--calib-batches", type=int, default=64, help="Number of calibration batches")
    parser.add_argument("--max-trace-len", type=int, default=200, help="Max gesture length")
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    wrapper = EncoderWrapper(model.encoder).eval()

    example_feats, example_lens = make_example_inputs(args.max_trace_len)

    LOG.info("Exporting FP32 ONNX -> %s", args.fp32_output)
    torch.onnx.export(
        wrapper,
        (example_feats, example_lens),
        args.fp32_output,
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
    onnx_model = onnx.load(args.fp32_output)
    onnx.checker.check_model(onnx_model)
    LOG.info("FP32 ONNX export verified")

    if args.fp16_output:
        if convert_float_to_float16 is None:
            LOG.warning("onnxruntime.tools not available; skipping FP16 conversion")
        else:
            LOG.info("Converting FP32 -> FP16 ONNX -> %s", args.fp16_output)
            convert_float_to_float16.convert_float_to_float16(
                args.fp32_output, args.fp16_output, keep_io_types=True
            )

    if args.skip_int8:
        LOG.info("INT8 quantization skipped")
        return

    manifest = ensure_default_manifest(args.calib_manifest, "data/train_final_val.jsonl")
    dataloader = create_calibration_loader(
        manifest,
        args.vocab,
        args.max_trace_len,
        batch_size=args.calib_batch_size,
        shuffle=False,
        num_workers=0,
    )

    reader = DatasetBackedCalibrationReader(dataloader, max_batches=args.calib_batches)
    LOG.info("Quantizing FP32 ONNX -> INT8 QDQ: %s", args.int8_output)
    quantize_static(
        model_input=args.fp32_output,
        model_output=args.int8_output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    LOG.info("✓ Saved INT8 ONNX to %s", args.int8_output)

    # Basic I/O check for quantized model
    sess = ort.InferenceSession(args.int8_output, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    assert input_names == {"features_bft", "lengths"}, f"Unexpected inputs: {input_names}"
    LOG.info("Done. Load with onnxruntime (web, Android, or server).")

    if args.package_dir:
        artifacts = [args.fp32_output]
        if args.fp16_output and Path(args.fp16_output).exists():
            artifacts.append(args.fp16_output)
        if not args.skip_int8 and Path(args.int8_output).exists():
            artifacts.append(args.int8_output)
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts(artifacts, args.package_dir, extra_files=extras)


if __name__ == "__main__":
    main()
