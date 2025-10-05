#!/usr/bin/env python3
"""
Export + optional quantization for Squeezeformer-CTC GestureCTCModel.

Usage:
  python new/export_ctc_onnx.py \
    --checkpoint path/to/epoch=XX-wer=....ckpt \
    --outdir web-demo/models/ctc \
    [--quantize]

Requires: train_squeezeformer_ctc.py (GestureCTCModel) to be importable.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

try:
    from train_squeezeformer_ctc import GestureCTCModel
except ImportError as e:
    print("Error: Could not import GestureCTCModel from train_squeezeformer_ctc.py.")
    print("Make sure train_squeezeformer_ctc.py is importable (same dir or PYTHONPATH).")
    print(e)
    sys.exit(1)


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Command failed:", " ".join(cmd))
        print(proc.stderr)
        sys.exit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export GestureCTCModel to ONNX + optional quant")
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--outdir', required=True, type=Path)
    ap.add_argument('--quantize', action='store_true', help='Apply dynamic 8-bit quantization via optimum-cli')
    ap.add_argument('--onnx-name', default='gesture_model.onnx')
    ap.add_argument('--quant-name', default='gesture_model_quant.onnx')
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.outdir / args.onnx_name
    quant_path = args.outdir / args.quant_name

    print(f"🚀 Loading model from {args.checkpoint} ...")
    try:
        model = GestureCTCModel.load_from_checkpoint(str(args.checkpoint), map_location='cpu')
        model.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        sys.exit(1)

    print(f"📦 Exporting to ONNX -> {onnx_path}")
    try:
        model.export(str(onnx_path))
        print("✅ Export complete.")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

    if args.quantize:
        print(f"✨ Quantizing with optimum-cli -> {quant_path}")
        cmd = [
            'optimum-cli','onnxruntime','quantize',
            '--onnx_path', str(onnx_path),
            '--output', str(quant_path),
            '--quantization_approach','dynamic',
            '--log-level','INFO'
        ]
        try:
            run_cmd(cmd)
            print("✅ Quantization complete.")
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            sys.exit(1)

    print("🎉 Done.")


if __name__ == '__main__':
    main()

