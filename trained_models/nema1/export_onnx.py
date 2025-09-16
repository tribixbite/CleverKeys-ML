#!/usr/bin/env python3
# export_onnx_quantized.py
# Export RNNT encoder to ONNX (FP32) then quantize to INT8 (QDQ) with ONNX Runtime.

import os
import argparse
import logging
import torch
import onnx
import onnxruntime as ort

# NeMo + our model class
from model_class import GestureRNNTModel, get_default_config
from swipe_data_utils import SwipeDataset, SwipeFeaturizer, KeyboardGrid, collate_fn

from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("export_onnx")

class EncoderWrapper(torch.nn.Module):
    """Wraps the ConformerEncoder: forward(B,F,T), lengths(B) -> (encoded, encoded_len)"""
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder.eval()
    def forward(self, feats_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=feats_bft, length=lengths)

def _to_encoder_inputs(batch):
    # collate_fn -> (features, feature_lengths, tokens, token_lengths)
    if isinstance(batch, dict):
        feats = batch["features"]
        lens  = batch.get("feat_lens") or batch.get("lengths")
        if lens is None:
            raise ValueError("Calibration batch dict missing 'feat_lens'/'lengths'")
    else:
        feats, lens = batch[0], batch[1]
    feats_bft = feats.transpose(1, 2).contiguous()  # (B,F,T)
    lens = lens.to(dtype=torch.int32, copy=False)    # OR int64 if you prefer
    return feats_bft, lens

class SwipeCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, in_feats_name: str, in_len_name: str):
        self.data_iter = iter(data_loader)
        self.in_feats = in_feats_name
        self.in_len   = in_len_name
    def get_next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            return None
        feats_bft, lens = _to_encoder_inputs(batch)
        return {
            self.in_feats: feats_bft.cpu().numpy(),
            self.in_len:   lens.cpu().numpy(),
        }

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Export RNNT encoder to quantized ONNX (QDQ INT8).")
    ap.add_argument("--nemo_model", required=True, help="Path to trained .nemo.")
    ap.add_argument("--val_manifest", required=True, help="Validation manifest for calibration.")
    ap.add_argument("--vocab", required=True, help="Path to vocab.txt (for dataset init).")
    ap.add_argument("--fp32_onnx", default="encoder_fp32.onnx", help="Intermediate FP32 ONNX path.")
    ap.add_argument("--output", default="encoder_int8_qdq.onnx", help="Quantized ONNX output.")
    ap.add_argument("--calib_batches", type=int, default=64, help="Calibration batches.")
    ap.add_argument("--calib_bs", type=int, default=8, help="Calibration batch size.")
    ap.add_argument("--max_trace_len", type=int, default=200, help="Nominal T used for export sample.")
    args = ap.parse_args()

    # Check if input is .nemo or .ckpt
    if args.nemo_model.endswith('.ckpt'):
        log.info(f"Loading checkpoint directly: {args.nemo_model}")
        # Load checkpoint and create model compatible with it
        ckpt = torch.load(args.nemo_model, map_location="cpu", weights_only=False)

        # Create model with proper config (load from hyperparameters in checkpoint if available)
        if 'hyper_parameters' in ckpt:
            cfg = ckpt['hyper_parameters']['cfg']
        else:
            # Use default config but with torch.compile disabled
            cfg = get_default_config()

        model = GestureRNNTModel(cfg).eval()

        # Clean up torch.compile keys
        state_dict = ckpt["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder._orig_mod."):
                new_key = key.replace("encoder._orig_mod.", "encoder.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore missing keys
    else:
        log.info(f"Loading trained model from {args.nemo_model}")
        model = GestureRNNTModel.restore_from(args.nemo_model, map_location="cpu").eval()

    encoder = model.encoder

    wrapper = EncoderWrapper(encoder).eval()

    # Example inputs for graph (B=1)
    B, F, T = 1, 37, int(args.max_trace_len)
    example_feats_bft = torch.randn(B, F, T, dtype=torch.float32)
    example_lens = torch.tensor([T], dtype=torch.int32)

    log.info(f"Exporting FP32 ONNX to {args.fp32_onnx}")
    torch.onnx.export(
        wrapper,
        (example_feats_bft, example_lens),
        args.fp32_onnx,
        opset_version=17,
        input_names=["features_bft", "lengths"],
        output_names=["encoded_btf", "encoded_lengths"],
        dynamic_axes={
            "features_bft": {0: "B", 2: "T"},
            "lengths":      {0: "B"},
            "encoded_btf":  {0: "B", 1: "T_out"},
            "encoded_lengths": {0: "B"},
        },
    )
    onnx_model = onnx.load(args.fp32_onnx)
    onnx.checker.check_model(onnx_model)
    log.info("FP32 ONNX export verified.")

    # Build calibration loader
    log.info("Preparing calibration loader...")
    featurizer = SwipeFeaturizer(KeyboardGrid(chars="abcdefghijklmnopqrstuvwxyz'"))
    vocab = {}
    with open(args.vocab, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    ds = SwipeDataset(args.val_manifest, featurizer, vocab, args.max_trace_len)
    calib_dl = torch.utils.data.DataLoader(ds, batch_size=args.calib_bs, collate_fn=collate_fn, shuffle=False, num_workers=0)

    # Discover input names robustly from the session
    sess = ort.InferenceSession(args.fp32_onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    assert set(in_names) == {"features_bft", "lengths"}, f"Unexpected ONNX inputs: {in_names}"
    reader = SwipeCalibrationDataReader(calib_dl, "features_bft", "lengths")

    log.info(f"Quantizing to INT8 QDQ -> {args.output}")
    quantize_static(
        model_input=args.fp32_onnx,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    log.info(f"✓ Saved quantized ONNX to {args.output}")

    # Optional: remove fp32 file
    # os.remove(args.fp32_onnx)

    log.info("Done. For web, load with onnxruntime-web (WASM SIMD or WebGPU). For Android, use onnxruntime-android.")

if __name__ == "__main__":
    main()
