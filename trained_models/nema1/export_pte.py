#!/usr/bin/env python3
# export_executorch.py
# Quantize the RNNT encoder to ExecuTorch .pte with PT2E + XNNPACK for Android.

import os
import argparse
import logging
import torch

# NeMo + your training class
import nemo
import nemo.collections.asr as nemo_asr
from train_final import GestureRNNTModel  # your subclass from training
from swipe_data_utils import SwipeDataset, SwipeFeaturizer, KeyboardGrid, collate_fn

# ExecuTorch PT2E
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("export_executorch")

def _to_encoder_inputs(batch):
    """
    Collate output expected: (features, feature_lengths, tokens, token_lengths)
    Features: (B, T, F) -> need (B, F, T); lengths int32/long (B,)
    """
    if isinstance(batch, dict):
        feats = batch["features"]
        lens  = batch.get("feat_lens") or batch.get("lengths")
        if lens is None:
            raise ValueError("Calibration batch dict missing 'feat_lens'/'lengths'")
    else:
        feats, lens = batch[0], batch[1]
    feats = feats.transpose(1, 2).contiguous()  # (B, F, T)
    lens  = lens.to(dtype=torch.int32, copy=False)
    return feats, lens

@torch.no_grad()
def calibrate_encoder(prepared_encoder: torch.nn.Module,
                      val_manifest: str,
                      vocab_path: str,
                      num_batches: int = 64,
                      batch_size: int = 16,
                      max_trace_len: int = 200):
    log.info(f"Calibrating encoder with up to {num_batches} batches (bs={batch_size})...")
    featurizer = SwipeFeaturizer(KeyboardGrid(chars="abcdefghijklmnopqrstuvwxyz'"))
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    ds = SwipeDataset(val_manifest, featurizer, vocab, max_trace_len)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0
    )
    prepared_encoder.eval()
    for i, batch in enumerate(dl):
        if i >= num_batches:
            break
        feats_bft, lens = _to_encoder_inputs(batch)
        prepared_encoder(feats_bft, lens)  # signature: (audio_signal(B,F,T), length(B,))
        if (i + 1) % 10 == 0:
            log.info(f"  Calibrated {i+1}/{num_batches} batches")
    log.info("Calibration complete.")

def main():
    ap = argparse.ArgumentParser(description="Export quantized ExecuTorch (.pte) encoder from a trained NeMo RNNT.")
    ap.add_argument("--nemo_model", required=True, help="Path to trained .nemo produced by your training script.")
    ap.add_argument("--val_manifest", required=True, help="Validation manifest for calibration.")
    ap.add_argument("--vocab", required=True, help="Path to vocab.txt used in training.")
    ap.add_argument("--output", default="encoder_quant_xnnpack.pte", help="Output .pte path.")
    ap.add_argument("--calib_batches", type=int, default=64, help="Max calibration batches.")
    ap.add_argument("--calib_bs", type=int, default=16, help="Calibration batch size.")
    ap.add_argument("--max_trace_len", type=int, default=200, help="Max trace len (T) used at export/calib.")
    args = ap.parse_args()

    # 1) Load your trained subclass (ensures same module graph)
    log.info(f"Loading trained model from {args.nemo_model}")
    model = GestureRNNTModel.restore_from(args.nemo_model, map_location="cpu").eval()
    encoder = model.encoder  # ConformerEncoder

    # 2) Prepare PT2E quant with XNNPACK
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    log.info("Inserting observers (prepare_pt2e)...")
    prepared_encoder = prepare_pt2e(encoder, quantizer)

    # 3) Calibrate with representative data (tuple-aware)
    calibrate_encoder(prepared_encoder, args.val_manifest, args.vocab,
                      num_batches=args.calib_batches, batch_size=args.calib_bs,
                      max_trace_len=args.max_trace_len)

    # 4) Convert to quantized
    log.info("Converting to quantized encoder (convert_pt2e)...")
    quant_encoder = convert_pt2e(prepared_encoder).eval()

    # 5) Lower to ExecuTorch (Edge IR -> XNNPACK partition -> ExecuTorch program)
    # Example inputs for export graph (B=1, F=37, T=max_trace_len)
    B, F, T = 1, 37, int(args.max_trace_len)
    example_feats_bft = torch.randn(B, F, T, dtype=torch.float32)
    example_lens = torch.tensor([T], dtype=torch.int32)

    log.info("Lowering to ExecuTorch Edge IR...")
    exported = torch.export.export(quant_encoder, (example_feats_bft, example_lens))
    edge = to_edge(exported)
    edge = edge.to_backend(XnnpackPartitioner())
    prog = edge.to_executorch()

    buf = getattr(prog, "buffer", None)
    if buf is None and hasattr(prog, "to_buffer"):
        buf = prog.to_buffer()
    if buf is None:
        raise RuntimeError("Unable to obtain ExecuTorch program buffer.")

    with open(args.output, "wb") as f:
        f.write(buf)
    log.info(f"✓ Saved ExecuTorch encoder to {args.output}")
    log.info("Note: decoder+joint stay float in app (small; run with PyTorch Lite or ExecuTorch unquantized).")

if __name__ == "__main__":
    main()
