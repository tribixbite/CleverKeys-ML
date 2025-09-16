#!/usr/bin/env python3
# export_pte_fp32.py
# Export non-quantized encoder to ExecuTorch .pte format

import os
import argparse
import logging
import torch

# NeMo + our model class
from model_class import GestureRNNTModel, get_default_config
from swipe_data_utils import SwipeDataset, SwipeFeaturizer, KeyboardGrid, collate_fn

# ExecuTorch
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("export_pte_fp32")

def main():
    ap = argparse.ArgumentParser(description="Export non-quantized ExecuTorch (.pte) encoder from a trained checkpoint.")
    ap.add_argument("--nemo_model", required=True, help="Path to trained .ckpt/.nemo.")
    ap.add_argument("--output", default="encoder_fp32.pte", help="Output .pte path.")
    ap.add_argument("--max_trace_len", type=int, default=200, help="Max trace len (T) used at export.")
    args = ap.parse_args()

    # Check if input is .nemo or .ckpt
    if args.nemo_model.endswith('.ckpt'):
        log.info(f"Loading checkpoint directly: {args.nemo_model}")
        ckpt = torch.load(args.nemo_model, map_location="cpu", weights_only=False)

        if 'hyper_parameters' in ckpt:
            cfg = ckpt['hyper_parameters']['cfg']
        else:
            cfg = get_default_config()

        model = GestureRNNTModel(cfg).eval()

        state_dict = ckpt["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder._orig_mod."):
                new_key = key.replace("encoder._orig_mod.", "encoder.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=False)
    else:
        log.info(f"Loading trained model from {args.nemo_model}")
        model = GestureRNNTModel.restore_from(args.nemo_model, map_location="cpu").eval()

    encoder = model.encoder

    # Wrap the encoder for export
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, feats_bft, lengths):
            return self.encoder(audio_signal=feats_bft, length=lengths)

    wrapped_encoder = EncoderWrapper(encoder).eval()

    # Example inputs for export
    B, F, T = 1, 37, int(args.max_trace_len)
    example_feats_bft = torch.randn(B, F, T, dtype=torch.float32)
    example_lens = torch.tensor([T], dtype=torch.int32)

    log.info("Exporting to ExecuTorch Edge IR...")
    exported = torch.export.export(wrapped_encoder, (example_feats_bft, example_lens))
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

if __name__ == "__main__":
    main()