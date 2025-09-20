#!/usr/bin/env python3
# export_rnnt_step.py
# Exports a single-step RNNT decoder+joint (fp32) to ONNX and ExecuTorch .pte.

import argparse
import logging
import torch
import types

import json

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge

from export_common import load_trained_model, make_example_inputs, package_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("export_rnnt_step")

class RNNTStep(torch.nn.Module):
    """
    One decoding step:
      Inputs:
        - y_prev_ids: (B,) int64 token ids
        - h0: (L,B,H) float32
        - c0: (L,B,H) float32
        - enc_t: (B,D) float32 (one encoder frame projected by encoder)
      Outputs:
        - logits: (B,V) float32
        - h1: (L,B,H) float32
        - c1: (L,B,H) float32
    Assumes model.decoder has an embedding + LSTM stack; model.joint combines enc_t & pred_h.
    """
    def __init__(self, model):
        super().__init__()
        # The model, when loaded, has the final vocabulary with the blank token at the end.
        # This is the source of truth.
        # Vocabulary is stored in the joint module, blank_idx in the decoder
        self.vocabulary = model.joint.vocabulary
        self.vocab_size = len(self.vocabulary)
        self.blank_idx = model.decoder.blank_idx

        log.info(f"Initializing RNNTStep with vocab size: {self.vocab_size} and blank_id: {self.blank_idx}")
        if self.vocab_size != 30 or self.blank_idx != 29:
            log.warning(f"Expected vocab_size=30 and blank_id=29 for blank_as_pad=True, but got {self.vocab_size} and {self.blank_idx}")

        # Decoder prediction net
        # Try standard NeMo layout: model.decoder.prednet.{embedding, decoder}
        dec = model.decoder
        prednet = getattr(dec, "prednet", dec)
        # Find embedding
        self.embedding = getattr(prednet, "embedding", None)
        if self.embedding is None:
            # Fallback: first nn.Embedding found
            for m in prednet.modules():
                if isinstance(m, torch.nn.Embedding):
                    self.embedding = m
                    break
        if self.embedding is None:
            raise RuntimeError("Could not locate decoder embedding module.")

        # Find LSTM stack
        self.lstm = getattr(prednet, "decoder", None)
        if self.lstm is None:
            # Common alternative attr
            self.lstm = getattr(prednet, "pred_rnn", None)
        if self.lstm is None or not isinstance(self.lstm, torch.nn.LSTM):
            # Fallback: first nn.LSTM
            lstm_found = None
            for m in prednet.modules():
                if isinstance(m, torch.nn.LSTM):
                    lstm_found = m
                    break
            if lstm_found is None:
                raise RuntimeError("Could not locate decoder LSTM module.")
            self.lstm = lstm_found

        # RNNT joint
        self.joint = model.joint   # expects joint(enc_t, pred_t) -> logits

        # Derive dimensions
        self.num_layers = self.lstm.num_layers
        self.hidden_size = self.lstm.hidden_size

    def forward(self, y_prev_ids: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor, enc_t: torch.Tensor):
        """
        y_prev_ids: (B,) int64
        h0, c0: (L,B,H)
        enc_t: (B,D)
        """
        # (B,E)
        emb = self.embedding(y_prev_ids)
        # LSTM expects (T,B,E); single-timestep T=1
        x = emb.unsqueeze(0)  # (1,B,E)
        out, (h1, c1) = self.lstm(x, (h0, c0))  # out: (1,B,H)
        pred_t = out.squeeze(0)                 # (B,H)

        # Joint expects specific format for NeMo RNNT joint
        # The NeMo joint typically takes encoder_outputs as (B, D, T=1) and decoder_outputs as (B, H, T=1)
        enc_t_3d = enc_t.unsqueeze(-1)  # (B, D) -> (B, D, 1)
        pred_t_3d = pred_t.unsqueeze(-1)  # (B, H) -> (B, H, 1)
        logits = self.joint(encoder_outputs=enc_t_3d, decoder_outputs=pred_t_3d)  # (B, V, 1)
        logits = logits.squeeze(-1)  # (B, V, 1) -> (B, V)
        return logits, h1, c1


def save_pte(module: torch.nn.Module, example_inputs, out_path: str):
    module.eval()
    exported = torch.export.export(module, example_inputs)
    edge = to_edge(exported)
    # Keep step in fp32; XNNPACK still helps on ARM
    edge = edge.to_backend(XnnpackPartitioner())
    prog = edge.to_executorch()
    buf = getattr(prog, "buffer", None)
    if buf is None and hasattr(prog, "to_buffer"):
        buf = prog.to_buffer()
    if buf is None:
        raise RuntimeError("ExecuTorch: unable to get program buffer.")
    with open(out_path, "wb") as f:
        f.write(buf)


def save_runtime_metadata(out_path: str, vocab, blank_id: int):
    """Saves critical metadata for the runtime decoder."""
    # Convert OmegaConf ListConfig to regular list
    vocab_list = list(vocab) if hasattr(vocab, '__iter__') else vocab
    meta = {
        "vocab_size": len(vocab_list),
        "blank_id": blank_id,
        "tokens": vocab_list,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"✓ Saved runtime metadata to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Export single-step RNNT decoder+joint to ONNX and ExecuTorch.")
    ap.add_argument("--checkpoint", required=True, help="Path to trained .ckpt or .nemo artifact.")
    ap.add_argument("--onnx_out", default="rnnt_step_fp32.onnx", help="Output ONNX path.")
    ap.add_argument("--pte_out",  default="rnnt_step_fp32.pte",  help="Output ExecuTorch .pte path.")
    ap.add_argument("--meta_out", default="runtime_meta.json", help="Output runtime metadata JSON path.")
    ap.add_argument("--layers", type=int, default=None, help="Override L (num_layers) if needed.")
    ap.add_argument("--hidden", type=int, default=None, help="Override H (hidden_size) if needed.")
    ap.add_argument("--enc_dim", type=int, default=None, help="Override encoder D if needed.")
    ap.add_argument("--lexicon", help="Optional lexicon to include when packaging")
    ap.add_argument("--package-dir", help="Copy exports (and assets) into this directory")
    ap.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional assets (metadata, trie) to include when packaging",
    )
    args = ap.parse_args()

    # Remove the old, confusing blank_id derivation logic
    model = load_trained_model(args.checkpoint)
    step = RNNTStep(model).eval()

    # Save the metadata that the runtime needs
    save_runtime_metadata(args.meta_out, step.vocabulary, step.blank_idx)

    # Example shapes
    B = 1
    L = args.layers or step.num_layers
    H = args.hidden or step.hidden_size
    if args.enc_dim:
        D = args.enc_dim
    else:
        try:
            feats, lens = make_example_inputs(10, feature_dim=model.cfg.encoder.get("feat_in", 37))
            enc_bdt, _ = model.encoder(audio_signal=feats, length=lens)
            D = enc_bdt.shape[1]
        except Exception:  # pragma: no cover - defensive fallback
            D = 512

    y_prev = torch.tensor([step.blank_idx], dtype=torch.long)
    h0 = torch.zeros(L, B, H)
    c0 = torch.zeros(L, B, H)
    enc_t = torch.randn(B, D)

    # Export ONNX (fp32)
    log.info(f"Exporting ONNX: {args.onnx_out}")
    torch.onnx.export(
        step,
        (y_prev, h0, c0, enc_t),
        args.onnx_out,
        opset_version=17,
        input_names=["y_prev", "h0", "c0", "enc_t"],
        output_names=["logits", "h1", "c1"],
        dynamic_axes={
            "y_prev": {0: "B"},
            "h0":     {0: "L", 1: "B"},
            "c0":     {0: "L", 1: "B"},
            "enc_t":  {0: "B"},
            "logits": {0: "B"},
            "h1":     {0: "L", 1: "B"},
            "c1":     {0: "L", 1: "B"},
        },
    )
    log.info("ONNX export complete.")

    # Export ExecuTorch (fp32)
    log.info(f"Exporting ExecuTorch .pte: {args.pte_out}")
    save_pte(step, (y_prev, h0, c0, enc_t), args.pte_out)
    log.info("ExecuTorch export complete.")

    if args.package_dir:
        artifacts = [args.onnx_out, args.pte_out, args.meta_out]
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        package_artifacts(artifacts, args.package_dir, extra_files=extras)

if __name__ == "__main__":
    main()
