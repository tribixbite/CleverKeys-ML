#!/usr/bin/env python3
# export_rnnt_step.py
# Exports a single-step RNNT decoder+joint (fp32) to ONNX and ExecuTorch .pte.

import argparse
import logging
import torch
import types

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
        self.blank_idx = 0
        self.vocab_size = len(model.cfg.labels)

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


def main():
    ap = argparse.ArgumentParser(description="Export single-step RNNT decoder+joint to ONNX and ExecuTorch.")
    ap.add_argument("--checkpoint", required=True, help="Path to trained .ckpt or .nemo artifact.")
    ap.add_argument("--onnx_out", default="rnnt_step_fp32.onnx", help="Output ONNX path.")
    ap.add_argument("--pte_out",  default="rnnt_step_fp32.pte",  help="Output ExecuTorch .pte path.")
    ap.add_argument("--layers", type=int, default=None, help="Override L (num_layers) if needed.")
    ap.add_argument("--hidden", type=int, default=None, help="Override H (hidden_size) if needed.")
    ap.add_argument("--enc_dim", type=int, default=None, help="Override encoder D if needed.")
    ap.add_argument("--vocab", type=str, default=None, help="Path to vocabulary file for deriving blank_id.")
    ap.add_argument("--lexicon", help="Optional lexicon to include when packaging")
    ap.add_argument("--package-dir", help="Copy exports (and assets) into this directory")
    ap.add_argument(
        "--bundle-assets",
        nargs="*",
        default=None,
        help="Additional assets (metadata, trie) to include when packaging",
    )
    args = ap.parse_args()

    # Derive blank_id from vocabulary if provided
    blank_id = 0  # default fallback
    if args.vocab:
        try:
            with open(args.vocab, "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f if line.strip()]
            token_to_idx = {tok: i for i, tok in enumerate(tokens)}
            blank_id = token_to_idx.get("<blank>", 0)
            if "<blank>" not in token_to_idx:
                log.warning("'<blank>' token not found in %s; defaulting blank_id=0", args.vocab)
            log.info(f"Derived blank_id={blank_id} from vocab file: {args.vocab}")
        except Exception as e:
            log.warning(f"Failed to load vocab file {args.vocab}: {e}. Using default blank_id=0")

    # Check if input is .nemo or .ckpt
    model = load_trained_model(args.checkpoint)

    step = RNNTStep(model).eval()
    # Override the step's blank_idx with derived value
    step.blank_idx = blank_id

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

    y_prev = torch.tensor([blank_id], dtype=torch.long)
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
        artifacts = [args.onnx_out, args.pte_out]
        extras = list(args.bundle_assets or [])
        if args.lexicon:
            extras.append(args.lexicon)
        if args.vocab:
            extras.append(args.vocab)
        package_artifacts(artifacts, args.package_dir, extra_files=extras)

if __name__ == "__main__":
    main()
