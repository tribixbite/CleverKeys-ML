#!/usr/bin/env python3
# export_rnnt_step.py
# Exports a single-step RNNT decoder+joint (fp32) to ONNX and ExecuTorch .pte.

import argparse
import logging
import torch
import types

from model_class import GestureRNNTModel, get_default_config

# ExecuTorch
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

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
    ap.add_argument("--nemo_model", required=True, help="Path to trained .nemo (your subclass).")
    ap.add_argument("--onnx_out", default="rnnt_step_fp32.onnx", help="Output ONNX path.")
    ap.add_argument("--pte_out",  default="rnnt_step_fp32.pte",  help="Output ExecuTorch .pte path.")
    ap.add_argument("--layers", type=int, default=None, help="Override L (num_layers) if needed.")
    ap.add_argument("--hidden", type=int, default=None, help="Override H (hidden_size) if needed.")
    ap.add_argument("--enc_dim", type=int, default=None, help="Override encoder D if needed.")
    ap.add_argument("--vocab", type=str, default=None, help="Path to vocabulary file for deriving blank_id.")
    args = ap.parse_args()

    # Derive blank_id from vocabulary if provided
    blank_id = 0  # default fallback
    if args.vocab:
        try:
            with open(args.vocab, "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f if line.strip()]
            token_to_idx = {tok: i for i, tok in enumerate(tokens)}
            blank_id = token_to_idx.get("<blank>", 0)
            log.info(f"Derived blank_id={blank_id} from vocab file: {args.vocab}")
        except Exception as e:
            log.warning(f"Failed to load vocab file {args.vocab}: {e}. Using default blank_id=0")

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
        log.info(f"Loading model: {args.nemo_model}")
        model = GestureRNNTModel.restore_from(args.nemo_model, map_location="cpu").eval()

    step = RNNTStep(model).eval()
    # Override the step's blank_idx with derived value
    step.blank_idx = blank_id

    # Example shapes
    B = 1
    L = args.layers or step.num_layers
    H = args.hidden or step.hidden_size
    # encoder proj dim D: derive from model.joint or from a dummy forward
    if args.enc_dim:
        D = args.enc_dim
    else:
        # Run one encoder frame through joint to infer expected D
        # Make a dummy pred vector H
        with torch.no_grad():
            pred_dummy = torch.randn(B, H)
            # Try a guess for D via a small binary search; fallback  step:
            # Safer: run encoder once:
            D = None
            try:
                # Use model.encoder with correct keyword arguments
                F = model.cfg.encoder.get("feat_in", 37)
                T = 10
                feats_bft = torch.randn(B, F, T)
                lens = torch.tensor([T], dtype=torch.int32)
                enc_bdt, _ = model.encoder(audio_signal=feats_bft, length=lens)   # (B,D,T')
                D = enc_bdt.shape[1]  # Encoder outputs [B, D, T], so D is at index 1
            except Exception:
                # Last resort, guess 512
                D = 512

    y_prev = torch.tensor([blank_id], dtype=torch.long)  # Use derived blank_id as start token
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

if __name__ == "__main__":
    main()
