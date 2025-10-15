#!/usr/bin/env python3
"""
Stateful ONNX export for RNN-T models.

This script properly exports the encoder and decoder as separate ONNX models,
with the decoder explicitly handling LSTM states as inputs/outputs for proper
RNN-T beam search in JavaScript.
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn as nn
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).parent.absolute()))

from train_transducer_personalized import (
    PersonalizedRNNTModel,
    CONFIG,
    build_model_config,
    load_vocab,
    PersonalizedSwipeFeaturizer,
)


class StatefulRNNTDecoder(nn.Module):
    """
    Wrapper for NeMo's RNN-T decoder to make LSTM states explicit.
    This enables proper stateful inference in ONNX Runtime.
    """
    def __init__(self, nemo_decoder):
        super().__init__()
        self.embedding = nemo_decoder.prediction.embed
        self.dec_rnn = nemo_decoder.prediction.dec_rnn
        self.lstm = self.dec_rnn.lstm  # The actual multi-layer LSTM
        self.dropout = nemo_decoder.prediction.dropout if hasattr(nemo_decoder.prediction, 'dropout') else None
        # Project/projection layer is optional
        self.projection = getattr(nemo_decoder.prediction, 'project', None) or getattr(nemo_decoder.prediction, 'projection', None)
        self.vocab_size = nemo_decoder.vocab_size if hasattr(nemo_decoder, 'vocab_size') else nemo_decoder.num_classes_with_blank
        self.blank_idx = nemo_decoder.blank_idx

        # Store LSTM configuration
        self.num_layers = self.lstm.num_layers
        self.hidden_size = self.lstm.hidden_size

    def forward(self, input_tokens, h_prev, c_prev):
        """
        Stateful forward pass with explicit LSTM state management.

        Args:
            input_tokens: [batch_size, 1] - Previous token IDs
            h_prev: [num_layers, batch_size, hidden_size] - Previous hidden state
            c_prev: [num_layers, batch_size, hidden_size] - Previous cell state

        Returns:
            decoder_output: [batch_size, 1, hidden_size] - Decoder features
            h_next: [num_layers, batch_size, hidden_size] - Next hidden state
            c_next: [num_layers, batch_size, hidden_size] - Next cell state
        """
        # Embed input tokens
        embedded = self.embedding(input_tokens)  # [batch, 1, embed_dim]

        # Process through multi-layer LSTM with state
        # The LSTM expects states as (h, c) where each has shape [num_layers, batch, hidden]
        output, (h_next, c_next) = self.lstm(embedded, (h_prev, c_prev))

        # Apply dropout if available
        if self.dropout is not None:
            output = self.dropout(output)

        # Project to decoder output space (not vocab space yet - joint does that)
        decoder_output = self.projection(output) if self.projection else output

        return decoder_output, h_next, c_next


class StatefulRNNTJoint(nn.Module):
    """
    RNN-T joint network that combines encoder and decoder outputs.
    """
    def __init__(self, nemo_joint):
        super().__init__()
        # The joint network has separate projections for encoder and decoder
        self.enc_proj = nemo_joint.enc
        self.pred_proj = nemo_joint.pred
        self.joint_net = nemo_joint.joint_net
        self.vocab_size = nemo_joint._num_classes

    def forward(self, encoder_output, decoder_output):
        """
        Combine encoder and decoder outputs to produce logits.

        Args:
            encoder_output: [batch_size, 1, encoder_dim]
            decoder_output: [batch_size, 1, decoder_dim]

        Returns:
            logits: [batch_size, 1, vocab_size]
        """
        # Project encoder and decoder to joint space
        enc_proj = self.enc_proj(encoder_output)
        pred_proj = self.pred_proj(decoder_output)

        # Combine and compute logits
        combined = enc_proj + pred_proj  # Element-wise addition in joint space
        joint_out = self.joint_net(combined)
        return joint_out


def _infer_feature_dim_from_state_dict(state_dict: dict) -> int | None:
    """Infer input feature dimension from checkpoint parameters.

    Prefer Linear pre-encode projection weight: [d_model, feat_in].
    Fallback to first conv input channels if present.
    """
    # 1) Linear pre-encode projection
    for key in list(state_dict.keys()):
        if key.endswith("pre_encode.out.weight"):
            w = state_dict[key]
            try:
                return int(w.shape[1])
            except Exception:
                pass
    # 2) Compiled variant may have _orig_mod in the path
    for key in list(state_dict.keys()):
        if key.endswith("_orig_mod.pre_encode.out.weight"):
            w = state_dict[key]
            try:
                return int(w.shape[1])
            except Exception:
                pass
    # 3) First conv input channels (less reliable, but a fallback)
    for suffix in ("pre_encode.conv.0.weight", "_orig_mod.pre_encode.conv.0.weight"):
        for key in list(state_dict.keys()):
            if key.endswith(suffix):
                w = state_dict[key]
                try:
                    return int(w.shape[1])
                except Exception:
                    pass
    return None


def export_stateful_onnx(model, output_dir, quantize_int8=False, verbose=False):
    """Export the model as separate stateful ONNX components."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Exporting stateful ONNX models...")

    # 1. Export Encoder (straightforward, no state needed)
    print("Exporting encoder...")
    encoder = model.encoder
    encoder.eval()

    # Example inputs for encoder
    batch_size = 1
    time_steps = 96
    # Derive feature dim from the instantiated model when possible
    feat_dim = getattr(model.encoder, 'feat_in', None) or getattr(model, 'feat_in', None)
    if feat_dim is None:
        feat_dim = PersonalizedSwipeFeaturizer().feature_dim

    dummy_audio = torch.randn(batch_size, feat_dim, time_steps)
    dummy_length = torch.tensor([time_steps], dtype=torch.long)

    # Wrap encoder to handle NeMo's kwargs requirement
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, audio_signal, length):
            return self.encoder(audio_signal=audio_signal, length=length)

    encoder_wrapper = EncoderWrapper(encoder)
    encoder_wrapper.eval()

    encoder_path = output_dir / "encoder.onnx"
    if verbose:
        print("Exporting encoder with verbose output...")
    torch.onnx.export(
        encoder_wrapper,
        (dummy_audio, dummy_length),
        str(encoder_path),
        input_names=["audio_signal", "length"],
        output_names=["encoded", "encoded_lengths"],
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "time"},
            "length": {0: "batch"},
            "encoded": {0: "batch", 1: "time"},
            "encoded_lengths": {0: "batch"}
        },
        opset_version=14,
        export_params=True,
        do_constant_folding=True,
        verbose=verbose
    )
    print(f"Encoder exported to {encoder_path}")

    # 2. Export Stateful Decoder
    print("Exporting stateful decoder...")
    stateful_decoder = StatefulRNNTDecoder(model.decoder)
    stateful_decoder.eval()

    # Get LSTM dimensions from the stateful decoder
    num_layers = stateful_decoder.num_layers
    hidden_size = stateful_decoder.hidden_size

    # Example inputs for decoder
    dummy_tokens = torch.zeros(batch_size, 1, dtype=torch.long)
    dummy_h = torch.zeros(num_layers, batch_size, hidden_size)
    dummy_c = torch.zeros(num_layers, batch_size, hidden_size)

    decoder_path = output_dir / "decoder.onnx"
    torch.onnx.export(
        stateful_decoder,
        (dummy_tokens, dummy_h, dummy_c),
        str(decoder_path),
        input_names=["input_tokens", "h_in", "c_in"],
        output_names=["decoder_output", "h_out", "c_out"],
        dynamic_axes={
            "input_tokens": {0: "batch"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "decoder_output": {0: "batch"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"}
        },
        opset_version=14,
        export_params=True,
        do_constant_folding=True
    )
    print(f"Decoder exported to {decoder_path}")

    # 3. Export Joint Network
    print("Exporting joint network...")
    stateful_joint = StatefulRNNTJoint(model.joint)
    stateful_joint.eval()

    # Example inputs for joint
    encoder_dim = model.encoder.d_model
    decoder_dim = model.decoder.pred_hidden if hasattr(model.decoder, 'pred_hidden') else hidden_size

    dummy_enc = torch.randn(batch_size, 1, encoder_dim)
    dummy_dec = torch.randn(batch_size, 1, decoder_dim)

    joint_path = output_dir / "joint.onnx"
    torch.onnx.export(
        stateful_joint,
        (dummy_enc, dummy_dec),
        str(joint_path),
        input_names=["encoder_output", "decoder_output"],
        output_names=["logits"],
        dynamic_axes={
            "encoder_output": {0: "batch"},
            "decoder_output": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=14,
        export_params=True,
        do_constant_folding=True
    )
    print(f"Joint network exported to {joint_path}")

    # 4. Apply INT8 quantization if requested
    if quantize_int8:
        print("Applying INT8 quantization...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        for model_path in [encoder_path, decoder_path, joint_path]:
            quantized_path = model_path.with_suffix('.int8.onnx')
            quantize_dynamic(
                str(model_path),
                str(quantized_path),
                weight_type=QuantType.QInt8
            )
            print(f"Quantized: {quantized_path}")

    print("Export complete!")


def generate_runtime_meta(model, vocab, output_dir):
    """Generate runtime metadata for JavaScript decoder."""
    print(f"Generating runtime_meta.json in {output_dir}...")

    # Get LSTM configuration
    lstm = model.decoder.prediction.dec_rnn.lstm
    num_layers = lstm.num_layers
    hidden_size = lstm.hidden_size

    meta = {
        "vocab_size": len(vocab) + 1,  # Include blank token
        "blank_id": model.decoder.blank_idx,
        "tokens": list(vocab.keys()) + [""],  # Add empty string for blank
        "char_to_id": {**vocab, "": model.decoder.blank_idx},
        "id_to_char": {v: k for k, v in vocab.items()},
        "decoder_config": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "encoder_dim": model.encoder.d_model,
            "decoder_dim": hidden_size
        }
    }

    # Fix id_to_char for blank token
    meta["id_to_char"][model.decoder.blank_idx] = ""

    with open(output_dir / "runtime_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    print("runtime_meta.json generated.")


def validate_exported_models(output_dir):
    """Validate that exported models can be loaded and run inference."""
    try:
        import onnxruntime as ort
        print("\nValidating exported models...")

        # Check encoder
        encoder_session = ort.InferenceSession(str(output_dir / "encoder.onnx"))
        print(f"✓ Encoder loaded successfully")
        print(f"  Inputs: {[i.name for i in encoder_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in encoder_session.get_outputs()]}")

        # Check decoder
        decoder_session = ort.InferenceSession(str(output_dir / "decoder.onnx"))
        print(f"✓ Decoder loaded successfully")
        print(f"  Inputs: {[i.name for i in decoder_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in decoder_session.get_outputs()]}")

        # Check joint
        joint_session = ort.InferenceSession(str(output_dir / "joint.onnx"))
        print(f"✓ Joint network loaded successfully")
        print(f"  Inputs: {[i.name for i in joint_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in joint_session.get_outputs()]}")

        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def _safe_load_checkpoint(path: str) -> dict:
    import torch, omegaconf
    with torch.serialization.safe_globals([omegaconf.dictconfig.DictConfig]):
        return torch.load(path, weights_only=False)


def _write_runtime_helper(output_dir: Path, model: PersonalizedRNNTModel, vocab: dict, feature_dim: int, feature_names: list[str]):
    md = []
    md.append("# CleverKeys RNNT Runtime Helper\n")
    md.append("\n## Dimensions\n")
    md.append(f"- feature_dim: {feature_dim}\n")
    md.append(f"- encoder d_model: {getattr(model.encoder, 'd_model', 'unknown')}\n")
    lstm = model.decoder.prediction.dec_rnn.lstm
    md.append(f"- decoder LSTM: layers={lstm.num_layers}, hidden_size={lstm.hidden_size}\n")
    md.append(f"- vocab_size (w/o blank): {len(vocab)}\n")
    md.append(f"- blank_id: {model.decoder.blank_idx}\n")
    md.append("\n## Feature Columns (order)\n")
    for i, name in enumerate(feature_names):
        md.append(f"- {i}: {name}\n")
    md.append("\n## Encoder ONNX I/O\n")
    md.append("- Inputs: `audio_signal` [batch, feature_dim, time], `length` [batch]\n")
    md.append("- Outputs: `encoded` [batch, time, encoder_dim], `encoded_lengths` [batch]\n")
    md.append("\n## Decoder ONNX (stateful)\n")
    md.append("- Inputs: `input_tokens` [batch, 1], `h_in` [layers, batch, hidden], `c_in` [layers, batch, hidden]\n")
    md.append("- Outputs: `decoder_output` [batch, 1, decoder_dim], `h_out`, `c_out`\n")
    md.append("\n## Joint ONNX\n")
    md.append("- Inputs: `encoder_output` [batch, 1, encoder_dim], `decoder_output` [batch, 1, decoder_dim]\n")
    md.append("- Output: `logits` [batch, 1, vocab_size_with_blank]\n")
    md.append("\n## Greedy Decoding Outline\n")
    md.append("1. Featurize points to [T, feature_dim], transpose to [1, feature_dim, T], pass through encoder.\n")
    md.append("2. Initialize decoder states (zeros) and token=blank (or BOS if used).\n")
    md.append("3. Loop: joint(encoder_t, decoder) -> argmax token; if blank, advance time; else emit token and feed back to decoder.\n")
    md.append("4. Stop after max_symbols or EOS.\n")
    (output_dir / "runtime_helper.md").write_text("".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stateful ONNX Export for RNN-T")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .ckpt file")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save exported files")
    parser.add_argument("--quantize", choices=['int8', 'none'], default='none', help="Quantization type")
    parser.add_argument("--validate", action="store_true", help="Validate exported models")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose export output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading checkpoint to infer shapes...")
    ckpt = _safe_load_checkpoint(args.checkpoint)
    state_dict = ckpt.get('state_dict', {})
    # Normalize keys if compiled
    norm_sd = {}
    for k,v in state_dict.items():
        nk = k.replace('encoder._orig_mod.', 'encoder.').replace('decoder._orig_mod.', 'decoder.').replace('joint._orig_mod.', 'joint.').replace('._orig_mod.', '.')
        norm_sd[nk] = v
    state_dict = norm_sd

    # Build config using inferred feature dim
    feat_dim = _infer_feature_dim_from_state_dict(state_dict) or PersonalizedSwipeFeaturizer().feature_dim
    print(f"Inferred feature_dim={feat_dim}")
    cfg = DictConfig(CONFIG)
    vocab = load_vocab(cfg.data.vocab_path)
    labels = list(vocab.keys())
    nemo_cfg = build_model_config(cfg, labels, int(feat_dim))
    model = PersonalizedRNNTModel(cfg=nemo_cfg)
    print(f"Loading weights into model")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Export models
    export_stateful_onnx(model, output_dir, quantize_int8=(args.quantize == 'int8'))

    # Generate runtime metadata
    generate_runtime_meta(model, vocab, output_dir)

    # Write runtime helper documentation
    feature_names = PersonalizedSwipeFeaturizer().FEATURE_NAMES
    _write_runtime_helper(output_dir, model, vocab, int(feat_dim), feature_names)

    # Validate if requested
    if args.validate:
        validate_exported_models(output_dir)

    print(f"\nExport complete! Files saved to {output_dir}")
    print("Files created:")
    print("  - encoder.onnx: Conformer encoder")
    print("  - decoder.onnx: Stateful LSTM decoder")
    print("  - joint.onnx: Joint network")
    print("  - runtime_meta.json: Vocabulary and model config")

    if args.quantize == 'int8':
        print("  - *.int8.onnx: Quantized versions")


if __name__ == "__main__":
    main()
