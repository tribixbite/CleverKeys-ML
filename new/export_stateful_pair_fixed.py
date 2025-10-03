#!/usr/bin/env python3
"""
Fixed CleverKeys RNNT Web Exporter for PersonalizedRNNTModel

Produces the ONNX models and runtime metadata needed for the web demo:
- encoder.onnx (Conformer encoder)
- decoder_joint.onnx (RNNT predictor + joint)
- runtime_meta.json (tokens, blank_id, mappings)

USAGE:
python new/export_stateful_pair_fixed.py \
  --checkpoint 9292025script/.../epoch=...ckpt \
  --outdir web-demo/models/oct3_export
"""

import argparse
import json
from pathlib import Path
import logging
import os
import re
import datetime as dt
import torch
import sys

# Add path for PersonalizedRNNTModel
sys.path.insert(0, os.path.dirname(__file__))
from train_transducer_personalized import PersonalizedRNNTModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('export_stateful_pair')


def load_trained_model(checkpoint_path):
    """Load PersonalizedRNNTModel from checkpoint"""
    log.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model from checkpoint config
    model = PersonalizedRNNTModel(checkpoint['hyper_parameters']['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Store config for metadata extraction
    model.checkpoint_cfg = checkpoint['hyper_parameters']['cfg']

    return model


def write_runtime_meta(model, out_path: Path, source_checkpoint: str | None = None):
    """Extract and write runtime metadata"""

    # Get vocabulary from checkpoint config
    tokens = model.checkpoint_cfg['labels']

    # NeMo's blank_as_pad=True puts blank at position 0, but functional blank is at end
    # The actual blank token for RNN-T decoding is at the end
    blank_id = len(tokens)  # Functional blank is after all vocab tokens

    # Add blank token to tokens list for display
    tokens_with_blank = list(tokens) + ['<blank>']

    # Build mappings
    char_to_id = {tok: i for i, tok in enumerate(tokens)}
    char_to_id['<blank>'] = blank_id

    id_to_char = {str(i): tok for i, tok in enumerate(tokens)}
    id_to_char[str(blank_id)] = '<blank>'

    # Get decoder config
    decoder_cfg = model.checkpoint_cfg.get('decoder', {})
    pred_cfg = decoder_cfg.get('prednet', {})

    num_layers = pred_cfg.get('pred_rnn_layers', 2)
    hidden_size = pred_cfg.get('pred_hidden', 320)

    # Get encoder dimension
    encoder_cfg = model.checkpoint_cfg.get('encoder', {})
    encoder_dim = encoder_cfg.get('d_model', 256)

    meta = {
        'vocab_size': len(tokens),
        'blank_id': blank_id,
        'tokens': tokens_with_blank,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'decoder_config': {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'encoder_dim': encoder_dim,
        },
        'export_info': {
            'export_time_utc': dt.datetime.utcnow().isoformat() + 'Z',
            'source_checkpoint': source_checkpoint or '',
            'exporter': 'new/export_stateful_pair_fixed.py',
        }
    }

    out_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    log.info('✓ Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description='Export encoder.onnx and decoder_joint.onnx')
    ap.add_argument('--checkpoint', required=True, help='Path to .ckpt')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--force-cpu', action='store_true', help='Force CPU export')
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Load model
    model = load_trained_model(args.checkpoint)

    # Export using NeMo's built-in export
    tmp = out_dir / 'model.onnx'
    log.info('Calling NeMo model.export -> %s', tmp)

    try:
        # Export the model
        model.export(str(tmp))

        # NeMo creates encoder-model.onnx and decoder_joint-model.onnx
        enc_src = out_dir / 'encoder-model.onnx'
        dec_src = out_dir / 'decoder_joint-model.onnx'

        # Rename to expected names
        if enc_src.exists():
            enc_dst = out_dir / 'encoder.onnx'
            enc_dst.unlink(missing_ok=True)
            enc_src.rename(enc_dst)
            log.info('✓ Exported: %s', enc_dst)
        else:
            log.error('encoder-model.onnx not found after export')

        if dec_src.exists():
            dec_dst = out_dir / 'decoder_joint.onnx'
            dec_dst.unlink(missing_ok=True)
            dec_src.rename(dec_dst)
            log.info('✓ Exported: %s', dec_dst)
        else:
            log.error('decoder_joint-model.onnx not found after export')

    except Exception as e:
        log.error(f"Export failed: {e}")
        raise

    # Write runtime metadata
    write_runtime_meta(model, out_dir / 'runtime_meta.json', source_checkpoint=args.checkpoint)

    # Cleanup
    tmp.unlink(missing_ok=True)

    log.info(f"\n✓ Export complete! Models saved to {out_dir}")
    log.info("\nTo test the exported models:")
    log.info(f"  cd web-demo/test")
    log.info(f"  python test_stateful_onnx.py")


if __name__ == '__main__':
    main()