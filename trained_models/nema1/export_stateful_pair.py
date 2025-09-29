#!/usr/bin/env python3
"""
Export encoder.onnx and decoder_joint.onnx from a trained RNNT (.nemo or .ckpt).

Also writes runtime_meta.json with vocab and blank_id for web runtime.
"""
import argparse
import json
from pathlib import Path
import logging

from export_common import load_trained_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('export_stateful_pair')


def write_runtime_meta(model, out_path: Path):
    # Attempt to fetch tokens from joint.vocabulary; fallback to cfg.labels
    tokens = None
    blank_id = None
    # Use the model's vocabulary order exactly if available
    try:
        vocab = getattr(model.joint, 'vocabulary', None)
        if vocab is not None:
            tokens = list(vocab)
    except Exception:
        pass
    if tokens is None:
        try:
            cfg_labels = getattr(model, 'cfg', None)
            if cfg_labels is not None and hasattr(cfg_labels, 'labels'):
                tokens = list(cfg_labels.labels)
        except Exception:
            pass
    try:
        blank = getattr(model.decoder, 'blank_idx', None)
        if blank is not None:
            blank_id = int(blank)
    except Exception:
        pass
    if tokens is None:
        try:
            tokens = list(model.cfg.labels)
        except Exception:
            pass
    if tokens is None:
        raise RuntimeError('Unable to derive tokens for runtime_meta')
    if blank_id is None:
        blank_id = len(tokens) - 1

    # Build mappings
    char_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_char = {str(i): tok for i, tok in enumerate(tokens)}
    meta = {
        'vocab_size': len(tokens),
        'blank_id': blank_id,
        'tokens': tokens,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
    }
    out_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    log.info('✓ Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description='Export encoder.onnx and decoder_joint.onnx')
    ap.add_argument('--checkpoint', required=True, help='Path to .nemo or .ckpt')
    ap.add_argument('--outdir', required=True, help='Output directory')
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(args.checkpoint)
    tmp = out_dir / 'model.onnx'
    log.info('Calling NeMo model.export -> %s', tmp)
    model.export(str(tmp))

    # NeMo writes split artifacts alongside the requested path
    cand_enc = [out_dir / 'encoder-model.onnx', out_dir / 'encoder.onnx']
    cand_dec = [out_dir / 'decoder_joint-model.onnx', out_dir / 'decoder_joint.onnx']

    enc_path = None
    dec_path = None

    for p in cand_enc:
        if p.exists():
            enc_path = p
            break
    for p in cand_dec:
        if p.exists():
            dec_path = p
            break

    if enc_path is None or dec_path is None:
        raise RuntimeError('Expected encoder/decoder_joint ONNX not found after export')

    # Normalize names
    if enc_path.name != 'encoder.onnx':
        (out_dir / 'encoder.onnx').unlink(missing_ok=True)
        enc_path.rename(out_dir / 'encoder.onnx')
        enc_path = out_dir / 'encoder.onnx'
    if dec_path.name != 'decoder_joint.onnx':
        (out_dir / 'decoder_joint.onnx').unlink(missing_ok=True)
        dec_path.rename(out_dir / 'decoder_joint.onnx')
        dec_path = out_dir / 'decoder_joint.onnx'

    log.info('✓ Exported: %s', enc_path)
    log.info('✓ Exported: %s', dec_path)

    # Runtime meta
    write_runtime_meta(model, out_dir / 'runtime_meta.json')

    # Cleanup tmp
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

if __name__ == '__main__':
    main()
