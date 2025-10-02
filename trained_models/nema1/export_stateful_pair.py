#!/usr/bin/env python3
"""
README: CleverKeys RNNT Web Exporter (stateful pair)

This exporter produces the ONNX models and runtime metadata needed for the web demo:
- encoder.onnx (Conformer encoder)
- decoder_joint.onnx (RNNT predictor + joint)
- runtime_meta.json (tokens, blank_id, mappings, decoder_config)

USAGE
- Explicit checkpoint:
  python trained_models/nema1/export_stateful_pair.py \
    --checkpoint path/to/model.ckpt|model.nemo \
    --outdir web-demo/models/latest_pair --force-cpu

- Auto-discover (no --checkpoint provided):
  The script searches the training run base for the highest-epoch .ckpt
  and exports from that. The base defaults to $CKS_RUN_BASE or '9292025script'.
  Example:
  CKS_RUN_BASE=9292025script \
  python trained_models/nema1/export_stateful_pair.py --outdir web-demo/models/latest_pair --force-cpu

NOTES
- Do not hardcode token indices. Always consume blank_id/tokens from runtime_meta.json.
- If the serialized vocabulary length is shorter than blank_id+1 (observed in some archives),
  this exporter pads the tokens with an empty string to make blank_id valid. A warning is logged.
"""
import argparse
import json
from pathlib import Path
import logging

import os
import re
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
    # Ensure tokens array covers blank_id index (NeMo often uses functional blank at end)
    if blank_id >= len(tokens):
        log.warning(
            f"blank_id {blank_id} >= tokens length {len(tokens)}; padding tokens with empty string to align."
        )
        tokens = list(tokens) + ['']

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


def _find_highest_epoch_checkpoint(search_base: str) -> str:
    """Find the .ckpt with the highest epoch number under search_base.
    Falls back to the most recently modified .ckpt/.nemo if epochs not parsable.
    """
    epoch_re = re.compile(r"epoch=([0-9]+)")
    best_ckpt = None
    best_epoch = -1
    newest_mtime = -1.0
    newest_path = None
    for root, _, files in os.walk(search_base):
        for f in files:
            if f.endswith('.ckpt'):
                path = os.path.join(root, f)
                m = epoch_re.search(f)
                if m:
                    ep = int(m.group(1))
                    if ep > best_epoch:
                        best_epoch = ep
                        best_ckpt = path
                st = os.stat(path)
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
                    newest_path = path
            elif f.endswith('.nemo'):
                # track newest nemo in case no ckpt exists
                path = os.path.join(root, f)
                st = os.stat(path)
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
                    newest_path = path
    if best_ckpt:
        log.info(f"Auto-discovered highest-epoch checkpoint: {best_ckpt} (epoch={best_epoch})")
        return best_ckpt
    if newest_path:
        log.info(f"Auto-discovered newest archive: {newest_path}")
        return newest_path
    raise FileNotFoundError(f"No .ckpt or .nemo found under {search_base}")


def main():
    ap = argparse.ArgumentParser(description='Export encoder.onnx and decoder_joint.onnx')
    ap.add_argument('--checkpoint', help='Path to .nemo or .ckpt (auto-discover if omitted)')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--force-cpu', action='store_true', help='Force CPU export (ignore CUDA)')
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    ckpt_path = args.checkpoint
    if not ckpt_path:
        base = os.environ.get('CKS_RUN_BASE', '9292025script')
        ckpt_path = _find_highest_epoch_checkpoint(base)

    model = load_trained_model(ckpt_path)
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
