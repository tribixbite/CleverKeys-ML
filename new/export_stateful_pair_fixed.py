#!/usr/bin/env python3
"""
Fixed stateful RNNT exporter with richer runtime metadata and IO hints.

Outputs:
  - encoder.onnx (stateless encoder)
  - decoder_joint.onnx (stateful predictor + joint)
  - runtime_meta.json (vocab, blank_id, predictor mapping, decoder config, io dtypes)

Improvements vs export_stateful_pair.py:
  - Embeds predictor label mapping (joint<->predictor) under meta['predictor']
  - Attaches export_info (checkpoint path, versions, utc time)
  - Inspects decoder_joint.onnx inputs/outputs and records names/dtypes
    (targets/target_length dtype and state dtypes) to avoid inference mismatches
  - Ensures tokens cover blank_id index, padding if needed. BOS for predictor is
    index of '<blank>' token when present, else 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import os
import re
import datetime as dt

import onnxruntime as ort
import nemo
import torch

from trained_models.nema1.export_common import load_trained_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('export_stateful_pair_fixed')


def _write_runtime_meta(model, out_path: Path, decoder_joint_path: Path, source_checkpoint: str | None):
    # Resolve tokens and blank
    tokens = None
    blank_id = None
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
    if tokens is None:
        raise RuntimeError('Unable to derive tokens for runtime meta')

    try:
        blank = getattr(model.decoder, 'blank_idx', None)
        if blank is not None:
            blank_id = int(blank)
    except Exception:
        pass
    if blank_id is None:
        blank_id = len(tokens) - 1
    # Ensure tokens array covers blank_id index
    if blank_id >= len(tokens):
        log.warning('blank_id %d >= tokens length %d; padding tokens with empty string', blank_id, len(tokens))
        tokens = list(tokens) + ['']

    # Build predictor mapping (blankless predictor space)
    joint_vocab_size = len(tokens)
    joint2pred = []
    pred2joint = []
    for i in range(joint_vocab_size):
        if i == blank_id:
            joint2pred.append(-1)
        else:
            pid = i if i < blank_id else i - 1
            joint2pred.append(pid)
    for pid in range(joint_vocab_size - 1):
        jid = pid if pid < blank_id else pid + 1
        pred2joint.append(jid)

    # Decoder config
    num_layers = None            # predictor LSTM layers
    hidden_size = None           # predictor hidden size
    encoder_dim = None           # encoder model d_model
    encoder_n_layers = None      # encoder Conformer n_layers
    try:
        dc = getattr(model, 'cfg', None)
        if dc is not None and hasattr(dc, 'decoder'):
            prednet = getattr(dc.decoder, 'prednet', None)
            if prednet is not None:
                num_layers = int(getattr(prednet, 'pred_rnn_layers', 1))
                hidden_size = int(getattr(prednet, 'pred_hidden', 256))
        if dc is not None and hasattr(dc, 'encoder'):
            encoder_dim = int(getattr(dc.encoder, 'd_model', 256))
            # Some configs use 'n_layers' for Conformer layer count
            try:
                encoder_n_layers = int(getattr(dc.encoder, 'n_layers'))
            except Exception:
                encoder_n_layers = None
    except Exception:
        pass

    # Inspect decoder_joint IO names/dtypes
    io = {}
    try:
        sess = ort.InferenceSession(str(decoder_joint_path), providers=['CPUExecutionProvider'])
        inputs = [{ 'name': i.name, 'type': i.type, 'shape': [str(x) for x in i.shape] } for i in sess.get_inputs()]
        outputs = [{ 'name': o.name, 'type': o.type, 'shape': [str(x) for x in o.shape] } for o in sess.get_outputs()]
        io = {'inputs': inputs, 'outputs': outputs}
    except Exception as e:
        log.warning('Could not introspect decoder_joint IO: %s', e)

    # Predictor BOS in predictor space: index of '<blank>' token if present else 0
    bos_id = 0
    if isinstance(tokens, list) and '<blank>' in tokens:
        bos_id = int(tokens.index('<blank>'))

    meta = {
        'vocab_size': len(tokens),
        'blank_id': blank_id,  # RNNT joint blank
        'tokens': tokens,
        'char_to_id': {tok: i for i, tok in enumerate(tokens)},
        'id_to_char': {str(i): tok for i, tok in enumerate(tokens)},
        'predictor': {
            'uses_blankless_labels': True,
            'joint_blank_id': blank_id,
            'label_map': {
                'joint2pred': joint2pred,
                'pred2joint': pred2joint,
            },
            'bos_id': bos_id,
        },
        'io': io,
        'export_info': {
            'export_time_utc': dt.datetime.utcnow().isoformat() + 'Z',
            'source_checkpoint': source_checkpoint or '',
            'nemo_version': getattr(nemo, '__version__', ''),
            'torch_version': getattr(torch, '__version__', ''),
            'exporter': 'new/export_stateful_pair_fixed.py',
        },
    }
    if num_layers is not None and hidden_size is not None and encoder_dim is not None:
        meta['decoder_config'] = {
            'num_layers': num_layers,       # predictor LSTM layers
            'hidden_size': hidden_size,
            'encoder_dim': encoder_dim,
        }
    if encoder_n_layers is not None:
        meta['encoder_config'] = {
            'n_layers': encoder_n_layers,
            'd_model': encoder_dim,
        }
    out_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    log.info('✓ Wrote %s', out_path)


def _find_highest_epoch_checkpoint(search_base: str) -> str:
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
                path = os.path.join(root, f)
                st = os.stat(path)
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
                    newest_path = path
    if best_ckpt:
        log.info("Auto-discovered highest-epoch checkpoint: %s (epoch=%d)", best_ckpt, best_epoch)
        return best_ckpt
    if newest_path:
        log.info("Auto-discovered newest archive: %s", newest_path)
        return newest_path
    raise FileNotFoundError(f"No .ckpt or .nemo found under {search_base}")


def main():
    ap = argparse.ArgumentParser(description='Export fixed encoder.onnx and decoder_joint.onnx with rich meta')
    ap.add_argument('--checkpoint', help='Path to .nemo or .ckpt (auto-discover if omitted)')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--force-cpu', action='store_true', help='Force CPU export')
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_cpu:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    ckpt_path = args.checkpoint
    if not ckpt_path:
        base = os.environ.get('CKS_RUN_BASE', '9292025script')
        ckpt_path = _find_highest_epoch_checkpoint(base)

    model = load_trained_model(ckpt_path)
    tmp = out_dir / 'model.onnx'
    log.info('Calling NeMo model.export -> %s', tmp)
    model.export(str(tmp))

    # Discover exported parts
    cand_enc = [out_dir / 'encoder-model.onnx', out_dir / 'encoder.onnx']
    cand_dec = [out_dir / 'decoder_joint-model.onnx', out_dir / 'decoder_joint.onnx']
    enc_path = next((p for p in cand_enc if p.exists()), None)
    dec_path = next((p for p in cand_dec if p.exists()), None)
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

    # Write runtime meta with IO hints and predictor mapping
    _write_runtime_meta(model, out_dir / 'runtime_meta.json', dec_path, source_checkpoint=ckpt_path)

    # Cleanup tmp
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == '__main__':
    main()
