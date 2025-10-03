#!/usr/bin/env python3
"""
Export from 9292025script checkpoint using NeMo directly
"""

import json
import os
from pathlib import Path
import datetime as dt
import torch
import nemo
import nemo.collections.asr as nemo_asr

def write_runtime_meta(model, out_path, source_checkpoint=None):
    """Write runtime metadata from model"""
    # Get vocabulary from model
    tokens = list(model.joint.vocabulary)
    blank_id = int(model.decoder.blank_idx)

    # NeMo RNN-T uses functional blank at the end, beyond the vocabulary
    # The vocab has a placeholder <blank> at position 0, but the real blank is at blank_id
    # We need to extend the tokens array to include the functional blank
    while len(tokens) <= blank_id:
        tokens.append('')

    # Ensure the blank position has the right token
    if blank_id < len(tokens):
        tokens[blank_id] = ''  # Functional blank is empty string

    # Build mappings
    char_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_char = {str(i): tok for i, tok in enumerate(tokens)}

    # Get decoder/encoder config
    num_layers = int(model.cfg.decoder.prednet.pred_rnn_layers)
    hidden_size = int(model.cfg.decoder.prednet.pred_hidden)
    encoder_dim = int(model.cfg.encoder.d_model)

    # Build predictor label mapping
    joint_vocab_size = len(tokens)
    joint_blank_id = int(blank_id)
    joint2pred = []
    pred2joint = []

    for i in range(joint_vocab_size):
        if i == joint_blank_id:
            joint2pred.append(-1)
        else:
            pid = i if i < joint_blank_id else i - 1
            joint2pred.append(pid)

    for pid in range(joint_vocab_size - 1):
        jid = pid if pid < joint_blank_id else pid + 1
        pred2joint.append(jid)

    meta = {
        'vocab_size': len(tokens),
        'blank_id': joint_blank_id,
        'tokens': tokens,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'predictor': {
            'uses_blankless_labels': True,
            'joint_blank_id': joint_blank_id,
            'label_map': {
                'joint2pred': joint2pred,
                'pred2joint': pred2joint,
            },
            'bos_id': 0,
        },
        'decoder_config': {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'encoder_dim': encoder_dim,
        },
        'export_info': {
            'export_time_utc': dt.datetime.utcnow().isoformat() + 'Z',
            'source_checkpoint': source_checkpoint or '',
            'nemo_version': nemo.__version__,
            'torch_version': torch.__version__,
            'exporter': 'export_9292025.py',
        }
    }

    out_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'✓ Wrote {out_path}')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

    checkpoint_path = '/home/will/git/swype/cleverkeys/9292025script/20251002/rnnt_checkpoints_medium_balanced_20251003_013434/lightning_logs/version_0/checkpoints/epoch=epoch=71-wer=val_wer=0.457.ckpt'
    outdir = Path('web-demo/models/correct_9292025')
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load the NeMo model
    model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model = model.eval()

    # Export to ONNX
    tmp = outdir / 'model.onnx'
    print(f'Calling NeMo model.export -> {tmp}')
    model.export(str(tmp))

    # NeMo writes split artifacts alongside the requested path
    cand_enc = [outdir / 'encoder-model.onnx', outdir / 'encoder.onnx']
    cand_dec = [outdir / 'decoder_joint-model.onnx', outdir / 'decoder_joint.onnx']

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
        (outdir / 'encoder.onnx').unlink(missing_ok=True)
        enc_path.rename(outdir / 'encoder.onnx')
        enc_path = outdir / 'encoder.onnx'
    if dec_path.name != 'decoder_joint.onnx':
        (outdir / 'decoder_joint.onnx').unlink(missing_ok=True)
        dec_path.rename(outdir / 'decoder_joint.onnx')
        dec_path = outdir / 'decoder_joint.onnx'

    print(f'✓ Exported: {enc_path}')
    print(f'✓ Exported: {dec_path}')

    # Write runtime meta
    write_runtime_meta(model, outdir / 'runtime_meta.json', source_checkpoint=checkpoint_path)

    # Cleanup tmp
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n✅ Export complete!")


if __name__ == '__main__':
    main()