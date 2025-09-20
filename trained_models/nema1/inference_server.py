#!/usr/bin/env python3
"""
Simple inference server for RNNT model.
This can be called from TypeScript to get predictions.
"""

import torch
import numpy as np
import json
import sys
import nemo.collections.asr as nemo_asr
from pathlib import Path

# Load model once at startup
CHECKPOINT_PATH = "/home/will/git/swype/cleverkeys/rnnt_checkpoints_rare_words_20250919_140007/lightning_logs/version_0/checkpoints/epoch=epoch=80-wer=val_wer=0.152.ckpt"

print("Loading RNNT model...", file=sys.stderr)
model = nemo_asr.models.EncDecRNNTModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
else:
    model = model.cpu()
    device = 'cpu'

print(f"Model loaded on {device}", file=sys.stderr)

def predict(features):
    """Run inference on features."""
    # Convert to tensor
    features_tensor = torch.from_numpy(features).unsqueeze(0).transpose(1, 2).to(device)
    length_tensor = torch.tensor([features.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        # Encode
        encoded, encoded_len = model.encoder(
            audio_signal=features_tensor,
            length=length_tensor
        )

        # Decode with beam search
        hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len,
            return_hypotheses=True
        )

    results = []
    for hyp in hypotheses:
        results.append({
            'text': hyp.text if hasattr(hyp, 'text') else "",
            'score': float(hyp.score) if hasattr(hyp, 'score') else 0.0,
            'tokens': hyp.y_sequence.tolist() if hasattr(hyp, 'y_sequence') else []
        })

    return results

def main():
    """Process inference requests from stdin."""
    for line in sys.stdin:
        try:
            data = json.loads(line)
            features = np.array(data['features'], dtype=np.float32)

            # Get predictions
            results = predict(features)

            # Output as JSON
            print(json.dumps({
                'predictions': results,
                'status': 'success'
            }))
            sys.stdout.flush()

        except Exception as e:
            print(json.dumps({
                'error': str(e),
                'status': 'error'
            }))
            sys.stdout.flush()

if __name__ == "__main__":
    main()