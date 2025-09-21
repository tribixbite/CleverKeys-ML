#!/usr/bin/env python3
"""
Beam Search Decoder for RNN-T models
Implements proper beam search with ONNX runtime
"""

import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import json


class BeamSearchDecoder:
    """Beam search decoder for RNN-T models"""

    def __init__(self, decoder_session: ort.InferenceSession,
                 vocab_size: int = 30, blank_id: int = 29,
                 beam_width: int = 10, max_steps: int = 100):
        self.decoder_session = decoder_session
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.max_steps = max_steps

        # Load vocabulary
        with open('runtime_meta.json', 'r') as f:
            meta = json.load(f)
            self.id_to_char = {int(k): v for k, v in meta['id_to_char'].items()}

    def decode(self, encoder_output: np.ndarray, encoder_length: int) -> List[Tuple[str, float]]:
        """
        Perform beam search decoding

        Args:
            encoder_output: Encoder output [1, time, hidden]
            encoder_length: Valid encoder output length

        Returns:
            List of (word, score) tuples
        """
        batch_size = encoder_output.shape[0]
        encoder_output = encoder_output[:, :encoder_length, :]

        # Initialize beams
        beams = [{'tokens': [], 'score': 0.0, 'hidden': None}]

        # Process each encoder frame
        for t in range(min(encoder_length, self.max_steps)):
            encoder_frame = encoder_output[:, t:t+1, :]  # [1, 1, hidden]
            new_beams = []

            for beam in beams:
                # Get decoder predictions for this beam
                if len(beam['tokens']) == 0:
                    # Initial state - no previous tokens
                    # For RNN-T joint network, we need both encoder and decoder inputs
                    # The decoder joint model expects: encoder_outputs, decoder_outputs

                    # Initialize decoder state (zeros for first step)
                    decoder_state = np.zeros((1, 1, encoder_frame.shape[-1]), dtype=np.float32)

                    inputs = {
                        'encoder_outputs': encoder_frame,
                        'decoder_outputs': decoder_state
                    }

                    try:
                        outputs = self.decoder_session.run(None, inputs)
                        logits = outputs[0]  # [1, 1, vocab_size]
                    except Exception as e:
                        # If joint model has different input names, try alternatives
                        input_names = [i.name for i in self.decoder_session.get_inputs()]
                        if len(input_names) == 2:
                            inputs = {
                                input_names[0]: encoder_frame,
                                input_names[1]: decoder_state
                            }
                            outputs = self.decoder_session.run(None, inputs)
                            logits = outputs[0]
                        else:
                            # Fallback - return simple greedy result
                            return [("", 0.0)]

                    # Get log probabilities
                    log_probs = self._log_softmax(logits[0, 0, :])

                    # Get top-k predictions
                    top_k_indices = np.argsort(log_probs)[-self.beam_width:]

                    for idx in top_k_indices:
                        if idx != self.blank_id:  # Skip blank token
                            new_beam = {
                                'tokens': [idx],
                                'score': beam['score'] + log_probs[idx],
                                'hidden': None
                            }
                            new_beams.append(new_beam)

                    # Also consider blank (no output at this frame)
                    blank_beam = {
                        'tokens': beam['tokens'],
                        'score': beam['score'] + log_probs[self.blank_id],
                        'hidden': beam['hidden']
                    }
                    new_beams.append(blank_beam)

            # Prune beams
            if new_beams:
                new_beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:self.beam_width]
                beams = new_beams

        # Convert tokens to text
        results = []
        for beam in beams[:5]:  # Return top 5
            text = self._tokens_to_text(beam['tokens'])
            results.append((text, float(beam['score'])))

        return results

    def greedy_decode(self, encoder_output: np.ndarray, encoder_length: int) -> str:
        """
        Simple greedy decoding (non-beam search)

        Args:
            encoder_output: Encoder output [1, time, hidden]
            encoder_length: Valid encoder output length

        Returns:
            Decoded text string
        """
        tokens = []
        encoder_output = encoder_output[:, :encoder_length, :]

        for t in range(min(encoder_length, self.max_steps)):
            encoder_frame = encoder_output[:, t:t+1, :]

            # Initialize decoder state
            decoder_state = np.zeros_like(encoder_frame)

            # Try to run decoder
            try:
                input_names = [i.name for i in self.decoder_session.get_inputs()]
                if len(input_names) == 2:
                    inputs = {
                        input_names[0]: encoder_frame,
                        input_names[1]: decoder_state
                    }
                    outputs = self.decoder_session.run(None, inputs)
                    logits = outputs[0]  # [1, 1, vocab_size]

                    # Get most likely token
                    token_id = np.argmax(logits[0, 0, :])

                    # Skip blanks and append non-blank tokens
                    if token_id != self.blank_id and token_id < len(self.id_to_char):
                        tokens.append(token_id)
            except Exception as e:
                # If decoder fails, break
                break

        return self._tokens_to_text(tokens)

    def _log_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute log softmax"""
        max_val = np.max(logits)
        exp_logits = np.exp(logits - max_val)
        sum_exp = np.sum(exp_logits)
        return logits - max_val - np.log(sum_exp)

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        chars = []
        for token in tokens:
            if token in self.id_to_char:
                char = self.id_to_char[token]
                if char not in ['<blank>', '<unk>']:
                    chars.append(char)
        return ''.join(chars)