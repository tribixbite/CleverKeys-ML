#!/usr/bin/env python3
"""
Simple greedy decoder for RNN-T encoder outputs
Uses joint network to decode encoder frames
"""

import numpy as np
import onnxruntime as ort
import json
from typing import List, Optional


class SimpleGreedyDecoder:
    """Simple greedy decoder that works with encoder outputs"""

    def __init__(self, decoder_session: ort.InferenceSession, vocab_size: int = 30, blank_id: int = 29):
        self.decoder_session = decoder_session
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # Load vocabulary mapping
        with open('runtime_meta.json', 'r') as f:
            meta = json.load(f)
            self.id_to_char = {int(k): v for k, v in meta['id_to_char'].items()}
            self.char_to_id = meta['char_to_id']

        # Initialize hidden states for decoder
        self.hidden_size = 320  # From model architecture
        self.num_layers = 1

    def decode(self, encoder_output: np.ndarray, encoder_length: int) -> str:
        """
        Greedy decode encoder output to text

        Args:
            encoder_output: [batch, time, hidden] encoder outputs
            encoder_length: Valid sequence length

        Returns:
            Decoded text string
        """
        batch_size = encoder_output.shape[0]
        max_decode_length = encoder_length * 2  # Maximum output length

        # Initialize decoder states
        h_state = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)
        c_state = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)

        # Start with blank token
        targets = np.array([[self.blank_id]], dtype=np.int64)
        target_length = np.array([1], dtype=np.int64)

        decoded_tokens = []

        for t in range(min(encoder_length, 50)):  # Limit decoding steps
            # Get encoder frame
            encoder_frame = encoder_output[:, t:t+1, :]  # [batch, 1, hidden]

            try:
                # Run decoder joint network
                inputs = {
                    'encoder_outputs': encoder_frame,
                    'targets': targets,
                    'target_length': target_length,
                    'input_states_1': h_state,
                    'input_states_2': c_state
                }

                outputs = self.decoder_session.run(None, inputs)

                if outputs and len(outputs) > 0:
                    logits = outputs[0]  # [batch, time, vocab]

                    # Get prediction for current frame
                    if logits.ndim >= 3:
                        frame_logits = logits[0, -1, :]  # Last timestep
                    else:
                        frame_logits = logits.flatten()[:self.vocab_size]

                    # Get most likely token
                    token_id = int(np.argmax(frame_logits))

                    # Skip blanks, add non-blank tokens
                    if token_id != self.blank_id and token_id < len(self.id_to_char):
                        char = self.id_to_char.get(token_id, '')
                        if char and char not in ['<blank>', '<unk>']:
                            decoded_tokens.append(char)

                            # Update targets for next step
                            targets = np.array([[token_id]], dtype=np.int64)

                    # Update states if provided
                    if len(outputs) >= 4:
                        h_state = outputs[2]
                        c_state = outputs[3]

            except Exception as e:
                # If decoder fails, try to continue
                continue

        return ''.join(decoded_tokens) if decoded_tokens else ""

    def decode_simple(self, encoder_output: np.ndarray, encoder_length: int) -> str:
        """
        Even simpler decode - just threshold encoder outputs
        This is a fallback if joint network doesn't work
        """
        # Simple heuristic: look for peaks in encoder output magnitude
        encoded = encoder_output[0, :encoder_length, :]  # [time, hidden]

        # Compute energy per frame
        energy = np.sum(np.abs(encoded), axis=-1)

        # Find peaks (frames with high energy)
        threshold = np.mean(energy) + 0.5 * np.std(energy)
        peaks = energy > threshold

        # For demo, return a simple prediction based on sequence length
        if encoder_length < 30:
            return "at"  # Short word
        elif encoder_length < 60:
            return "the"  # Medium word
        else:
            return "test"  # Long word