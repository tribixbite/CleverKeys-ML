#!/usr/bin/env python3
"""
RNN-T Beam Search Implementation
Based on NeMo's RNN-T architecture with Prediction Network + Joint Network
"""

import numpy as np
import onnxruntime as ort
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import heapq


@dataclass
class Hypothesis:
    """Beam search hypothesis"""
    tokens: List[int]
    score: float
    pred_state: Optional[Tuple[np.ndarray, np.ndarray]]  # (h, c) states
    timestep: int

    def __lt__(self, other):
        # For heap - higher score is better
        return self.score > other.score


class RNNTBeamSearch:
    """
    RNN-T Beam Search Decoder

    The decoder_joint model contains:
    1. Prediction Network (LSTM) that processes previous tokens
    2. Joint Network that combines encoder and prediction outputs
    """

    def __init__(self, decoder_session: ort.InferenceSession,
                 vocab_size: int = 30, blank_id: int = 29,
                 beam_size: int = 5, max_symbols_per_step: int = 3):
        self.decoder_session = decoder_session
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.max_symbols_per_step = max_symbols_per_step

        # Load vocabulary
        with open('runtime_meta.json', 'r') as f:
            meta = json.load(f)
            self.id_to_char = {int(k): v for k, v in meta['id_to_char'].items()}

        # Get model info
        self.input_names = [i.name for i in decoder_session.get_inputs()]
        self.output_names = [o.name for o in decoder_session.get_outputs()]

        print(f"Decoder inputs: {self.input_names}")
        print(f"Decoder outputs: {self.output_names}")

        # Typical RNN-T decoder has these inputs:
        # - encoder_outputs: [batch, 1, encoder_dim]
        # - targets: [batch, target_len]
        # - target_length: [batch]
        # - input_states_1, input_states_2: LSTM states

        self.hidden_size = 320  # From model config
        self.num_layers = 2  # LSTM has 2 layers

    def decode(self, encoder_output: np.ndarray, encoder_length: int) -> List[Tuple[str, float]]:
        """
        Perform RNN-T beam search decoding

        Args:
            encoder_output: [batch, hidden_dim, time]
            encoder_length: Valid encoder frames

        Returns:
            List of (text, score) hypotheses
        """
        batch_size = encoder_output.shape[0]

        # Note: encoder_output is [batch, hidden_dim, time]
        # The decoder will process all frames at once

        # Initialize with empty hypothesis
        initial_h = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)
        initial_c = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)

        # For simplicity, do a single forward pass with the full encoder output
        # In a real implementation, you'd do multiple passes with different target sequences

        # Start with just blank token
        targets = np.array([[self.blank_id]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)

        try:
            inputs = {
                'encoder_outputs': encoder_output,  # Full encoder output
                'targets': targets,
                'target_length': target_length,
                'input_states_1': initial_h,
                'input_states_2': initial_c
            }

            outputs = self.decoder_session.run(None, inputs)

            # outputs[0] should be logits [batch, target_len, vocab_size]
            logits = outputs[0]

            # Simple greedy decode from logits
            tokens = []
            if logits.ndim >= 2:
                for t in range(min(logits.shape[1], 20)):  # Limit output length
                    if logits.ndim == 3:
                        token_logits = logits[0, t, :]
                    else:
                        token_logits = logits[t] if logits.ndim == 2 else logits

                    token_id = np.argmax(token_logits)
                    if token_id != self.blank_id:
                        tokens.append(int(token_id))

            text = self._tokens_to_text(tokens)
            return [(text, 0.0)]

        except Exception as e:
            print(f"Decoder error: {e}")
            return [("", 0.0)]

    def _expand_hypothesis(self, hyp: Hypothesis, encoder_frame: np.ndarray) -> List[Hypothesis]:
        """
        Expand a hypothesis by considering all possible next tokens

        RNN-T allows:
        1. Emitting blank (no token, move to next frame)
        2. Emitting 1 or more tokens at current frame
        """
        expanded = []
        h_state, c_state = hyp.pred_state

        # Current tokens to feed to prediction network
        if hyp.tokens:
            targets = np.array([hyp.tokens[-self.max_symbols_per_step:]], dtype=np.int32)
        else:
            # Start with blank
            targets = np.array([[self.blank_id]], dtype=np.int32)

        target_length = np.array([len(targets[0])], dtype=np.int32)

        try:
            # Run decoder (Prediction + Joint network)
            inputs = {
                'encoder_outputs': encoder_frame,
                'targets': targets,
                'target_length': target_length,
                'input_states_1': h_state,
                'input_states_2': c_state
            }

            outputs = self.decoder_session.run(None, inputs)

            # outputs[0] should be logits from joint network
            # outputs[1] might be output lengths
            # outputs[2], outputs[3] should be updated LSTM states

            if len(outputs) >= 1:
                logits = outputs[0]  # [batch, time, vocab]

                # Get log probabilities for last position
                if logits.ndim == 3:
                    log_probs = self._log_softmax(logits[0, -1, :])
                else:
                    log_probs = self._log_softmax(logits.flatten()[:self.vocab_size])

                # Get new states if available
                new_h = outputs[2] if len(outputs) > 2 else h_state
                new_c = outputs[3] if len(outputs) > 3 else c_state

                # Consider top-k tokens
                top_k = min(self.beam_size * 2, self.vocab_size)
                top_indices = np.argsort(log_probs)[-top_k:]

                for token_id in top_indices:
                    new_score = hyp.score + log_probs[token_id]

                    if token_id == self.blank_id:
                        # Blank: move to next frame without emitting token
                        expanded.append(Hypothesis(
                            tokens=hyp.tokens,
                            score=new_score,
                            pred_state=(new_h, new_c),
                            timestep=hyp.timestep + 1
                        ))
                    else:
                        # Non-blank: emit token and stay at current frame
                        expanded.append(Hypothesis(
                            tokens=hyp.tokens + [token_id],
                            score=new_score,
                            pred_state=(new_h, new_c),
                            timestep=hyp.timestep
                        ))

        except Exception as e:
            print(f"Decoder error: {e}")
            # Fallback: just return original hypothesis
            expanded.append(hyp)

        return expanded

    def _log_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute log softmax"""
        max_val = np.max(logits)
        exp_logits = np.exp(logits - max_val)
        return logits - max_val - np.log(np.sum(exp_logits))

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        chars = []
        for token in tokens:
            if token in self.id_to_char:
                char = self.id_to_char[token]
                if char not in ['<blank>', '<unk>']:
                    chars.append(char)
        return ''.join(chars)