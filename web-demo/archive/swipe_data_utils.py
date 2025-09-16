"""
Shared utilities for swipe data processing and feature engineering.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple


class KeyboardGrid:
    """Maps between keyboard coordinates and character positions."""
    def __init__(self, chars='abcdefghijklmnopqrstuvwxyz'):
        self.chars = chars
        # Standard QWERTY layout
        self.layout = [
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm'
        ]
        self.char_to_pos = {}
        for row_idx, row in enumerate(self.layout):
            for col_idx, char in enumerate(row):
                if char in chars:
                    # Normalize positions to [0, 1]
                    x = (col_idx + 0.5) / 10  # 10 chars max width
                    y = (row_idx + 0.5) / 3   # 3 rows
                    self.char_to_pos[char] = (x, y)
    
    def get_position(self, char):
        return self.char_to_pos.get(char, (0.5, 0.5))  # Center as default


class SwipeFeaturizer:
    """Converts swipe traces to feature vectors."""
    def __init__(self, grid: KeyboardGrid):
        self.grid = grid
    
    def __call__(self, points: List) -> np.ndarray:
        """Extract features from a trace of points."""
        if len(points) < 2:
            # Return minimal features for very short traces
            return np.zeros((1, 37))
        
        features = []
        for i in range(len(points)):
            feat = self._extract_point_features(points, i)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_point_features(self, points: List, idx: int) -> np.ndarray:
        """Extract features for a single point in the trace."""
        curr = points[idx]
        
        # Basic position features
        x, y = curr['x'], curr['y']
        t = curr.get('t', idx * 10) / 1000.0  # Normalize time to seconds
        
        # Velocity features
        if idx > 0:
            prev = points[idx - 1]
            dt = max((curr.get('t', idx * 10) - prev.get('t', (idx-1) * 10)) / 1000.0, 0.001)
            vx = (x - prev['x']) / dt
            vy = (y - prev['y']) / dt
            speed = np.sqrt(vx**2 + vy**2)
        else:
            vx = vy = speed = 0
        
        # Acceleration features
        if idx > 1:
            prev = points[idx - 1]
            prev2 = points[idx - 2]
            dt1 = max((curr.get('t', idx * 10) - prev.get('t', (idx-1) * 10)) / 1000.0, 0.001)
            dt2 = max((prev.get('t', (idx-1) * 10) - prev2.get('t', (idx-2) * 10)) / 1000.0, 0.001)
            
            vx_prev = (prev['x'] - prev2['x']) / dt2
            vy_prev = (prev['y'] - prev2['y']) / dt2
            
            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
            acc = np.sqrt(ax**2 + ay**2)
        else:
            ax = ay = acc = 0
        
        # Direction features
        if idx > 0:
            angle = np.arctan2(vy, vx)
            angle_sin = np.sin(angle)
            angle_cos = np.cos(angle)
        else:
            angle = angle_sin = angle_cos = 0
        
        # Curvature (change in angle)
        if idx > 1:
            prev = points[idx - 1]
            prev2 = points[idx - 2]
            angle_prev = np.arctan2(prev['y'] - prev2['y'], prev['x'] - prev2['x'])
            curvature = angle - angle_prev
            # Normalize to [-pi, pi]
            while curvature > np.pi:
                curvature -= 2 * np.pi
            while curvature < -np.pi:
                curvature += 2 * np.pi
        else:
            curvature = 0
        
        # Distance to nearest keys (top 5)
        key_distances = []
        for char, (kx, ky) in self.grid.char_to_pos.items():
            dist = np.sqrt((x - kx)**2 + (y - ky)**2)
            key_distances.append(dist)
        key_distances = sorted(key_distances)[:5]
        while len(key_distances) < 5:
            key_distances.append(1.0)  # Max distance as padding
        
        # Segment features
        total_points = len(points)
        progress = idx / max(total_points - 1, 1)  # Position in trace [0, 1]
        
        # Start/end indicators
        is_start = 1.0 if idx == 0 else 0.0
        is_end = 1.0 if idx == total_points - 1 else 0.0
        
        # Compile all features
        features = [
            x, y, t,                          # Position and time (3)
            vx, vy, speed,                    # Velocity (3)
            ax, ay, acc,                      # Acceleration (3)
            angle, angle_sin, angle_cos,      # Direction (3)
            curvature,                        # Curvature (1)
            *key_distances,                   # Nearest keys (5)
            progress,                         # Progress (1)
            is_start, is_end,                 # Indicators (2)
        ]
        
        # Add statistical features from window
        window_size = 5
        window_start = max(0, idx - window_size // 2)
        window_end = min(len(points), idx + window_size // 2 + 1)
        window_points = points[window_start:window_end]
        
        if len(window_points) > 1:
            window_x = [p['x'] for p in window_points]
            window_y = [p['y'] for p in window_points]
            
            features.extend([
                np.mean(window_x), np.std(window_x),  # X stats (2)
                np.mean(window_y), np.std(window_y),  # Y stats (2)
                np.max(window_x) - np.min(window_x),  # X range (1)
                np.max(window_y) - np.min(window_y),  # Y range (1)
            ])
        else:
            features.extend([x, 0, y, 0, 0, 0])  # Use current point with no variance
        
        # Add padding features to reach target dimension
        while len(features) < 37:
            features.append(0.0)
        
        return np.array(features[:37], dtype=np.float32)  # Ensure exactly 37 features


class SwipeDataset(Dataset):
    def __init__(self, manifest_path, featurizer, vocab, max_trace_len):
        super().__init__()
        self.featurizer = featurizer
        self.vocab = vocab
        self.max_trace_len = max_trace_len
        self.data = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Validate data: ensure 'points' and 'word' exist and are not empty
                if 'points' in item and 'word' in item and item['points'] and item['word']:
                    self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Process the gesture trace using the featurizer
        features = self.featurizer(item['points'])
        features_tensor = torch.from_numpy(features).float()
        
        # 2. Process the target word
        word = item['word']
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in word]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return (
            features_tensor,
            torch.tensor(features_tensor.shape[0], dtype=torch.long),
            tokens_tensor,
            torch.tensor(len(tokens), dtype=torch.long)
        )


def _stack_time(x, lengths, factor=2):
    """Frame stacking for effective sequence length reduction.

    Args:
        x: (B, T, F) tensor
        lengths: (B,) tensor of sequence lengths
        factor: stacking factor (default 2)

    Returns:
        x_stacked: (B, T//factor, F*factor) tensor
        lengths_stacked: (B,) tensor with adjusted lengths
    """
    B, T, F = x.shape
    T_trim = (T // factor) * factor
    x = x[:, :T_trim, :]  # Trim to multiple of factor
    x = x.view(B, T_trim // factor, F * factor)  # Stack adjacent frames
    lengths = torch.div(lengths, factor, rounding_mode='floor')
    return x, lengths


def collate_fn(batch, use_frame_stacking=False, stack_factor=2):
    """Pads traces and tokens to create uniform batches."""
    features, feature_lengths, tokens, token_lengths = zip(*batch)

    # Pad features to max length in batch
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Apply frame stacking if enabled (for small config)
    if use_frame_stacking:
        padded_features, feature_lengths = _stack_time(padded_features, torch.stack(feature_lengths), stack_factor)
    else:
        # Stack lengths normally
        feature_lengths = torch.stack(feature_lengths)

    # Pad tokens to max length in batch
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)

    # Stack token lengths
    token_lengths = torch.stack(token_lengths)

    return padded_features, feature_lengths, padded_tokens, token_lengths
