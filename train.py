#!/usr/bin/env python3
"""
State-of-the-Art Swipe Gesture Recognition Model
Advanced architecture with Conformer blocks, sophisticated data augmentation,
and production-ready training pipeline.
"""

import json
import math
import os
import random
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pyctcdecode
from huggingface_hub import hf_hub_download

# Advanced Configuration
@dataclass
class SOTAConfig:
    # Data paths
    train_data_path: str = "data/train_final_train.jsonl"
    val_data_path: str = "data/train_final_val.jsonl"
    vocab_path: str = "vocab/final_vocab.txt"
    
    # Model architecture
    model_type: str = "conformer"  # "conformer" or "transformer"
    input_dim: int = 35  # Will be calculated
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 1536
    dropout: float = 0.15
    
    # Conformer specific
    conv_kernel_size: int = 31
    use_macaron_style: bool = True
    
    # Training parameters
    batch_size: int = 128  # Reduced for memory
    learning_rate: float = 2e-4
    warmup_steps: int = 2000
    num_epochs: int = 100
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    label_smoothing: float = 0.1
    
    # Advanced features
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_spec_augment: bool = True
    use_curriculum: bool = True
    
    # Data augmentation
    aug_temporal_warp: bool = True
    aug_spatial_noise: float = 0.02
    aug_velocity_noise: float = 0.1
    aug_dropout_prob: float = 0.1
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4
    max_seq_length: int = 200
    
    # Decoding
    beam_width: int = 20
    use_lm: bool = True
    
    # Paths
    checkpoint_dir: str = "checkpoints_sota"
    log_dir: str = "logs_sota"
    export_dir: str = "exports_sota"

CONFIG = SOTAConfig()

# Create directories
for dir_path in [CONFIG.checkpoint_dir, CONFIG.log_dir, CONFIG.export_dir]:
    Path(dir_path).mkdir(exist_ok=True)

# --- Advanced Feature Engineering ---
class AdvancedSwipeFeaturizer:
    def __init__(self, keyboard_layout="qwerty"):
        self.grid = self._create_keyboard_grid(keyboard_layout)
        self.num_keys = len(self.grid.keys)
        
    def _create_keyboard_grid(self, layout):
        if layout == "qwerty":
            keys = [
                ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                ['z', 'x', 'c', 'v', 'b', 'n', 'm', "'"]
            ]
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        class Grid:
            def __init__(self):
                self.keys = []
                self.positions = {}
        
        grid = Grid()
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                x = col_idx / 9.0
                y = row_idx / 2.0
                grid.keys.append(key)
                grid.positions[key] = (x, y)
        
        grid.num_keys = len(grid.keys)
        return grid
    
    def extract_features(self, points: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract advanced features including curvature, angles, and jerk."""
        if len(points) < 2:
            return np.zeros((len(points), 35 + 10))  # Extra features
        
        points_array = np.array(points)
        
        # Ensure points_array is 2D
        if len(points_array.shape) == 1:
            # Reshape if flattened
            points_array = points_array.reshape(-1, 3)
        
        xs, ys, ts = points_array[:, 0], points_array[:, 1], points_array[:, 2]
        
        # Normalize coordinates
        xs = (xs - xs.mean()) / (xs.std() + 1e-8)
        ys = (ys - ys.mean()) / (ys.std() + 1e-8)
        
        # Time deltas with safety
        delta_times = np.diff(ts, prepend=ts[0])
        delta_times = np.where(delta_times == 0, 1e-3, delta_times)
        
        # Spatial deltas
        dx = np.diff(xs, prepend=xs[0])
        dy = np.diff(ys, prepend=ys[0])
        
        # Velocity
        vx = np.clip(dx / delta_times, -10.0, 10.0)
        vy = np.clip(dy / delta_times, -10.0, 10.0)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Acceleration
        ax = np.clip(np.diff(vx, prepend=vx[0]) / delta_times, -50.0, 50.0)
        ay = np.clip(np.diff(vy, prepend=vy[0]) / delta_times, -50.0, 50.0)
        
        # Jerk (rate of change of acceleration)
        jx = np.clip(np.diff(ax, prepend=ax[0]) / delta_times, -100.0, 100.0)
        jy = np.clip(np.diff(ay, prepend=ay[0]) / delta_times, -100.0, 100.0)
        
        # Angles and curvature
        angles = np.arctan2(dy, dx)
        angle_changes = np.diff(angles, prepend=angles[0])
        # Normalize angle changes to [-pi, pi]
        angle_changes = np.arctan2(np.sin(angle_changes), np.cos(angle_changes))
        curvature = angle_changes / (speed + 1e-8)
        curvature = np.clip(curvature, -10.0, 10.0)
        
        # Distance from start
        cumulative_dist = np.cumsum(np.sqrt(dx**2 + dy**2))
        normalized_dist = cumulative_dist / (cumulative_dist[-1] + 1e-8)
        
        # One-hot encoding for nearest keys
        nearest_keys = np.zeros((len(points), self.num_keys))
        for i, (x, y) in enumerate(zip(xs, ys)):
            # Denormalize for key finding
            x_denorm = x * xs.std() + xs.mean()
            y_denorm = y * ys.std() + ys.mean()
            nearest_idx = self._find_nearest_key(x_denorm, y_denorm)
            nearest_keys[i, nearest_idx] = 1.0
        
        # Combine all features
        features = np.stack([
            xs, ys, dx, dy, vx, vy, ax, ay,
            speed, jx, jy, angles, angle_changes, curvature, normalized_dist
        ], axis=1)
        
        # Add nearest key encoding
        features = np.concatenate([features, nearest_keys], axis=1)
        
        # Safety check
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return features.astype(np.float32)
    
    def _find_nearest_key(self, x: float, y: float) -> int:
        min_dist = float('inf')
        nearest_idx = 0
        for idx, (key_x, key_y) in enumerate(self.grid.positions.values()):
            dist = (x - key_x)**2 + (y - key_y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx

# --- Data Augmentation ---
class DataAugmentation:
    def __init__(self, config: SOTAConfig):
        self.config = config
    
    def temporal_warp(self, points: np.ndarray) -> np.ndarray:
        """Apply temporal warping to simulate speed variations."""
        if not self.config.aug_temporal_warp or len(points) < 4:
            return points
        
        # Check if points is 1D or 2D
        if len(points.shape) == 1:
            return points
            
        # Random speed changes
        speed_factor = np.random.uniform(0.8, 1.2, size=len(points))
        speed_factor = np.cumsum(speed_factor) / np.sum(speed_factor) * len(points)
        
        # Interpolate to maintain sequence length
        old_indices = np.arange(len(points))
        new_points = np.zeros_like(points)
        for dim in range(points.shape[1]):
            new_points[:, dim] = np.interp(old_indices, speed_factor, points[:, dim])
        
        return new_points
    
    def spatial_noise(self, points: np.ndarray) -> np.ndarray:
        """Add spatial noise to simulate finger position variations."""
        if self.config.aug_spatial_noise > 0:
            noise = np.random.normal(0, self.config.aug_spatial_noise, size=(len(points), 2))
            points[:, :2] += noise
        return points
    
    def dropout_points(self, points: np.ndarray) -> np.ndarray:
        """Randomly drop some points to simulate sampling variations."""
        if self.config.aug_dropout_prob > 0 and len(points) > 10:
            keep_prob = 1 - self.config.aug_dropout_prob
            mask = np.random.random(len(points)) < keep_prob
            # Always keep first and last points
            mask[0] = mask[-1] = True
            return points[mask]
        return points
    
    def augment(self, points: List[Tuple]) -> List[Tuple]:
        """Apply all augmentations."""
        if not hasattr(self, 'training') or not self.training:
            return points
        
        points_array = np.array(points)
        
        # Check if array is valid (2D with 3 columns)
        if len(points_array.shape) != 2 or points_array.shape[1] != 3:
            return points
        
        # Apply augmentations with probability
        if np.random.random() < 0.5:
            points_array = self.temporal_warp(points_array)
        if np.random.random() < 0.5:
            points_array = self.spatial_noise(points_array)
        if np.random.random() < 0.3:
            points_array = self.dropout_points(points_array)
        
        return [tuple(p) for p in points_array]

# --- Advanced Model Architectures ---

class ConformerBlock(nn.Module):
    """Conformer block combining convolution and self-attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward, conv_kernel_size, dropout=0.1):
        super().__init__()
        
        # First feed-forward module (macaron style)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Second feed-forward module
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # First feed-forward
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        attn_out = self.attn_norm(x)
        attn_out, _ = self.self_attn(attn_out, attn_out, attn_out, key_padding_mask=mask)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution module
        conv_out = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        conv_out = self.conv_module(conv_out)
        conv_out = conv_out.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = x + conv_out
        
        # Second feed-forward
        x = x + 0.5 * self.ff2(x)
        
        return self.layer_norm(x)

class SOTAGestureModel(nn.Module):
    """State-of-the-art model with Conformer architecture."""
    
    def __init__(self, config: SOTAConfig):
        super().__init__()
        self.config = config
        
        # Calculate input dimension
        featurizer = AdvancedSwipeFeaturizer()
        self.input_dim = 15 + featurizer.num_keys  # Advanced features
        
        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding with learnable parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, config.d_model) * 0.02)
        
        # Conformer or Transformer blocks
        if config.model_type == "conformer":
            self.blocks = nn.ModuleList([
                ConformerBlock(
                    config.d_model, config.nhead, config.dim_feedforward,
                    config.conv_kernel_size, config.dropout
                )
                for _ in range(config.num_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # CTC head with layer norm
        self.output_norm = nn.LayerNorm(config.d_model)
        self.ctc_head = nn.Linear(config.d_model, 28)  # 26 letters + apostrophe + blank
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Pass through blocks
        if self.config.model_type == "conformer":
            for block in self.blocks:
                x = block(x, mask)
        else:
            x = self.blocks(x, src_key_padding_mask=mask)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.ctc_head(x)
        
        # Return in (T, B, C) format for CTC loss
        return logits.transpose(0, 1)

# --- Advanced Dataset ---
class SOTASwipeDataset(Dataset):
    def __init__(self, jsonl_path, featurizer, tokenizer, config, augmentation=None, is_training=True):
        self.featurizer = featurizer
        self.tokenizer = tokenizer
        self.config = config
        self.augmentation = augmentation
        self.is_training = is_training
        self.data = []
        
        # Load and filter data
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                # Filter by sequence length
                if 10 <= len(sample['points']) <= config.max_seq_length:
                    # Convert to lowercase
                    sample['word'] = sample['word'].lower()
                    if all(c in tokenizer.vocab for c in sample['word']):
                        self.data.append(sample)
        
        # Curriculum learning: sort by difficulty
        if config.use_curriculum and is_training:
            self.data.sort(key=lambda x: len(x['word']) + len(x['points']) / 50)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        points = sample['points']
        word = sample['word']
        
        # Data augmentation
        if self.augmentation and self.is_training:
            self.augmentation.training = True
            points = self.augmentation.augment(points)
        
        # Extract features
        features = self.featurizer.extract_features(points)
        
        # Tokenize
        tokens = self.tokenizer.encode(word)
        
        return {
            'features': torch.FloatTensor(features),
            'tokens': torch.LongTensor(tokens),
            'word': word
        }

def collate_fn_sota(batch):
    """Custom collate with proper padding."""
    features = [item['features'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    words = [item['word'] for item in batch]
    
    # Pad features
    max_len = max(f.size(0) for f in features)
    padded_features = torch.zeros(len(batch), max_len, features[0].size(1))
    feature_lengths = []
    
    for i, f in enumerate(features):
        padded_features[i, :f.size(0)] = f
        feature_lengths.append(f.size(0))
    
    # Pad tokens
    max_token_len = max(len(t) for t in tokens)
    padded_tokens = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    token_lengths = []
    
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t
        token_lengths.append(len(t))
    
    # Create attention mask
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(feature_lengths):
        mask[i, length:] = True
    
    return {
        'features': padded_features,
        'feature_lengths': torch.LongTensor(feature_lengths),
        'tokens': padded_tokens,
        'token_lengths': torch.LongTensor(token_lengths),
        'mask': mask,
        'words': words
    }

# --- Training with Advanced Techniques ---
class SOTATrainer:
    def __init__(self, config: SOTAConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.featurizer = AdvancedSwipeFeaturizer()
        self.tokenizer = CharTokenizer()
        self.augmentation = DataAugmentation(config)
        
        # Create model
        self.model = SOTAGestureModel(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self.get_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config.mixed_precision)
        
        # Loss with label smoothing
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        self.global_step = 0
        self.best_loss = float('inf')
    
    def get_scheduler(self):
        """Cosine scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (
                self.config.num_epochs * 2500 - self.config.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            features = batch['features'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            token_lengths = batch['token_lengths'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # MixUp augmentation
            if self.config.use_mixup and np.random.random() < 0.5:
                lambda_mix = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
                batch_size = features.size(0)
                index = torch.randperm(batch_size).to(self.device)
                
                features = lambda_mix * features + (1 - lambda_mix) * features[index]
                feature_lengths = torch.maximum(feature_lengths, feature_lengths[index])
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                log_probs = self.model(features, mask)
                
                # Calculate loss
                log_probs = F.log_softmax(log_probs, dim=2)
                loss = self.criterion(log_probs, tokens, feature_lengths, token_lengths)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at step {self.global_step}")
                continue
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            self.global_step += 1
            
            if self.global_step % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'grad': f'{grad_norm:.2f}'
                })
                
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                self.writer.add_scalar('LR', current_lr, self.global_step)
                self.writer.add_scalar('GradNorm', grad_norm, self.global_step)
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                token_lengths = batch['token_lengths'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                with autocast(enabled=self.config.mixed_precision):
                    log_probs = self.model(features, mask)
                    log_probs = F.log_softmax(log_probs, dim=2)
                    loss = self.criterion(log_probs, tokens, feature_lengths, token_lengths)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self):
        # Create datasets
        train_dataset = SOTASwipeDataset(
            self.config.train_data_path, self.featurizer, self.tokenizer,
            self.config, self.augmentation, is_training=True
        )
        val_dataset = SOTASwipeDataset(
            self.config.val_data_path, self.featurizer, self.tokenizer,
            self.config, None, is_training=False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers,
            collate_fn=collate_fn_sota, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size * 2,
            shuffle=False, num_workers=self.config.num_workers,
            collate_fn=collate_fn_sota, pin_memory=True
        )
        
        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = Path(self.config.checkpoint_dir)
        if is_best:
            torch.save(checkpoint, path / 'best_model.pth')
            print(f"Saved best model with loss {self.best_loss:.4f}")
        else:
            torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}.pth')

class CharTokenizer:
    def __init__(self):
        self.vocab = {char: idx + 1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz'")}
        self.vocab['<blank>'] = 0
    
    def encode(self, text):
        return [self.vocab.get(char, 0) for char in text.lower()]

if __name__ == "__main__":
    print("Starting State-of-the-Art Training Pipeline")
    print("=" * 50)
    
    trainer = SOTATrainer(CONFIG)
    trainer.train()