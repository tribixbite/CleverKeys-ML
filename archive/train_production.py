#!/usr/bin/env python3
"""
Production-Ready Swipe Gesture Recognition Training
Optimized for maximum performance with robust error handling
"""

import json
import math
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pyctcdecode
from huggingface_hub import hf_hub_download

# Production Configuration
CONFIG = {
    # Data
    "train_data": "data/train_final_train.jsonl",
    "val_data": "data/train_final_val.jsonl",
    "vocab_file": "vocab/final_vocab.txt",
    "max_seq_length": 200,
    
    # Model - Optimized Transformer
    "d_model": 384,
    "nhead": 8,
    "num_layers": 8,
    "dim_feedforward": 1536,
    "dropout": 0.1,
    "activation": "gelu",
    
    # Training
    "batch_size": 128,
    "learning_rate": 3e-4,
    "warmup_steps": 2000,
    "num_epochs": 50,
    "gradient_clip": 1.0,
    "weight_decay": 1e-5,
    
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": True,
    "num_workers": 4,
    "checkpoint_dir": "checkpoints_production",
    "log_dir": "logs_production",
    "seed": 42,
}

# Set random seeds
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# Create directories
Path(CONFIG["checkpoint_dir"]).mkdir(exist_ok=True)
Path(CONFIG["log_dir"]).mkdir(exist_ok=True)

class RobustFeaturizer:
    """Robust feature extraction with extensive validation"""
    
    def __init__(self):
        self.keyboard_layout = self._create_keyboard_layout()
        self.num_keys = len(self.keyboard_layout)
    
    def _create_keyboard_layout(self):
        """Create QWERTY keyboard layout"""
        keys = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', "'"]
        ]
        
        layout = {}
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                x = col_idx / 9.0
                y = row_idx / 2.0
                layout[key] = (x, y)
        return layout
    
    def extract_features(self, points: List[Tuple]) -> np.ndarray:
        """Extract robust features with validation"""
        # Validate input
        if not points or len(points) < 2:
            return np.zeros((2, 15 + self.num_keys), dtype=np.float32)
        
        # Convert to numpy array
        try:
            points_array = np.array(points, dtype=np.float32)
            if len(points_array.shape) != 2 or points_array.shape[1] != 3:
                # Try to reshape
                if points_array.size % 3 == 0:
                    points_array = points_array.reshape(-1, 3)
                else:
                    # Fallback: create dummy features
                    return np.zeros((len(points), 15 + self.num_keys), dtype=np.float32)
        except:
            return np.zeros((len(points), 15 + self.num_keys), dtype=np.float32)
        
        xs = points_array[:, 0]
        ys = points_array[:, 1] 
        ts = points_array[:, 2]
        
        # Normalize coordinates
        x_mean, x_std = xs.mean(), xs.std() + 1e-8
        y_mean, y_std = ys.mean(), ys.std() + 1e-8
        xs_norm = (xs - x_mean) / x_std
        ys_norm = (ys - y_mean) / y_std
        
        # Time deltas with safety
        dt = np.diff(ts, prepend=ts[0])
        dt = np.where(dt == 0, 1e-3, dt)
        dt = np.clip(dt, 1e-4, 1.0)
        
        # Spatial deltas
        dx = np.diff(xs_norm, prepend=xs_norm[0])
        dy = np.diff(ys_norm, prepend=ys_norm[0])
        
        # Velocity
        vx = np.clip(dx / dt, -20.0, 20.0)
        vy = np.clip(dy / dt, -20.0, 20.0)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Acceleration
        dvx = np.diff(vx, prepend=vx[0])
        dvy = np.diff(vy, prepend=vy[0])
        ax = np.clip(dvx / dt, -100.0, 100.0)
        ay = np.clip(dvy / dt, -100.0, 100.0)
        
        # Angles
        angles = np.arctan2(dy, dx)
        angle_changes = np.diff(angles, prepend=angles[0])
        # Normalize to [-pi, pi]
        angle_changes = np.arctan2(np.sin(angle_changes), np.cos(angle_changes))
        
        # Curvature
        curvature = np.clip(angle_changes / (speed + 1e-8), -10.0, 10.0)
        
        # Distance features
        cumulative_dist = np.cumsum(np.sqrt(dx**2 + dy**2))
        norm_dist = cumulative_dist / (cumulative_dist[-1] + 1e-8)
        
        # Nearest key encoding
        key_encoding = np.zeros((len(points_array), self.num_keys), dtype=np.float32)
        for i in range(len(points_array)):
            # Find nearest key
            min_dist = float('inf')
            nearest_idx = 0
            
            # Denormalize for key finding
            x_real = xs[i]
            y_real = ys[i]
            
            for key_idx, (key_x, key_y) in enumerate(self.keyboard_layout.values()):
                dist = (x_real - key_x)**2 + (y_real - key_y)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = key_idx
            
            key_encoding[i, nearest_idx] = 1.0
        
        # Combine features
        features = np.stack([
            xs_norm, ys_norm, dx, dy, vx, vy, ax, ay,
            speed, angles, angle_changes, curvature, norm_dist,
            np.log1p(dt), np.log1p(speed)
        ], axis=1)
        
        # Add key encoding
        features = np.concatenate([features, key_encoding], axis=1)
        
        # Final safety check
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return features.astype(np.float32)

class ImprovedTransformerModel(nn.Module):
    """Enhanced Transformer with better architecture"""
    
    def __init__(self, input_dim, config):
        super().__init__()
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config["d_model"]),
            nn.LayerNorm(config["d_model"]),
            nn.Dropout(config["dropout"])
        )
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, config["d_model"]) * 0.02
        )
        
        # Transformer encoder with improvements
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            activation=config["activation"],
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"],
            norm=nn.LayerNorm(config["d_model"])
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config["d_model"]),
            nn.Linear(config["d_model"], 28)  # 26 letters + apostrophe + blank
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Output projection
        logits = self.output_proj(x)
        
        # Return in (T, B, C) format for CTC
        return logits.transpose(0, 1)

class RobustDataset(Dataset):
    """Dataset with robust error handling"""
    
    def __init__(self, jsonl_path, featurizer, max_seq_length=200):
        self.featurizer = featurizer
        self.max_seq_length = max_seq_length
        self.data = []
        
        # Load and validate data
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    sample = json.loads(line)
                    points = sample.get('points', [])
                    word = sample.get('word', '').lower()
                    
                    # Validate
                    if (10 <= len(points) <= max_seq_length and 
                        word and all(c in "abcdefghijklmnopqrstuvwxyz'" for c in word)):
                        self.data.append({
                            'points': points,
                            'word': word
                        })
                except Exception as e:
                    print(f"Skipping line {line_num}: {e}")
        
        print(f"Loaded {len(self.data)} valid samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        features = self.featurizer.extract_features(sample['points'])
        
        # Tokenize word
        tokens = [ord(c) - ord('a') + 1 if c != "'" else 27 
                 for c in sample['word']]
        
        return {
            'features': torch.FloatTensor(features),
            'tokens': torch.LongTensor(tokens),
            'word': sample['word']
        }

def collate_fn(batch):
    """Custom collate function with padding"""
    features = [item['features'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    words = [item['word'] for item in batch]
    
    # Pad features
    max_feat_len = max(f.size(0) for f in features)
    padded_features = torch.zeros(len(batch), max_feat_len, features[0].size(1))
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
    
    # Create mask
    mask = torch.zeros(len(batch), max_feat_len, dtype=torch.bool)
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

class ProductionTrainer:
    """Production training with all optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        
        # Initialize components
        self.featurizer = RobustFeaturizer()
        
        # Calculate input dimension
        input_dim = 15 + self.featurizer.num_keys
        
        # Create model
        self.model = ImprovedTransformerModel(input_dim, config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Scheduler with warmup
        self.scheduler = self.get_cosine_schedule_with_warmup()
        
        # Loss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=config["mixed_precision"])
        
        # Logging
        self.writer = SummaryWriter(config["log_dir"])
        self.global_step = 0
        self.best_loss = float('inf')
    
    def get_cosine_schedule_with_warmup(self):
        """Cosine learning rate schedule with linear warmup"""
        def lr_lambda(step):
            if step < self.config["warmup_steps"]:
                return step / self.config["warmup_steps"]
            
            progress = (step - self.config["warmup_steps"]) / (
                self.config["num_epochs"] * 5000 - self.config["warmup_steps"]
            )
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            features = batch['features'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            token_lengths = batch['token_lengths'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Ensure valid lengths for CTC
            feature_lengths = torch.maximum(feature_lengths, token_lengths)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.config["mixed_precision"]):
                log_probs = self.model(features, mask)
                log_probs = F.log_softmax(log_probs, dim=2)
                
                # Calculate CTC loss
                loss = self.criterion(
                    log_probs, tokens,
                    feature_lengths, token_lengths
                )
            
            # Skip if NaN
            if torch.isnan(loss):
                print(f"NaN loss at step {self.global_step}")
                continue
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["gradient_clip"]
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if self.global_step % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'grad': f'{grad_norm:.2f}'
                })
                
                self.writer.add_scalar('Loss/train', loss_val, self.global_step)
                self.writer.add_scalar('LR', current_lr, self.global_step)
                self.writer.add_scalar('GradNorm', grad_norm, self.global_step)
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                token_lengths = batch['token_lengths'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Ensure valid lengths
                feature_lengths = torch.maximum(feature_lengths, token_lengths)
                
                with torch.amp.autocast('cuda', enabled=self.config["mixed_precision"]):
                    log_probs = self.model(features, mask)
                    log_probs = F.log_softmax(log_probs, dim=2)
                    
                    loss = self.criterion(
                        log_probs, tokens,
                        feature_lengths, token_lengths
                    )
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = Path(self.config["checkpoint_dir"])
        if is_best:
            torch.save(checkpoint, path / 'best_model.pth')
            print(f"Saved best model with loss {self.best_loss:.4f}")
        else:
            torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}.pth')
    
    def train(self):
        """Main training loop"""
        # Create datasets
        train_dataset = RobustDataset(
            self.config["train_data"], self.featurizer,
            self.config["max_seq_length"]
        )
        val_dataset = RobustDataset(
            self.config["val_data"], self.featurizer,
            self.config["max_seq_length"]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(1, self.config["num_epochs"] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log validation loss
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")

if __name__ == "__main__":
    print("Starting Production Training")
    print("="*50)
    
    trainer = ProductionTrainer(CONFIG)
    trainer.train()