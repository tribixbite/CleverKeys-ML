#!/usr/bin/env python3
"""
Stable Training Script for CleverKeys Gesture Typing Model
Incorporates all fixes for NaN issues based on comprehensive analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
from tqdm import tqdm
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_stable.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

@dataclass
class TrainingConfig:
    # Model parameters
    d_model: int = 384
    nhead: int = 6
    num_encoder_layers: int = 8
    dim_feedforward: int = 1536
    vocab_size: int = 28  # 26 letters + space + blank
    dropout: float = 0.1
    max_seq_length: int = 100
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05  # Increased from 0.01 for better regularization
    num_epochs: int = 50
    gradient_clip: float = 0.5  # Reduced from 1.0 for more stability
    warmup_steps: int = 2000
    
    # Stability parameters (NEW)
    logit_clamp_min: float = -10.0
    logit_clamp_max: float = 10.0
    pos_emb_init_std: float = 0.01  # Reduced from 0.02
    max_grad_scaler_scale: float = 32768.0  # Cap the GradScaler
    monitor_frequency: int = 200  # Steps between monitoring logs
    
    # Paths
    data_path: str = "data/train_final_train.jsonl"
    val_path: str = "data/train_final_val.jsonl"
    checkpoint_dir: str = "checkpoints_stable"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True  # Can disable for debugging

CONFIG = TrainingConfig()

# ============================================
# Stable Feature Extraction
# ============================================

class StableFeaturizer:
    """Feature extraction with comprehensive stability checks"""
    
    def __init__(self):
        self.keyboard_layout = self._init_keyboard()
        self.num_keys = len(self.keyboard_layout)
    
    def _init_keyboard(self):
        layout = {
            'q': (0.0, 0.0), 'w': (0.1, 0.0), 'e': (0.2, 0.0), 'r': (0.3, 0.0), 
            't': (0.4, 0.0), 'y': (0.5, 0.0), 'u': (0.6, 0.0), 'i': (0.7, 0.0), 
            'o': (0.8, 0.0), 'p': (0.9, 0.0),
            'a': (0.05, 0.33), 's': (0.15, 0.33), 'd': (0.25, 0.33), 'f': (0.35, 0.33), 
            'g': (0.45, 0.33), 'h': (0.55, 0.33), 'j': (0.65, 0.33), 'k': (0.75, 0.33), 
            'l': (0.85, 0.33),
            'z': (0.15, 0.67), 'x': (0.25, 0.67), 'c': (0.35, 0.67), 'v': (0.45, 0.67), 
            'b': (0.55, 0.67), 'n': (0.65, 0.67), 'm': (0.75, 0.67)
        }
        return layout
    
    def extract_features(self, points_list):
        """Extract features with comprehensive safety checks"""
        if not points_list or len(points_list) < 2:
            return None
        
        # Convert to numpy array
        points_array = np.array([[p['x'], p['y'], p['t']] for p in points_list])
        
        # Validate data
        if np.any(np.isnan(points_array)) or np.any(np.isinf(points_array)):
            logger.warning("Invalid points detected (NaN or Inf)")
            return None
        
        # Clamp input values to reasonable range
        points_array[:, :2] = np.clip(points_array[:, :2], -2.0, 2.0)
        
        features = []
        for i, point in enumerate(points_array):
            feat = []
            
            # Position (2)
            feat.extend([point[0], point[1]])
            
            # Velocity (2) with safe division
            if i > 0:
                dt = max(point[2] - points_array[i-1, 2], 1e-6)
                vx = (point[0] - points_array[i-1, 0]) / dt
                vy = (point[1] - points_array[i-1, 1]) / dt
                feat.extend([np.clip(vx, -10, 10), np.clip(vy, -10, 10)])
            else:
                feat.extend([0.0, 0.0])
            
            # Acceleration (2) with safe division
            if i > 1:
                prev_dt = max(points_array[i-1, 2] - points_array[i-2, 2], 1e-6)
                prev_vx = (points_array[i-1, 0] - points_array[i-2, 0]) / prev_dt
                prev_vy = (points_array[i-1, 1] - points_array[i-2, 1]) / prev_dt
                
                dt = max(point[2] - points_array[i-1, 2], 1e-6)
                ax = (vx - prev_vx) / dt if i > 0 else 0
                ay = (vy - prev_vy) / dt if i > 0 else 0
                feat.extend([np.clip(ax, -10, 10), np.clip(ay, -10, 10)])
            else:
                feat.extend([0.0, 0.0])
            
            # Jerk (2) - simplified to zeros for stability
            feat.extend([0.0, 0.0])
            
            # Angle (1)
            if i > 0 and i < len(points_array) - 1:
                dx = points_array[i+1, 0] - points_array[i-1, 0]
                dy = points_array[i+1, 1] - points_array[i-1, 1]
                angle = np.arctan2(dy, dx)
                feat.append(angle)
            else:
                feat.append(0.0)
            
            # Curvature (1) - simplified
            feat.append(0.0)
            
            # Distance features (2)
            dist_start = np.linalg.norm(point[:2] - points_array[0, :2])
            dist_end = np.linalg.norm(point[:2] - points_array[-1, :2])
            feat.extend([np.clip(dist_start, 0, 5), np.clip(dist_end, 0, 5)])
            
            # Progress (1)
            progress = i / max(len(points_array) - 1, 1)
            feat.append(progress)
            
            # Bearing (1) - simplified
            feat.append(0.0)
            
            # Cumulative distance (1)
            if i > 0:
                cum_dist = sum(np.linalg.norm(points_array[j+1, :2] - points_array[j, :2]) 
                              for j in range(i))
                feat.append(np.clip(cum_dist, 0, 10))
            else:
                feat.append(0.0)
            
            # Key distances (27) with stable Gaussian encoding
            for key, pos in self.keyboard_layout.items():
                dist = np.linalg.norm(point[:2] - np.array(pos))
                # Use bounded exponential to prevent numerical issues
                gaussian_val = np.exp(-min(dist * dist / 0.05, 10))
                feat.append(gaussian_val)
            
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        
        # Final safety check
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = np.clip(features, -10.0, 10.0)
        
        return features

# ============================================
# Stable Model Architecture
# ============================================

class StableTransformerModel(nn.Module):
    """Transformer model with stability improvements"""
    
    def __init__(self, input_dim: int, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Input projection with stable initialization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Initialize input projection carefully
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=0.5)
        nn.init.zeros_(self.input_proj[0].bias)
        
        # Learnable positional embeddings with controlled initialization
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_seq_length, self.d_model) * config.pos_emb_init_std
        )
        
        # Register a hook to monitor positional embeddings
        self.register_buffer('pos_emb_stats', torch.zeros(2))  # [norm, max_abs]
        
        # Transformer encoder with pre-norm for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            norm_first=True  # Pre-norm architecture for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Output projection with careful initialization
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, config.vocab_size)
        )
        
        # Small initialization for output layer
        nn.init.xavier_uniform_(self.output_proj[1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[1].bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with stability checks"""
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional embeddings (safely)
        if seq_len <= self.config.max_seq_length:
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            # Handle sequences longer than expected
            pos_emb = self.pos_embedding.repeat(1, (seq_len // self.config.max_seq_length) + 1, 1)
            x = x + pos_emb[:, :seq_len, :]
        
        # Monitor positional embedding stats
        with torch.no_grad():
            self.pos_emb_stats[0] = self.pos_embedding.norm().item()
            self.pos_emb_stats[1] = self.pos_embedding.abs().max().item()
        
        # Transformer expects (seq, batch, dim)
        x = x.transpose(0, 1)
        
        # Create attention mask if needed
        if mask is not None:
            mask = mask.bool()
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to vocabulary
        x = self.output_proj(x)
        
        # CRITICAL: Clamp logits to prevent overflow in log_softmax
        x = torch.clamp(x, min=self.config.logit_clamp_min, max=self.config.logit_clamp_max)
        
        # Return in CTC format: (seq, batch, vocab)
        return x

# ============================================
# Dataset
# ============================================

class GestureDataset(Dataset):
    """Dataset with comprehensive validation"""
    
    def __init__(self, data_path: str, featurizer: StableFeaturizer, max_seq_len: int = 100):
        self.featurizer = featurizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        # Load and validate data
        with open(data_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    sample = json.loads(line.strip())
                    
                    # Validate sample
                    if 'word' not in sample or 'points' not in sample:
                        continue
                    
                    if len(sample['word']) == 0 or len(sample['points']) < 2:
                        continue
                    
                    # Check for valid points
                    valid = True
                    for p in sample['points']:
                        if not all(k in p for k in ['x', 'y', 't']):
                            valid = False
                            break
                        if not all(isinstance(p[k], (int, float)) for k in ['x', 'y', 't']):
                            valid = False
                            break
                    
                    if valid:
                        self.samples.append(sample)
                
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num}")
                    continue
        
        logger.info(f"Loaded {len(self.samples)} valid samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract features
        features = self.featurizer.extract_features(sample['points'])
        
        if features is None:
            # Return a dummy sample if feature extraction fails
            features = np.zeros((2, 15 + self.featurizer.num_keys), dtype=np.float32)
        
        # Encode target word
        word = sample['word'].lower()
        char_to_idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz ')}
        targets = [char_to_idx.get(c, 0) for c in word]
        
        return {
            'features': features,
            'targets': targets,
            'word': word
        }

def collate_fn(batch, max_seq_len=100):
    """Custom collate function with padding and validation"""
    features_list = []
    targets_list = []
    input_lengths = []
    target_lengths = []
    masks = []
    
    for item in batch:
        features = item['features']
        targets = item['targets']
        
        # Skip invalid samples
        if len(features) == 0 or len(targets) == 0:
            continue
        
        # Truncate or pad features
        seq_len = min(len(features), max_seq_len)
        
        if seq_len < max_seq_len:
            # Pad
            padded = np.zeros((max_seq_len, features.shape[1]), dtype=np.float32)
            padded[:seq_len] = features[:seq_len]
            features = padded
            mask = torch.cat([torch.zeros(seq_len), torch.ones(max_seq_len - seq_len)])
        else:
            features = features[:max_seq_len]
            mask = torch.zeros(max_seq_len)
        
        features_list.append(torch.FloatTensor(features))
        targets_list.append(torch.LongTensor(targets))
        input_lengths.append(seq_len)
        target_lengths.append(len(targets))
        masks.append(mask)
    
    if len(features_list) == 0:
        # Return a dummy batch if all samples are invalid
        return None
    
    return {
        'features': torch.stack(features_list),
        'targets': torch.cat(targets_list),
        'input_lengths': torch.LongTensor(input_lengths),
        'target_lengths': torch.LongTensor(target_lengths),
        'mask': torch.stack(masks)
    }

# ============================================
# Training Functions
# ============================================

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    """Training loop with comprehensive monitoring"""
    model.train()
    total_loss = 0
    num_batches = 0
    num_skipped = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        # Move to device
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        mask = batch['mask'].to(device)
        
        # Monitoring (every N steps)
        global_step = epoch * len(dataloader) + batch_idx
        if global_step % config.monitor_frequency == 0:
            with torch.no_grad():
                pe_norm = model.pos_emb_stats[0].item()
                pe_max = model.pos_emb_stats[1].item()
                grad_scale = scaler.get_scale() if config.use_amp else 1.0
                
                logger.info(f"Step {global_step} | PosEmb: norm={pe_norm:.2f}, max={pe_max:.2f} | "
                           f"GradScale: {grad_scale:.1f}")
                
                # Check for concerning values
                if pe_max > 100:
                    logger.warning(f"Large positional embedding detected: {pe_max}")
                
                # Cap GradScaler if it gets too high
                if config.use_amp and grad_scale > config.max_grad_scaler_scale:
                    logger.warning(f"GradScaler too high ({grad_scale}), resetting")
                    scaler.update(scale=config.max_grad_scaler_scale)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            if config.use_amp:
                with autocast():
                    log_probs = model(features, mask)
                    log_probs = F.log_softmax(log_probs, dim=-1)
                    
                    loss = F.ctc_loss(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths,
                        blank=0,
                        reduction='mean',
                        zero_infinity=True
                    )
            else:
                log_probs = model(features, mask)
                log_probs = F.log_softmax(log_probs, dim=-1)
                
                loss = F.ctc_loss(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction='mean',
                    zero_infinity=True
                )
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at step {global_step}: {loss.item()}")
                num_skipped += 1
                
                # Save problematic batch for debugging
                if num_skipped == 1:
                    torch.save({
                        'batch': batch,
                        'model_state': model.state_dict(),
                        'step': global_step
                    }, 'debug_nan_batch.pt')
                
                continue
            
            # Backward pass
            if config.use_amp:
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.gradient_clip
                )
                
                # Check for gradient explosion
                if grad_norm > config.gradient_clip * 10:
                    logger.warning(f"Large gradient norm: {grad_norm:.2f}")
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.gradient_clip
                )
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / max(1, num_batches),
                'skipped': num_skipped
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            num_skipped += 1
            continue
    
    avg_loss = total_loss / max(1, num_batches)
    logger.info(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}, Skipped: {num_skipped}")
    
    return avg_loss

def validate(model, dataloader, device, config):
    """Validation loop"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue
            
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            mask = batch['mask'].to(device)
            
            log_probs = model(features, mask)
            log_probs = F.log_softmax(log_probs, dim=-1)
            
            loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True
            )
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / max(1, num_batches)

# ============================================
# Main Training Script
# ============================================

def main():
    """Main training function with all stability improvements"""
    
    logger.info("="*50)
    logger.info("Starting Stable Training")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*50)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Initialize components
    featurizer = StableFeaturizer()
    input_dim = 15 + featurizer.num_keys
    
    # Create datasets
    train_dataset = GestureDataset(CONFIG.data_path, featurizer, CONFIG.max_seq_length)
    val_dataset = GestureDataset(CONFIG.val_path, featurizer, CONFIG.max_seq_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, CONFIG.max_seq_length),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, CONFIG.max_seq_length),
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = StableTransformerModel(input_dim, CONFIG)
    model = model.to(CONFIG.device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Initialize optimizer with stronger weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
        eps=1e-8
    )
    
    # Learning rate scheduler (step once per EPOCH, not batch)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.num_epochs,
        eta_min=1e-6
    )
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler(enabled=CONFIG.use_amp)
    
    # Create checkpoint directory
    Path(CONFIG.checkpoint_dir).mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, 
            CONFIG.device, epoch, CONFIG
        )
        
        # Validate
        val_loss = validate(model, val_loader, CONFIG.device, CONFIG)
        
        # Step scheduler (once per epoch!)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG.__dict__
            }
            
            torch.save(
                checkpoint,
                Path(CONFIG.checkpoint_dir) / 'best_model.pth'
            )
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                checkpoint,
                Path(CONFIG.checkpoint_dir) / f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()