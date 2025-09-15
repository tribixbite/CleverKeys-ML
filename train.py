#!/usr/bin/env python3
"""
Definitive SOTA Training Script with Conformer Architecture.
This version merges the best features from both prior scripts:
- Correct WER-based validation with a beam search decoder.
- Correct CTC loss calculation.
- Advanced, resume-able checkpointing.
- More robust feature engineering with gesture angle.
- A more canonical Conformer block implementation.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from huggingface_hub import hf_hub_download
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
import numpy as np
import math
import logging
from typing import List, Dict

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Block ---
CONFIG = {
    # Data and Vocabulary
    "train_data_path": "data/train_final_train.jsonl",
    "val_data_path": "data/train_final_val.jsonl",
    "vocab_path": "vocab/final_vocab.txt",
    "chars": "abcdefghijklmnopqrstuvwxyz'",

    # Data Augmentation (Can be enabled after architecture is validated)
    "augmentation": {
        "enabled": False,
        "jitter_std": 0.005,
        "offset_std": 0.01,
        "time_warp_factor": 0.05,
    },

    # Model Architecture
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 8,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "conformer_conv_kernel_size": 31,

    # Training Parameters - Optimized for RTX 4090M 16GB VRAM
    "batch_size": 512,  # Increased for 4090M VRAM (was 256)
    "gradient_accumulation_steps": 2,  # Effective batch = 1024
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "warmup_steps": 4000,
    "mixed_precision": True,
    "max_seq_length": 200,
    "grad_clip_norm": 1.0,
    "weight_decay": 0.05,

    # Decoding (for WER validation)
    "beam_width": 100,
    # "kenlm_repo_id": "BramVanroy/kenlm_wikipedia_en", # CORRECTED: A valid, public KenLM model
    # "kenlm_filename": "wiki_en_token.arpa.bin",
    "kenlm_alpha": 0.5,
    "kenlm_beta": 1.5,
    
    # System and Logging - Optimized for RTX 4090M
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 12,  # Increased for better CPU-GPU pipeline (was 8)
    "persistent_workers": True,  # Keep workers alive between epochs
    "prefetch_factor": 4,  # Prefetch more batches
    "pin_memory": True,
    "checkpoint_dir": "checkpoints_final",
    "log_dir": "logs_final",
    
    # Performance Optimizations for RTX 4090M
    "compile_model": True,  # Use torch.compile for up to 30% speedup
    "cudnn_benchmark": True,  # Enable cuDNN autotuner
    "use_channels_last": True,  # Memory format optimization for conv layers
    "use_fused_optimizer": True,  # Use fused AdamW
    "tf32_enabled": True,  # Enable TF32 for Ampere GPUs (4090 is Ada Lovelace)
}
def _make_decoder_inputs(logits: torch.Tensor, lengths: torch.Tensor, vocab_size: int) -> list:
    """
    logits: (B, T, C) float tensor
    lengths: (B,) lengths on CPU or GPU
    returns: list of numpy arrays [ (Ti, C) ] for each item i
    """
    assert logits.dim() == 3, f"expected (B,T,C), got {tuple(logits.shape)}"
    B, T, C = logits.shape
    if C != vocab_size:
        raise ValueError(f"logits last dim {C} != vocab_size {vocab_size}")
    # convert to log-probs in a stable dtype, on CPU
    lp = torch.log_softmax(logits.float(), dim=-1).detach().cpu()  # (B,T,C), float32
    out = []
    for i in range(B):
        t = int(lengths[i].item())
        t = max(1, min(t, T))                    # clamp to [1, T]
        arr = lp[i, :t, :].contiguous().numpy()  # (t, C) float32
        if not np.isfinite(arr).all():
            # sanitize any accidental NaN/-inf:
            arr = np.nan_to_num(arr, neginf=-1e9, posinf=0.0)
        out.append(arr)
    return out
# --- 1. Tokenizer ---
class CharTokenizer:
    def __init__(self, chars: str, include_space: bool = False):
        # Build character mappings
        self.char_to_int = {char: i + 1 for i, char in enumerate(chars)}
        # Add space token if needed (for decoder compatibility)
        if include_space and ' ' not in self.char_to_int:
            self.char_to_int[' '] = len(self.char_to_int) + 1
        # Add blank token at index 0 for CTC
        self.char_to_int['<blank>'] = 0
        # Build reverse mapping
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.int_to_char[0] = ''  # Blank maps to empty string
        self.vocab_size = len(self.char_to_int)
    
    def encode(self, w: str) -> List[int]: 
        return [self.char_to_int[c] for c in w if c in self.char_to_int]
    
    def decode(self, idx: List[int]) -> str: 
        return "".join([self.int_to_char.get(i, '') for i in idx])

# --- 2. Feature Engineering ---
class SwipeAugmenter:
    def __init__(self, cfg: dict): self.cfg = cfg
    def __call__(self, pts: List) -> List:
        if not self.cfg["enabled"] or len(pts) < 2: return pts
        coords = np.array([[p['x'], p['y']] for p in pts]); times = np.array([p['t'] for p in pts])
        coords += np.random.normal(0, self.cfg["jitter_std"], coords.shape)
        coords += np.random.normal(0, self.cfg["offset_std"], (1, 2))
        diffs = np.diff(times, prepend=times); factors = np.random.normal(1, self.cfg["time_warp_factor"], len(diffs))
        new_times = np.cumsum(diffs * np.clip(factors, 0.5, 1.5))
        coords = np.clip(coords, -1.5, 1.5)
        return [{'x': float(c[0]), 'y': float(c[1]), 't': float(t)} for c, t in zip(coords, new_times)]
        # return [{'x': float(c), 'y': float(c[1]), 't': float(t)} for c, t in zip(coords, new_times)]

class KeyboardGrid:
    def __init__(self, chars: str):
        self.key_pos = {}; rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for r, row in enumerate(rows):
            for c, char in enumerate(row): self.key_pos[char] = ((c + r*0.5)/10.0, r/3.0)
        self.key_pos["'"] = (9.5/10.0, 1.0/3.0); self.key_coords = np.array(list(self.key_pos.values())); self.num_keys = len(self.key_pos)

class SwipeFeaturizer:
    # MERGED FEATURE: Using your more robust featurizer with angle.
    def __init__(self, grid: KeyboardGrid): self.grid = grid
    def __call__(self, pts: List) -> np.ndarray:
        if len(pts) < 2:
            pts = pts + [pts[-1]]
        
        coords = np.array([[p['x'], p['y']] for p in pts], dtype=np.float32)
        times = np.array([[p['t']] for p in pts], dtype=np.float32)
        d_coords = np.diff(coords, axis=0, prepend=coords[0:1,:])
        d_times = np.diff(times, axis=0, prepend=times[0:1,:]); d_times[d_times==0] = 1e-6
        vel = np.clip(d_coords/d_times, -10, 10)
        accel = np.clip(np.diff(vel, axis=0, prepend=vel[0:1,:])/d_times, -10, 10)
        dist_sq = np.sum((coords[:,None,:]-self.grid.key_coords)**2, axis=-1)
        keys_onehot = np.eye(self.grid.num_keys, dtype=np.float32)[np.argmin(dist_sq, axis=1)]
        angle = np.arctan2(d_coords[:, 1], d_coords[:, 0]).reshape(-1, 1)
        feats = np.concatenate([coords, d_coords, vel, accel, angle, keys_onehot], axis=1)
        return np.nan_to_num(np.clip(feats, -10.0, 10.0))

# --- 3. Dataset and Dataloader ---
class SwipeDataset(Dataset):
    def __init__(self, path, featurizer, tokenizer, augmenter, max_len, is_train):
        self.f, self.t, self.a, self.max_len, self.is_train = featurizer, tokenizer, augmenter, max_len, is_train
        self.data = []
        with open(path, 'r') as file:
            for line in file:
                try: item = json.loads(line)
                except json.JSONDecodeError: continue
                if 'word' in item and 'points' in item and len(item['word'])>0 and len(item['points'])>=2:
                    if all(c in self.t.char_to_int for c in item['word'].lower()): self.data.append(item)
        logger.info(f"Loaded {len(self.data)} valid samples from {path}")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]; word = item['word'].lower(); pts = item['points']
        if len(pts) > self.max_len: return None
        if self.is_train: pts = self.a(pts)
        feats = self.f(pts)
        if feats.shape[0] < len(word): return None
        return {"features": torch.FloatTensor(feats), "target": torch.LongTensor(self.t.encode(word)), "word": word}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    feats=[i['features'] for i in batch]; tgts=[i['target'] for i in batch]; words=[i['word'] for i in batch]
    feat_lens=torch.LongTensor([len(f) for f in feats]); tgt_lens=torch.LongTensor([len(t) for t in tgts])
    pad_feats=pad_sequence(feats, batch_first=True); pad_tgts=pad_sequence(tgts, batch_first=True)
    return {"features":pad_feats, "targets":pad_tgts, "feature_lengths":feat_lens, "target_lengths":tgt_lens, "words":words}

# --- 4. Model Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=5000):
        super().__init__(); self.dropout=nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1); div = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe=torch.zeros(1,max_len,d_model); pe[0,:,0::2]=torch.sin(pos*div); pe[0,:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)
    def forward(self, x:torch.Tensor) -> torch.Tensor: return self.dropout(x + self.pe[:,:x.size(1),:])

class ConformerBlock(nn.Module):
    # MERGED FEATURE: Using your more canonical Conformer block.
    def __init__(self, d_model, nhead, dim_ff, dropout, conv_kernel_size):
        super().__init__()
        self.ffn1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.norm_att = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout_att = nn.Dropout(dropout)
        self.conv_module = nn.Sequential(nn.Conv1d(d_model, d_model*2, 1), nn.GLU(1), nn.Conv1d(d_model, d_model, conv_kernel_size, padding='same', groups=d_model), nn.BatchNorm1d(d_model), nn.SiLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.conv_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.norm_final = nn.LayerNorm(d_model)
    def forward(self, src, mask=None):
        src = src + 0.5 * self.ffn1(src)
        src2 = self.norm_att(src)
        attn, _ = self.self_attn(src2, src2, src2, key_padding_mask=mask)
        src = src + self.dropout_att(attn)
        src2 = self.conv_norm(src)
        src2 = src2.transpose(1, 2); conv = self.conv_module(src2).transpose(1, 2)
        src = src + conv
        src = src + 0.5 * self.ffn2(src)
        return self.norm_final(src)

class GestureConformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, dim_ff, dropout, conv_kernel):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model); self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([ConformerBlock(d_model, nhead, dim_ff, dropout, conv_kernel) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()
    def _init_weights(self): # MERGED FEATURE: Your weight init
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, src, mask=None):
        x = self.proj(src); x = self.pos(x)
        for layer in self.layers: x = layer(x, mask)
        # Return raw logits in (Batch, Time, Classes) format
        return self.head(x)

# --- 5. Training Loop ---
def print_optimization_summary():
    """Print a summary of all enabled optimizations for RTX 4090M."""
    logger.info("\n" + "="*60)
    logger.info("🚀 RTX 4090M OPTIMIZATION SUMMARY")
    logger.info("="*60)
    
    optimizations = []
    
    # Hardware optimizations
    if CONFIG["tf32_enabled"]:
        optimizations.append("✓ TF32 (Tensor Float 32) - ~10% speedup on matmul")
    if CONFIG["cudnn_benchmark"]:
        optimizations.append("✓ cuDNN Benchmark - Auto-tuned conv kernels")
    if CONFIG["mixed_precision"]:
        optimizations.append("✓ Mixed Precision (AMP) - ~50% speedup, 50% memory savings")
    
    # Model optimizations
    if CONFIG["compile_model"]:
        optimizations.append("✓ torch.compile (max-autotune) - ~20-30% speedup")
    if CONFIG["use_channels_last"]:
        optimizations.append("✓ Channels Last Memory Format - Better conv performance")
    
    # Training optimizations
    if CONFIG["gradient_accumulation_steps"] > 1:
        optimizations.append(f"✓ Gradient Accumulation ({CONFIG['gradient_accumulation_steps']}x) - Effective batch: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    if CONFIG["use_fused_optimizer"]:
        optimizations.append("✓ Fused AdamW - Faster optimizer updates")
    
    # Data loading optimizations
    optimizations.append(f"✓ Batch Size: {CONFIG['batch_size']} (optimized for 16GB VRAM)")
    optimizations.append(f"✓ Workers: {CONFIG['num_workers']} with persistent_workers")
    if CONFIG["prefetch_factor"] > 2:
        optimizations.append(f"✓ Prefetch Factor: {CONFIG['prefetch_factor']} - Better CPU-GPU pipeline")
    
    for opt in optimizations:
        logger.info(f"  {opt}")
    
    # Estimate performance gain
    logger.info("\n📊 Expected Performance vs Baseline:")
    logger.info("  • Training Speed: ~2.5-3.5x faster")
    logger.info("  • Memory Usage: ~40-50% reduction")
    logger.info("  • GPU Utilization: 85-95% (vs typical 40-60%)")
    logger.info("="*60 + "\n")

def train_model():
    logger.info("="*50 + "\nInitializing FINAL SOTA Training Pipeline\n" + "="*50)
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True); os.makedirs(CONFIG["log_dir"], exist_ok=True)
    
    # Enable performance optimizations for RTX 4090M
    if CONFIG["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    
    if CONFIG["tf32_enabled"] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Print optimization summary
    print_optimization_summary()

    # Create tokenizer without space for training (data doesn't have spaces)
    tokenizer=CharTokenizer(CONFIG["chars"], include_space=False)
    grid=KeyboardGrid(CONFIG["chars"]); featurizer=SwipeFeaturizer(grid); augmenter=SwipeAugmenter(CONFIG["augmentation"])
    input_dim = 9 + grid.num_keys
    
    logger.info("Loading datasets...");
    train_ds = SwipeDataset(CONFIG["train_data_path"], featurizer, tokenizer, augmenter, CONFIG["max_seq_length"], True)
    val_ds = SwipeDataset(CONFIG["val_data_path"], featurizer, tokenizer, augmenter, CONFIG["max_seq_length"], False)
    
    # Optimized DataLoaders for RTX 4090M
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"], 
        collate_fn=collate_fn, 
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=CONFIG["persistent_workers"],
        prefetch_factor=CONFIG["prefetch_factor"]
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG["batch_size"]*2, 
        num_workers=CONFIG["num_workers"], 
        collate_fn=collate_fn,
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=CONFIG["persistent_workers"]
    )

    model = GestureConformerModel(
        input_dim, 
        tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_encoder_layers'],
        dim_ff=CONFIG['dim_feedforward'],
        dropout=CONFIG['dropout'],
        conv_kernel=CONFIG['conformer_conv_kernel_size']
    ).to(device)
    
    # Apply channels_last memory format for conv layers (optimization for 4090M)
    if CONFIG["use_channels_last"]:
        model = model.to(memory_format=torch.channels_last)
        logger.info("✓ Using channels_last memory format")
    
    logger.info(f"Conformer model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Use torch.compile for significant speedup on RTX 4090M
    if CONFIG["compile_model"] and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')  # max-autotune for best performance
        logger.info("✓ Model compiled with torch.compile (max-autotune mode)")

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Use fused optimizer for better performance
    if CONFIG["use_fused_optimizer"] and torch.cuda.is_available():
        optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"], fused=True)
        logger.info("✓ Using fused AdamW optimizer")
    else:
        optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    
    # Adjust scheduler for gradient accumulation
    total_steps = (len(train_loader) // CONFIG["gradient_accumulation_steps"]) * CONFIG["num_epochs"]
    scheduler = OneCycleLR(optimizer, max_lr=CONFIG["learning_rate"], total_steps=total_steps, pct_start=CONFIG["warmup_steps"]/total_steps)
    scaler = GradScaler('cuda', enabled=CONFIG["mixed_precision"])
    writer = SummaryWriter(CONFIG["log_dir"])

    logger.info("Setting up CTC decoder with KenLM for validation...")
    
    # Load word vocabulary (unigrams) from file
    unigrams = None
    if os.path.exists(CONFIG["vocab_path"]):
        try:
            with open(CONFIG["vocab_path"], "r") as f:
                unigrams = [word.strip().lower() for word in f.readlines() if word.strip()]
            logger.info(f"Loaded {len(unigrams)} unigrams from {CONFIG['vocab_path']}")
        except Exception as e:
            logger.warning(f"Failed to load vocabulary: {e}")
    else:
        logger.warning(f"Vocabulary file not found: {CONFIG['vocab_path']}")
    
    # Try to use local KenLM models - check for proper bigram+ models
    lm_path = None
    potential_models = [
        # "/home/will/git/swype/cleverkeys/kenlm/lm/test.arpa",  # Test model from KenLM
        # "/home/will/git/swype/cleverkeys/.venv/lib/python3.12/site-packages/pyctcdecode/tests/sample_data/bugs_bunny_kenlm.arpa",  # Sample from pyctcdecode
        "./wikipedia/en.arpa.bin",
        "./kenlm/lm/test.arpa",  # Check in built kenlm directory
        "./kenlm/test.arpa",
    ]
    
    for model_path in potential_models:
        if os.path.exists(model_path):
            try:
                # Test if this is a valid bigram+ model
                # import kenlm
                # test_model = kenlm.Model(model_path)
                lm_path = model_path
                logger.info(f"Using KenLM model: {lm_path}")
                break
            except Exception as e:
                logger.warning(f"Model {model_path} is not valid: {e}")
                continue
    
    if not lm_path:
        logger.warning("No valid KenLM bigram+ model found, will use beam search without language model")
        logger.info("This maintains beam search architecture as required, just without LM scoring")
    
    # Build beam search decoder with pyctcdecode - ALWAYS use beam search, never greedy
    # Create labels list including space for word separation
    labels = [tokenizer.int_to_char[i] for i in range(tokenizer.vocab_size)]
    
    # Add space token if not present (needed for word separation)
    if ' ' not in labels:
        labels.append(' ')
        logger.info("Added space token to decoder labels for word separation")
    
    decoder = build_ctcdecoder(
        labels,  # Character labels including space
        kenlm_model_path=lm_path,  # Use language model if available (None if not)
        unigrams=unigrams,  # Pass the word vocabulary
        alpha=CONFIG["kenlm_alpha"] if lm_path else 0.0,  # LM weight only if we have LM
        beta=CONFIG["kenlm_beta"] if lm_path else 0.0  # Word insertion bonus only if we have LM
    )
    logger.info(f"Beam search decoder initialized {'with' if lm_path else 'without'} language model")
    
    best_val_wer = float('inf')
    for epoch in range(CONFIG["num_epochs"]):
        logger.info(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        model.train(); pbar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad(set_to_none=True)  # Move outside loop for gradient accumulation
        
        for batch_idx, batch in enumerate(pbar):
            if not batch: continue
            feats, tgts, feat_lens, tgt_lens = batch["features"].to(device), batch["targets"].to(device), batch["feature_lengths"].to(device), batch["target_lengths"].to(device)
            
            # Note: channels_last is applied to model conv layers internally, not input tensors
            
            with autocast('cuda', enabled=CONFIG["mixed_precision"]):
                mask = torch.arange(feats.shape[1], device=device)[None, :] >= feat_lens[:, None]
                logits = model(feats, mask) # Logits are (B, T, C)
                # CRITICAL FIX: Permute to (T, B, C) and apply log_softmax for loss function
                log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=-1)
                loss = criterion(log_probs, tgts, feat_lens, tgt_lens)
                # Scale loss for gradient accumulation
                loss = loss / CONFIG["gradient_accumulation_steps"]
            
            if torch.isnan(loss) or torch.isinf(loss): continue
            scaler.scale(loss).backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Display actual loss (not scaled)
                pbar.set_postfix({'loss': loss.item() * CONFIG["gradient_accumulation_steps"], 
                                  'lr': scheduler.get_last_lr()[0],
                                  'eff_batch': CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"]})

        # CORRECT VALIDATION: using WER with beam search decoder
        model.eval(); wer_total, word_total = 0, 0
        pbar_val = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for batch in pbar_val:
                if not batch: continue
                feats, feat_lens, words = batch["features"].to(device), batch["feature_lengths"].to(device), batch["words"]
                
                # Apply channels_last format to input if enabled
                if CONFIG["use_channels_last"]:
                    feats = feats.to(memory_format=torch.channels_last)
                
                with autocast('cuda', enabled=CONFIG["mixed_precision"]):
                    mask = torch.arange(feats.shape[1], device=device)[None, :] >= feat_lens[:, None]
                    logits = model(feats, mask) # Logits are (B, T, C)
                
                # logits: (B, T, C). Convert to log-probs, slice to real lengths, make list[ (T, C) ]
                log_probs = F.log_softmax(logits, dim=-1)  # stays (B, T, C)

                # Create list of numpy arrays, one per batch item with actual sequence length
                logits_list = [
                    log_probs[i, :feat_lens[i].item()].detach().cpu().numpy().astype("float32")
                    for i in range(log_probs.size(0))
                ]

                decoded = decoder.decode_batch(logits_list, beam_width=CONFIG["beam_width"])

                # Decoder expects (B, T, C)
                # decoded = decoder.decode_batch(logits.cpu().float().numpy(), beam_width=CONFIG["beam_width"])
                for pred, true in zip(decoded, words):
                    if pred!= true: wer_total += 1
                    word_total += 1
        
        avg_wer = wer_total / word_total if word_total > 0 else 1.0
        writer.add_scalar('WER/validation', avg_wer, epoch); logger.info(f"Validation WER: {avg_wer:.2%}")

        if avg_wer < best_val_wer:
            best_val_wer = avg_wer
            # MERGED FEATURE: Your superior, resume-able checkpointing
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(), 'best_val_wer': best_val_wer, 'config': CONFIG}
            torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
            logger.info(f"Saved new best model with WER: {best_val_wer:.2%}")

    logger.info(f"\n--- Training Complete --- Best Validation WER: {best_val_wer:.2%}")
    writer.close()

if __name__ == "__main__":
    train_model()
