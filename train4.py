#!/usr/bin/env python3
"""
Definitive SOTA Training Script with Conformer Architecture.
v6: Uses the official, verified KenLM scorer model from the pyctcdecode
    authors and ensures the decoder vocabulary is formatted correctly.
    All previous API and logic fixes are maintained.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('training_final.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Configuration Block ---
CONFIG = {
    # Data and Vocabulary
    "train_data_path": "data/train_final_train.jsonl",
    "val_data_path": "data/train_final_val.jsonl",
    "vocab_path": "vocab/final_vocab.txt",
    "chars": "abcdefghijklmnopqrstuvwxyz'",

    # Data Augmentation
    "augmentation": {"enabled": False, "jitter_std": 0.005, "offset_std": 0.01, "time_warp_factor": 0.05},

    # Model Architecture
    "d_model": 256, "nhead": 4, "num_encoder_layers": 8, "dim_feedforward": 1024,
    "dropout": 0.1, "conformer_conv_kernel_size": 31,

    # Training Parameters
    "batch_size": 256, "learning_rate": 3e-4, "num_epochs": 50, "warmup_steps": 4000,
    "mixed_precision": True, "max_seq_length": 200, "grad_clip_norm": 1.0, "weight_decay": 0.05,

    # Decoding (for WER validation)
    "beam_width": 100,
    # FIX: Using the official, verified KenLM model from the library's authors.
    "kenlm_repo_id": "parlance/kenlm-en-large",
    "kenlm_filename": "kenlm.scorer",
    "kenlm_alpha": 0.5, "kenlm_beta": 1.0,
    
    # System and Logging
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8, "checkpoint_dir": "checkpoints_final_v6", "log_dir": "logs_final_v6",
}

# --- 1. Tokenizer ---
class CharTokenizer:
    def __init__(self, chars: str):
        self.char_to_int = {char: i + 1 for i, char in enumerate(chars)}; self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.char_to_int['<blank>'] = 0; self.int_to_char[0] = '' # Blank is empty string
        self.vocab_size = len(self.char_to_int)
    def encode(self, w: str) -> List[int]: return [self.char_to_int.get(c, 0) for c in w]
    def decode(self, idx: List[int]) -> str: return "".join([self.int_to_char.get(i, '') for i in idx])

# --- 2. Feature Engineering ---
class SwipeAugmenter:
    def __init__(self, cfg: dict): self.cfg = cfg
    def __call__(self, pts: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if not self.cfg["enabled"] or len(pts) < 2: return pts
        coords = np.array([[p['x'], p['y']] for p in pts]); times = np.array([p['t'] for p in pts])
        coords += np.random.normal(0, self.cfg["jitter_std"], coords.shape)
        coords += np.random.normal(0, self.cfg["offset_std"], (1, 2))
        diffs = np.diff(times, prepend=times[0]); factors = np.random.normal(1, self.cfg["time_warp_factor"], len(diffs))
        new_times = np.cumsum(diffs * np.clip(factors, 0.5, 1.5))
        coords = np.clip(coords, -1.5, 1.5)
        return [{'x': float(c[0]), 'y': float(c[1]), 't': float(t)} for c, t in zip(coords, new_times)]

class KeyboardGrid:
    def __init__(self, chars: str):
        self.key_pos = {}; rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                if char in chars: self.key_pos[char] = ((c + r*0.5)/10.0, r/3.0)
        if "'" in chars: self.key_pos["'"] = (9.5/10.0, 1.0/3.0)
        self.key_coords = np.array(list(self.key_pos.values())); self.num_keys = len(self.key_pos)

class SwipeFeaturizer:
    def __init__(self, grid: KeyboardGrid): self.grid = grid
    def __call__(self, pts: List[Dict[str, float]]) -> np.ndarray:
        if len(pts) < 2: pts.append(pts[0])
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
                if 'word' in item and 'points' in item and len(item['word']) > 0 and len(item['points']) >= 2:
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
    batch = [item for item in batch if item is not None];
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
        src2 = self.conv_norm(src).transpose(1, 2)
        conv = self.conv_module(src2).transpose(1, 2)
        src = src + conv
        src = src + 0.5 * self.ffn2(src)
        return self.norm_final(src)

class GestureConformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, conformer_conv_kernel_size):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model); self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([ConformerBlock(d_model, nhead, dim_feedforward, dropout, conformer_conv_kernel_size) for _ in range(num_encoder_layers)])
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, src, mask=None):
        x = self.proj(src); x = self.pos(x)
        for layer in self.layers: x = layer(x, mask)
        return self.head(x)

# --- 5. Training Loop ---
def train_model():
    logger.info("="*50 + "\nInitializing FINAL SOTA Training Pipeline (v6)\n" + "="*50)
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True); os.makedirs(CONFIG["log_dir"], exist_ok=True)

    tokenizer=CharTokenizer(CONFIG["chars"]); grid=KeyboardGrid(CONFIG["chars"]); featurizer=SwipeFeaturizer(grid); augmenter=SwipeAugmenter(CONFIG["augmentation"])
    input_dim = 9 + grid.num_keys
    
    train_ds = SwipeDataset(CONFIG["train_data_path"], featurizer, tokenizer, augmenter, CONFIG["max_seq_length"], True)
    val_ds = SwipeDataset(CONFIG["val_data_path"], featurizer, tokenizer, augmenter, CONFIG["max_seq_length"], False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"]*2, num_workers=CONFIG["num_workers"], collate_fn=collate_fn)

    model_args = {k:v for k,v in CONFIG.items() if k in ['d_model','nhead','num_encoder_layers','dim_feedforward','dropout','conformer_conv_kernel_size']}
    model = GestureConformerModel(input_dim, tokenizer.vocab_size, **model_args).to(device)
    logger.info(f"Conformer model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    total_steps = len(train_loader) * CONFIG["num_epochs"]
    scheduler = OneCycleLR(optimizer, max_lr=CONFIG["learning_rate"], total_steps=total_steps, pct_start=CONFIG["warmup_steps"]/total_steps)
    scaler = GradScaler(device.type, enabled=CONFIG["mixed_precision"])
    writer = SummaryWriter(CONFIG["log_dir"])

    logger.info("Setting up CTC decoder for validation...")
    try:
        lm_path = hf_hub_download(repo_id=CONFIG["kenlm_repo_id"], filename=CONFIG["kenlm_filename"])
        # FIX: Construct the alphabet correctly for the decoder to avoid warnings
        # The alphabet must be in the same order as the model's output neurons
        alphabet = [tokenizer.int_to_char.get(i, '') for i in range(tokenizer.vocab_size)]
        alphabet[tokenizer.char_to_int['<blank>']] = "" # Blank char is empty
        
        # The .scorer format is self-contained and doesn't need a separate vocab list
        decoder = build_ctcdecoder(alphabet, kenlm_model_path=lm_path, alpha=CONFIG["kenlm_alpha"], beta=CONFIG["kenlm_beta"])
        logger.info("Successfully loaded decoder with KenLM language model.")
    except Exception as e:
        logger.warning(f"Could not load KenLM model from Hub: {e}.")
        logger.warning("FALLING BACK to a decoder without a language model. WER will be higher.")
        decoder = build_ctcdecoder([tokenizer.int_to_char.get(i, '') for i in range(tokenizer.vocab_size)])

    best_val_wer = float('inf')
    for epoch in range(CONFIG["num_epochs"]):
        logger.info(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        model.train(); pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            if not batch: continue
            feats, tgts, feat_lens, tgt_lens = batch["features"].to(device), batch["targets"].to(device), batch["feature_lengths"].to(device), batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            # BUG FIX: Use modern autocast API
            with autocast(device.type, enabled=CONFIG["mixed_precision"]):
                mask = torch.arange(feats.shape[1], device=device)[None, :] >= feat_lens[:, None]
                logits = model(feats, mask)
                log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=-1)
                loss = criterion(log_probs, tgts, feat_lens, tgt_lens)
            if torch.isnan(loss) or torch.isinf(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
            # BUG FIX: Correct order for scheduler and optimizer steps
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

        model.eval(); wer_total, word_total = 0, 0
        pbar_val = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for batch in pbar_val:
                if not batch: continue
                feats, feat_lens, words = batch["features"].to(device), batch["feature_lengths"].to(device), batch["words"]
                # BUG FIX: Use modern autocast API
                with autocast(device.type, enabled=CONFIG["mixed_precision"]):
                    mask = torch.arange(feats.shape[1], device=device)[None, :] >= feat_lens[:, None]
                    logits = model(feats, mask)
                
                decoded = decoder.decode_batch(logits.cpu().float().numpy(), beam_width=CONFIG["beam_width"])
                for pred, true in zip(decoded, words):
                    if pred != true: wer_total += 1
                    word_total += 1
        
        avg_wer = wer_total / word_total if word_total > 0 else 1.0
        writer.add_scalar('WER/validation', avg_wer, epoch); logger.info(f"Validation WER: {avg_wer:.2%}")

        if avg_wer < best_val_wer:
            best_val_wer = avg_wer
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(), 'best_val_wer': best_val_wer, 'config': CONFIG}
            torch.save(checkpoint, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
            logger.info(f"Saved new best model with WER: {best_val_wer:.2%}")

    logger.info(f"\n--- Training Complete --- Best Validation WER: {best_val_wer:.2%}")
    writer.close()

if __name__ == "__main__":
    train_model()
