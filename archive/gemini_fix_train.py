import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import hf_hub_download
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
import numpy as np
import math
from typing import List, Dict

# --- Configuration Block ---
# All hyperparameters and paths are centralized here for easy modification.
CONFIG = {
    # Data and Vocabulary
    "train_data_path": "data/train_final_train.jsonl",
    "val_data_path": "data/train_final_val.jsonl",
    "vocab_path": "vocab/final_vocab.txt",
    "chars": "abcdefghijklmnopqrstuvwxyz'", # Includes apostrophe
    "max_seq_len": 200, # FIX 8: Add max sequence length to prevent OOM

    # Model Architecture
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1,

    # Training Parameters
    "batch_size": 256,
    "learning_rate": 3e-4,
    "num_epochs": 150,
    "grad_clip_norm": 1.0, # FIX 7: Add gradient clipping for stability
    "mixed_precision": True,

    # Decoding and Evaluation
    "beam_width": 100,
    # FIX 1: Use a valid, high-quality KenLM model from the Mozilla-Ocho repository
    "kenlm_repo_id": "Mozilla-Ocho/kenlm",
    "kenlm_filename": "English/ocho_en_4gram_small.arpa", # Updated filename
    "kenlm_alpha": 0.5,
    "kenlm_beta": 1.5,

    # System and Logging
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "export_dir": "exports",
}

# --- 1. Tokenizer ---
class CharTokenizer:
    """Manages mapping between characters and integer indices."""
    def __init__(self, chars: str):
        self.char_to_int = {char: i + 1 for i, char in enumerate(chars)}
        self.int_to_char = {i + 1: char for i, char in enumerate(chars)}
        self.char_to_int['<blank>'] = 0
        self.int_to_char[0] = '<blank>'
        self.vocab_size = len(self.char_to_int)

    def encode(self, word: str) -> List[int]:
        return [self.char_to_int[c] for c in word]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.int_to_char.get(i, '') for i in indices])

# --- 2. Feature Engineering ---
class KeyboardGrid:
    """A simplified representation of a keyboard layout for nearest key lookups."""
    def __init__(self, chars: str):
        self.key_positions = {}
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                if char in chars:
                    self.key_positions[char] = (c + r * 0.5, r)
        if "'" in chars:
             self.key_positions["'"] = (9.5, 1.0)
        self.key_names = list(self.key_positions.keys())
        self.key_coords = np.array(list(self.key_positions.values()))
        self.num_keys = len(self.key_names)

class SwipeFeaturizer:
    """Processes raw swipe data into rich feature vectors."""
    def __init__(self, grid: KeyboardGrid):
        self.grid = grid

    def __call__(self, points: List[Dict[str, float]]) -> np.ndarray:
        if len(points) < 2:
            points.append(points[0])

        coords = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        times = np.array([p['t'] for p in points], dtype=np.float32).reshape(-1, 1)

        delta_coords = np.diff(coords, axis=0, prepend=coords[0:1,:])
        delta_times = np.diff(times, axis=0, prepend=times[0:1,:])
        delta_times[delta_times == 0] = 1e-6
        velocity = delta_coords / delta_times
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1,:]) / delta_times

        dist_sq = np.sum((coords[:, np.newaxis, :] - self.grid.key_coords) ** 2, axis=-1)
        nearest_key_indices = np.argmin(dist_sq, axis=1)
        nearest_keys_onehot = np.eye(self.grid.num_keys, dtype=np.float32)[nearest_key_indices]

        # FIX 3: Correctly assemble the 8 kinematic features.
        features = np.concatenate([
            coords,         # 2 features
            delta_coords,   # 2 features
            velocity,       # 2 features
            acceleration,   # 2 features
            nearest_keys_onehot
        ], axis=1)
        return features

# --- 3. Dataset and Dataloader ---
class SwipeDataset(Dataset):
    def __init__(self, jsonl_path: str, featurizer: SwipeFeaturizer, tokenizer: CharTokenizer, max_seq_len: int):
        self.featurizer = featurizer
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len # FIX 8
        with open(jsonl_path, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        data = json.loads(line)
        word = data['word'].lower()

        # FIX 8: Enforce max sequence length
        if len(data['points']) > self.max_seq_len:
            return None

        if not all(c in self.tokenizer.char_to_int for c in word):
            return None

        features = self.featurizer(data['points'])
        target = self.tokenizer.encode(word)

        return {"features": torch.FloatTensor(features), "target": torch.LongTensor(target), "word": word}

def collate_fn(batch):
    """Pads sequences for batching and filters invalid samples."""
    # Filter out None items from dataset __getitem__
    batch = [item for item in batch if item is not None]

    # FIX 7: Filter out items where input length < target length (CTC constraint)
    batch = [item for item in batch if item['features'].shape[0] >= len(item['target'])]

    if not batch:
        return None

    features = [item['features'] for item in batch]
    targets = [item['target'] for item in batch]
    words = [item['word'] for item in batch]

    feature_lengths = torch.LongTensor([len(f) for f in features])
    target_lengths = torch.LongTensor([len(t) for t in targets])

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return {"features": padded_features, "targets": padded_targets, "feature_lengths": feature_lengths, "target_lengths": target_lengths, "words": words}

# --- 4. Model Architecture (CTC-based) ---
class PositionalEncoding(nn.Module):
    # FIX 5: Rework PositionalEncoding for batch_first=True
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape (1, max_len, d_model) for batch_first
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [batch_size, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class GestureCTCModel(nn.Module):
    """A Transformer Encoder model for CTC-based gesture recognition."""
    def __init__(self, input_dim: int, num_classes: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.ctc_head = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch_size, seq_len, input_dim)
            src_key_padding_mask: (batch_size, seq_len)
        Returns:
            log_probs: (seq_len, batch_size, num_classes) for CTCLoss
        """
        # FIX 4: Process tensors consistently as batch_first
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.ctc_head(output)
        # CTCLoss expects (seq_len, batch, num_classes)
        return F.log_softmax(logits.permute(1, 0, 2), dim=2)

# --- 5. Training and Evaluation Loop ---
def train_model():
    print("--- Initializing Production Training Pipeline ---")
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    os.makedirs(CONFIG["export_dir"], exist_ok=True)

    tokenizer = CharTokenizer(CONFIG["chars"])
    grid = KeyboardGrid(CONFIG["chars"])
    featurizer = SwipeFeaturizer(grid)
    
    # FIX 3: Correctly calculate input_dim based on SwipeFeaturizer output
    input_dim = 8 + grid.num_keys
    
    print("Loading datasets...")
    train_dataset = SwipeDataset(CONFIG["train_data_path"], featurizer, tokenizer, CONFIG["max_seq_len"])
    val_dataset = SwipeDataset(CONFIG["val_data_path"], featurizer, tokenizer, CONFIG["max_seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=True)

    model = GestureCTCModel(
        input_dim=input_dim, num_classes=tokenizer.vocab_size, d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"], num_encoder_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"], dropout=CONFIG["dropout"]
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    criterion = nn.CTCLoss(blank=tokenizer.char_to_int['<blank>'], reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # FIX 2A: Use modern torch.amp.GradScaler API
    scaler = torch.amp.GradScaler(device, enabled=CONFIG["mixed_precision"])
    
    writer = SummaryWriter(CONFIG["log_dir"])

    print("Setting up CTC decoder with KenLM...")
    with open(CONFIG["vocab_path"], "r") as f:
        vocab_list = [word.strip() for word in f.readlines()]
    labels = [tokenizer.int_to_char[i] for i in range(tokenizer.vocab_size)]
    
    # FIX 1: Use the corrected repo and filename
    lm_path = hf_hub_download(repo_id=CONFIG["kenlm_repo_id"], filename=CONFIG["kenlm_filename"])
    decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=CONFIG["kenlm_alpha"], beta=CONFIG["kenlm_beta"])
    print("Decoder setup complete.")
    
    best_val_wer = float('inf')
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        
        model.train()
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            if batch is None: continue
            features, targets = batch["features"].to(device), batch["targets"].to(device)
            feature_lengths, target_lengths = batch["feature_lengths"].to(device), batch["target_lengths"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # FIX 2B: Use modern torch.autocast API
            with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=CONFIG["mixed_precision"]):
                src_key_padding_mask = (torch.arange(features.shape[1], device=device)[None, :] >= feature_lengths[:, None])
                log_probs = model(features, src_key_padding_mask)
                loss = criterion(log_probs, targets, feature_lengths, target_lengths)
            
            scaler.scale(loss).backward()
            # FIX 7: Add gradient clipping after backward pass
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + pbar.n)
            # FIX 6: Schedulers (like OneCycleLR) should step after the optimizer
            # No scheduler is used here, but if one were, it would go here.

        model.eval()
        val_wer_total, val_word_total = 0, 0
        pbar_val = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for batch in pbar_val:
                if batch is None: continue
                features, feature_lengths, true_words = batch["features"].to(device), batch["feature_lengths"].to(device), batch["words"]

                with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=CONFIG["mixed_precision"]):
                    src_key_padding_mask = (torch.arange(features.shape[1], device=device)[None, :] >= feature_lengths[:, None])
                    log_probs = model(features, src_key_padding_mask)
                
                logits_cpu = log_probs.permute(1, 0, 2).cpu().float().numpy()
                decoded_words = decoder.decode_batch(logits_cpu, beam_width=CONFIG["beam_width"])

                for pred_word, true_word in zip(decoded_words, true_words):
                    if pred_word != true_word:
                        val_wer_total += 1
                    val_word_total += 1
                
                current_wer = val_wer_total / val_word_total if val_word_total > 0 else 0
                pbar_val.set_postfix({'WER': f'{current_wer:.2%}'})

        avg_val_wer = val_wer_total / val_word_total if val_word_total > 0 else 1.0
        writer.add_scalar('WER/validation', avg_val_wer, epoch)
        print(f"Epoch {epoch + 1} | Validation WER: {avg_val_wer:.2%}")

        if avg_val_wer < best_val_wer:
            best_val_wer = avg_val_wer
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with WER: {best_val_wer:.2%} to {checkpoint_path}")

    print(f"\n--- Training Complete --- Best Validation WER: {best_val_wer:.2%}")

if __name__ == "__main__":
    train_model()
