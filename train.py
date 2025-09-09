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
import time
from typing import List, Dict

# --- Configuration Block ---
# All hyperparameters and paths are centralized here for easy modification.
CONFIG = {
    # Data and Vocabulary
    "train_data_path": "path/to/your/500k_traces_train.jsonl",
    "val_data_path": "path/to/your/traces_val.jsonl",
    "vocab_path": "path/to/your/153k_word_vocab.txt",
    "chars": "abcdefghijklmnopqrstuvwxyz'", # Includes apostrophe

    # Model Architecture
    "d_model": 256,          # Embedding dimension (increased for more capacity)
    "nhead": 4,              # Number of attention heads
    "num_encoder_layers": 6, # Number of Transformer encoder layers
    "dim_feedforward": 1024, # Hidden dimension in feedforward networks
    "dropout": 0.1,

    # Training Parameters
    "batch_size": 256,       # Increased for 4090 VRAM
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "warmup_steps": 4000,
    "mixed_precision": True, # Use AMP (Automatic Mixed Precision) for speed

    # Decoding and Evaluation
    "beam_width": 100,       # Beam width for CTC decoding
    "kenlm_model_name": "kenlm-fg-small", # A good general-purpose LM from Hugging Face
    "kenlm_alpha": 0.5,      # Weight for the language model
    "kenlm_beta": 1.5,       # Weight for word insertion

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
        # CTC blank token must be at index 0
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
        # A simple heuristic for QWERTY layout
        rows = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ]
        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                self.key_positions[char] = (c + r * 0.5, r)
        self.key_positions["'"] = (9.5, 1.0) # Approximate position
        self.key_names = list(self.key_positions.keys())
        self.key_coords = np.array(list(self.key_positions.values()))
        self.num_keys = len(self.key_names)

class SwipeFeaturizer:
    """Processes raw swipe data into rich feature vectors."""
    def __init__(self, grid: KeyboardGrid):
        self.grid = grid

    def __call__(self, points: List[Dict[str, float]]) -> np.ndarray:
        if len(points) < 2:
            # Handle edge case of very short swipes
            points.append(points[0])

        coords = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        times = np.array([p['t'] for p in points], dtype=np.float32).reshape(-1, 1)

        # Kinematic features
        delta_coords = np.diff(coords, axis=0, prepend=coords[0:1,:])
        delta_times = np.diff(times, axis=0, prepend=times[0:1,:])
        delta_times[delta_times == 0] = 1e-6 # Avoid division by zero
        velocity = delta_coords / delta_times
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1,:]) / delta_times

        # Spatial features: nearest keys
        dist_sq = np.sum((coords[:, np.newaxis, :] - self.grid.key_coords) ** 2, axis=-1)
        nearest_key_indices = np.argmin(dist_sq, axis=1)
        nearest_keys_onehot = np.eye(self.grid.num_keys, dtype=np.float32)[nearest_key_indices]

        features = np.concatenate([
            coords,
            delta_coords,
            velocity,
            acceleration,
            nearest_keys_onehot
        ], axis=1)

        return features

# --- 3. Dataset and Dataloader ---
class SwipeDataset(Dataset):
    def __init__(self, jsonl_path: str, featurizer: SwipeFeaturizer, tokenizer: CharTokenizer):
        self.featurizer = featurizer
        self.tokenizer = tokenizer
        with open(jsonl_path, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        data = json.loads(line)
        word = data['word'].lower()
        
        # Filter out words with characters not in our vocabulary
        if not all(c in self.tokenizer.char_to_int for c in word):
            # Return a dummy sample, which will be filtered by collate_fn
            return None

        features = self.featurizer(data['points'])
        target = self.tokenizer.encode(word)
        
        return {
            "features": torch.FloatTensor(features),
            "target": torch.LongTensor(target),
            "word": word
        }

def collate_fn(batch):
    """Pads sequences for batching and handles filtering of invalid samples."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    features = [item['features'] for item in batch]
    targets = [item['target'] for item in batch]
    words = [item['word'] for item in batch]

    feature_lengths = torch.LongTensor([len(f) for f in features])
    target_lengths = torch.LongTensor([len(t) for t in targets])

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return {
        "features": padded_features,
        "targets": padded_targets,
        "feature_lengths": feature_lengths,
        "target_lengths": target_lengths,
        "words": words
    }

# --- 4. Model Architecture (CTC-based) ---
class PositionalEncoding(nn.Module):
    """Standard Transformer positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
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
            log_probs: (seq_len, batch_size, num_classes)
        """
        x = self.input_projection(src)
        # TransformerEncoder expects (seq_len, batch, dim) if batch_first=False
        x = self.pos_encoder(x.permute(1, 0, 2))
        # TransformerEncoder expects padding mask (batch, seq_len)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.ctc_head(output)
        # CTCLoss expects (seq_len, batch, num_classes) and log_softmax
        return F.log_softmax(logits, dim=2)


# --- 5. Training and Evaluation Loop ---
def train_model():
    """Main function to orchestrate the training process."""
    print("--- Initializing Production Training Pipeline ---")
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # Setup directories
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    os.makedirs(CONFIG["export_dir"], exist_ok=True)

    # Initialize components
    tokenizer = CharTokenizer(CONFIG["chars"])
    grid = KeyboardGrid(CONFIG["chars"])
    featurizer = SwipeFeaturizer(grid)
    input_dim = 10 + grid.num_keys # x,y,t,dx,dy,dt,vx,vy,ax,ay + onehot keys
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    train_dataset = SwipeDataset(CONFIG["train_data_path"], featurizer, tokenizer)
    val_dataset = SwipeDataset(CONFIG["val_data_path"], featurizer, tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False,
        num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=True
    )

    # Initialize model, loss, and optimizer
    model = GestureCTCModel(
        input_dim=input_dim,
        num_classes=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"]
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    criterion = nn.CTCLoss(blank=tokenizer.char_to_int['<blank>'], reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["learning_rate"],
        steps_per_epoch=len(train_loader), epochs=CONFIG["num_epochs"],
        pct_start=CONFIG["warmup_steps"]/(len(train_loader)*CONFIG["num_epochs"])
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["mixed_precision"])

    # Setup TensorBoard
    writer = SummaryWriter(CONFIG["log_dir"])

    # Prepare CTC Decoder with external language model
    print("Setting up CTC decoder with KenLM...")
    with open(CONFIG["vocab_path"], "r") as f:
        vocab_list = [word.strip() for word in f.readlines()]
    
    labels = [tokenizer.int_to_char[i] for i in range(tokenizer.vocab_size)]
    
    # Download KenLM model from Hugging Face Hub
    lm_path = hf_hub_download(repo_id="kensho/kenlm", filename=f"lm/en_us/4-gram-small.arpa",)

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        alpha=CONFIG["kenlm_alpha"],
        beta=CONFIG["kenlm_beta"],
    )
    print("Decoder setup complete.")
    
    # Training Loop
    best_val_wer = float('inf')
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            if batch is None: continue
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=CONFIG["mixed_precision"]):
                # Create padding mask (True for padded values)
                src_key_padding_mask = (torch.arange(features.shape[1])[None, :].to(device) >= feature_lengths[:, None])
                log_probs = model(features, src_key_padding_mask)
                loss = criterion(log_probs, targets, feature_lengths, target_lengths)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + pbar.n)
            writer.add_scalar('LR/step', scheduler.get_last_lr()[0], epoch * len(train_loader) + pbar.n)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # --- Validation Phase ---
        model.eval()
        val_wer_total = 0
        val_word_total = 0
        pbar_val = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for batch in pbar_val:
                if batch is None: continue
                features = batch["features"].to(device)
                feature_lengths = batch["feature_lengths"].to(device)
                true_words = batch["words"]

                with torch.cuda.amp.autocast(enabled=CONFIG["mixed_precision"]):
                    src_key_padding_mask = (torch.arange(features.shape[1])[None, :].to(device) >= feature_lengths[:, None])
                    log_probs = model(features, src_key_padding_mask)

                # Move to CPU for decoding, transpose to (batch, seq, class)
                logits_cpu = log_probs.permute(1, 0, 2).cpu().numpy()
                
                # Use pyctcdecode for strong decoding
                decoded_words = decoder.decode_batch(logits_cpu, beam_width=CONFIG["beam_width"])

                for pred_word, true_word in zip(decoded_words, true_words):
                    # Simple Word Error Rate calculation
                    if pred_word != true_word:
                        val_wer_total += 1
                    val_word_total += 1
                
                current_wer = val_wer_total / val_word_total if val_word_total > 0 else 0
                pbar_val.set_postfix({'WER': f'{current_wer:.2%}'})

        avg_val_wer = val_wer_total / val_word_total if val_word_total > 0 else 1.0
        writer.add_scalar('WER/validation', avg_val_wer, epoch)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Validation WER: {avg_val_wer:.2%}")

        # Save checkpoint if it's the best one
        if avg_val_wer < best_val_wer:
            best_val_wer = avg_val_wer
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with WER: {best_val_wer:.2%} to {checkpoint_path}")

    print("\n--- Training Complete ---")
    print(f"Best Validation WER: {best_val_wer:.2%}")

    # --- 6. Exporting for Production ---
    print("\n--- Exporting Model for Production ---")
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")))
    model.eval()

    # Create a dummy input for tracing/scripting
    dummy_input = torch.randn(1, 100, input_dim).to(device) # (batch, seq_len, features)
    dummy_mask = torch.zeros(1, 100, dtype=torch.bool).to(device) # No padding

    # 1. Export to TorchScript (a prerequisite for ExecuTorch)
    try:
        traced_model = torch.jit.trace(model, (dummy_input, dummy_mask))
        torchscript_path = os.path.join(CONFIG["export_dir"], "model.pt")
        traced_model.save(torchscript_path)
        print(f"Successfully exported to TorchScript: {torchscript_path}")
        print("Next step: Use the ExecuTorch toolchain to convert this .pt file to a .pte file for Android.")
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")

    # 2. Export to ONNX (for browser deployment)
    try:
        onnx_path = os.path.join(CONFIG["export_dir"], "model.onnx")
        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            onnx_path,
            input_names=['features', 'padding_mask'],
            output_names=['log_probs'],
            opset_version=14,
            dynamic_axes={'features' : {0 : 'batch_size', 1 : 'sequence_length'},
                          'padding_mask': {0: 'batch_size', 1: 'sequence_length'},
                          'log_probs' : {1 : 'batch_size', 0 : 'sequence_length'}}
        )
        print(f"Successfully exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

if __name__ == "__main__":
    train_model()
