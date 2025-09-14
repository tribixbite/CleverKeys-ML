# train_conformer.py
# A production-ready script to train a Conformer-Transducer model for gesture typing.
# This version includes robust feature engineering, corrected data loading, and enhanced logging.

import os
import json
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any

# --- Library Version Checks (ensure a reproducible environment) ---
# Import necessary libraries and specify their versions for clarity and stability.
# These versions correspond to the latest stable releases as of late 2025.
import nemo.collections.asr as nemo_asr # version 2.4.0
from nemo.utils import logging, model_utils

# --- Configuration ---
# All hyperparameters and paths are defined here for easy modification.
CONFIG = {
    "data": {
        "train_manifest": "data/train_final_train.jsonl",
        "val_manifest": "data/train_final_val.jsonl",
        "vocab_path": "data/vocab.txt",
        "chars": "abcdefghijklmnopqrstuvwxyz'",
        "max_trace_len": 200, # Maximum number of points in a swipe trace
    },
    "training": {
        "batch_size": 128,
        "num_workers": 8,
        "learning_rate": 3e-4,
        "max_epochs": 100,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": torch.cuda.device_count() if torch.cuda.is_available() else 1,
        # Use 'bf16-mixed' for modern GPUs (NVIDIA 30-series and newer) to leverage Tensor Cores.
        # This provides significant speedup with minimal precision loss.
        "precision": "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 32,
    },
    "model": {
        "encoder": {
            # The input feature dimension is determined by the SwipeFeaturizer.
            # (x, y, dx, dy, vel_x, vel_y, accel_x, accel_y, angle) = 9 features
            # + one-hot encoding of nearest key (28 keys for a-z and '). Total = 9 + 28 = 37
            "feat_in": 37,
            "d_model": 256,       # Model dimension
            "n_heads": 4,         # Number of attention heads
            "num_layers": 8,      # Number of Conformer blocks
        },
        "decoder": {
            "pred_hidden": 320,   # Hidden size of the prediction network (LSTM)
        },
        "joint": {
            "joint_hidden": 320,  # Hidden size of the joint network
        }
    }
}

# --- Feature Engineering ---
# These classes handle the conversion of raw swipe coordinates into rich features for the model.
class KeyboardGrid:
    """Represents the QWERTY keyboard layout to calculate key proximity."""
    def __init__(self, chars: str):
        self.key_pos: Dict[str, tuple] = {}
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for r, row in enumerate(rows):
            for c, char in enumerate(row):
                self.key_pos[char] = ((c + r * 0.5) / 10.0, r / 3.0)
        self.key_pos["'"] = (9.5 / 10.0, 1.0 / 3.0)
        self.key_coords = np.array(list(self.key_pos.values()), dtype=np.float32)
        self.num_keys = len(self.key_pos)

class SwipeFeaturizer:
    """
    Converts a raw list of swipe points into a rich feature tensor.
    Features include velocity, acceleration, angle, and proximity to keyboard keys.
    """
    def __init__(self, grid: KeyboardGrid):
        self.grid = grid

    def __call__(self, points: List) -> np.ndarray:
        if len(points) < 2:
            points = points + [points[-1]] # Duplicate the last point if trace is too short

        coords = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        times = np.array([[p['t']] for p in points], dtype=np.float32)

        # Calculate derivatives: velocity and acceleration
        d_coords = np.diff(coords, axis=0, prepend=coords[0:1, :])
        d_times = np.diff(times, axis=0, prepend=times[0:1, :])
        d_times[d_times == 0] = 1e-6 # Avoid division by zero
        
        vel = np.clip(d_coords / d_times, -10, 10)
        accel = np.clip(np.diff(vel, axis=0, prepend=vel[0:1, :]) / d_times, -10, 10)
        
        # Calculate gesture angle
        angle = np.arctan2(d_coords[:, 1], d_coords[:, 0]).reshape(-1, 1)

        # Calculate proximity to each key and create a one-hot vector for the nearest key
        dist_sq = np.sum((coords[:, None, :] - self.grid.key_coords) ** 2, axis=-1)
        nearest_key_idx = np.argmin(dist_sq, axis=1)
        keys_onehot = np.eye(self.grid.num_keys, dtype=np.float32)[nearest_key_idx]

        # Concatenate all features
        features = np.concatenate([coords, d_coords, vel, accel, angle, keys_onehot], axis=1)
        return np.nan_to_num(np.clip(features, -10.0, 10.0), nan=0.0)

# --- Custom Dataset for Swipe Traces ---
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

# --- Collate Function for Batching ---
def collate_fn(batch):
    """Pads traces and tokens to create uniform batches."""
    features, feature_lengths, tokens, token_lengths = zip(*batch)
    
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    
    return (
        padded_features,
        torch.stack(feature_lengths),
        padded_tokens,
        torch.stack(token_lengths)
    )

# --- Enhanced Logging Callback ---
class PredictionLogger(pl.Callback):
    """Logs a few validation predictions to TensorBoard at the end of each epoch."""
    def __init__(self, val_dataloader, vocab):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.vocab = vocab
        # Create reverse vocab for decoding
        self.idx_to_char = {idx: char for char, idx in vocab.items()}

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the TensorBoard logger
        logger = trainer.logger.experiment
        
        # Get a single batch from the validation set
        batch = next(iter(self.val_dataloader))
        features, feature_lengths, tokens, token_lengths = batch
        features = features.to(pl_module.device)
        feature_lengths = feature_lengths.to(pl_module.device)

        # Get model predictions
        with torch.no_grad():
            pl_module.eval()
            log_probs, encoded_len, _, _ = pl_module(
                input_signal=features, input_signal_length=feature_lengths
            )
            predictions = pl_module.decoding.rnnt_decoder_predictions_tensor(
                log_probs, encoded_len, return_hypotheses=False
            )
        
        # Log up to 5 examples
        log_text = "## Predictions vs. Ground Truth\n\n| Prediction | Ground Truth |\n|---|---|\n"
        for i in range(min(5, len(predictions))):
            pred_text = predictions[i]
            true_text = "".join([self.idx_to_char.get(id.item(), '') for id in tokens[i] if id.item() != 0])
            log_text += f"| `{pred_text}` | `{true_text}` |\n"
            
        logger.add_text("validation_predictions", log_text, global_step=trainer.current_epoch)

# --- Main Training Function ---
def main():
    cfg = DictConfig(CONFIG)
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Load Vocabulary
    vocab = {}
    with open(cfg.data.vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    vocab_size = len(vocab)
    logging.info(f"Vocabulary loaded with {vocab_size} tokens.")

    # 2. Setup Feature Engineering and DataLoaders
    grid = KeyboardGrid(cfg.data.chars)
    featurizer = SwipeFeaturizer(grid)
    
    train_dataset = SwipeDataset(
        manifest_path=cfg.data.train_manifest,
        featurizer=featurizer,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len
    )
    val_dataset = SwipeDataset(
        manifest_path=cfg.data.val_manifest,
        featurizer=featurizer,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # 3. Configure the Conformer-Transducer Model using NeMo's config system.
    model_cfg = DictConfig({
        # Required preprocessor config (we'll bypass it but NeMo needs it)
        'preprocessor': {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
            'sample_rate': 16000,  # Dummy value, we don't use audio
            'normalize': 'per_feature',
            'window_size': 0.025,
            'window_stride': 0.01,
            'features': cfg.model.encoder.feat_in,
            'n_fft': 512,
            'frame_splicing': 1,
            'dither': 0.00001,
        },
        'encoder': {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': cfg.model.encoder.feat_in,
            'n_layers': cfg.model.encoder.num_layers,
            'd_model': cfg.model.encoder.d_model,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': cfg.model.encoder.n_heads,
            'conv_kernel_size': 31,
            'dropout': 0.1,
        },
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
            'pred_hidden': cfg.model.decoder.pred_hidden,
            'in_features': vocab_size,
            'pred_rnn_layers': 1,
        },
        'joint': {
            '_target_': 'nemo.collections.asr.modules.RNNTJoint',
            'joint_hidden': cfg.model.joint.joint_hidden,
            'in_features': cfg.model.encoder.d_model + cfg.model.decoder.pred_hidden,
            'out_features': vocab_size,
        },
        'optim': {
            'name': 'adamw',
            'lr': cfg.training.learning_rate,
            'betas': [0.9, 0.98],
            'weight_decay': 1e-3,
            'sched': {
                'name': 'CosineAnnealing',
                'warmup_steps': 1000,
                'min_lr': 1e-6,
            }
        },
        'loss': {
            '_target_': 'nemo.collections.asr.losses.rnnt.RNNTLoss',
            'blank_idx': 0,  # blank token at index 0
        },
        
        # Required data configurations (even though we override with custom loaders)
        'train_ds': {
            'manifest_filepath': cfg.data.train_manifest,
            'sample_rate': 16000,  # Required field, dummy value for non-audio data
            'labels': list(vocab.keys()),  # Required: vocabulary labels
            'batch_size': cfg.training.batch_size,
            'shuffle': True,
        },
        'validation_ds': {
            'manifest_filepath': cfg.data.val_manifest,
            'sample_rate': 16000,  # Required field, dummy value for non-audio data
            'labels': list(vocab.keys()),  # Required: vocabulary labels
            'batch_size': cfg.training.batch_size,
            'shuffle': False,
        },
    })

    # 4. Instantiate the Model and Trainer
    # Use the proper EncDecRNNTModel which implements full RNN-T with:
    # - Encoder: Processes input features
    # - Decoder (Prediction Network): Models P(y_i | y_1...y_{i-1})
    # - Joint Network: Combines encoder and decoder outputs
    model = nemo_asr.models.EncDecRNNTModel(cfg=model_cfg)
    
    # Pass our custom datasets to the model
    model.setup_training_data(train_data_config=None)
    model.setup_validation_data(val_data_config=None)
    model._train_dl = train_loader
    model._validation_dl = val_loader
    
    # Instantiate the logging callback
    prediction_logger = PredictionLogger(val_loader, vocab)
    
    trainer = pl.Trainer(
        devices=cfg.training.devices,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        log_every_n_steps=100,
        callbacks=[prediction_logger]
    )
    
    # CRITICAL FIX: Attach the trainer to the model. This is required by NeMo
    # to properly handle distributed training and other framework integrations.
    model.set_trainer(trainer)

    # 5. Start Training
    logging.info("Starting model training...")
    trainer.fit(model)
    logging.info("Training complete.")

    # 6. Save the final model
    save_path = "swipe_conformer_transducer.nemo"
    model.save_to(save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()