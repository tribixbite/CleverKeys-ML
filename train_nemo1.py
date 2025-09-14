# train_conformer.py
# A production-ready script to train a Conformer-Transducer model for gesture typing.

import os
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf

# --- Library Version Checks (ensure a reproducible environment) ---
# Import necessary libraries and specify their versions for clarity and stability.
# These versions correspond to the latest stable releases as of late 2025.
import nemo.collections.asr as nemo_asr # version 2.4.0
from nemo.utils import logging

# --- Configuration ---
# All hyperparameters and paths are defined here for easy modification.
CONFIG = {
    "data": {
        "train_manifest": "data/train_final_train.jsonl",
        "val_manifest": "data/train_final_val.jsonl",
        "vocab_path": "data/vocab.txt",
        "max_trace_len": 200, # Maximum number of points in a swipe trace
        "max_word_len": 30,   # Maximum number of characters in a word
    },
    "training": {
        "batch_size": 128,
        "num_workers": 8,
        "learning_rate": 3e-4,
        "max_epochs": 100,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": torch.cuda.device_count() if torch.cuda.is_available() else 1,
        "precision": "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 32,
    },
    "model": {
        "encoder": {
            "d_model": 256,       # Model dimension
            "n_heads": 4,         # Number of attention heads
            "num_layers": 8,      # Number of Conformer blocks
            "feat_in": 3,         # Input features (x, y, t)
        },
        "decoder": {
            "pred_hidden": 320,   # Hidden size of the prediction network (LSTM)
        },
        "joint": {
            "joint_hidden": 320,  # Hidden size of the joint network
        }
    }
}

# --- Custom Dataset for Swipe Traces ---
# This class handles loading and preprocessing of the gesture data from the.jsonl files.
class SwipeDataset(Dataset):
    def __init__(self, manifest_path, vocab, max_trace_len, max_word_len):
        """
        Initializes the dataset.
        Args:
            manifest_path (str): Path to the.jsonl manifest file.
            vocab (dict): A dictionary mapping characters to integer IDs.
            max_trace_len (int): Maximum length to pad gesture traces to.
            max_word_len (int): Maximum length to pad target words to.
        """
        super().__init__()
        self.vocab = vocab
        self.max_trace_len = max_trace_len
        self.max_word_len = max_word_len
        self.data = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        # Returns the total number of samples in the dataset.
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves and preprocesses a single sample from the dataset.
        item = self.data[idx]
        
        # 1. Process the gesture trace
        trace = torch.tensor(item['trace'], dtype=torch.float32)
        trace_len = torch.tensor(trace.shape, dtype=torch.long)
        
        # 2. Process the target word
        word = item['word']
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in word]
        tokens = torch.tensor(tokens, dtype=torch.long)
        word_len = torch.tensor(len(tokens), dtype=torch.long)
        
        return trace, trace_len, tokens, word_len

# --- Collate Function for Batching ---
# This function takes a list of samples and pads them to create uniform batches.
def collate_fn(batch, max_trace_len, max_word_len):
    """
    Pads traces and tokens to the max length in the batch.
    This is essential for efficient batch processing on the GPU.
    """
    traces, trace_lens, tokens, word_lens = zip(*batch)
    
    # Pad traces to the max_trace_len defined in config
    padded_traces = torch.zeros(len(traces), max_trace_len, traces.shape)
    for i, trace in enumerate(traces):
        length = min(trace.shape, max_trace_len)
        padded_traces[i, :length, :] = trace[:length]
        
    # Pad tokens to the max_word_len defined in config
    padded_tokens = torch.zeros(len(tokens), max_word_len, dtype=torch.long)
    for i, token_seq in enumerate(tokens):
        length = min(token_seq.shape, max_word_len)
        padded_tokens[i, :length] = token_seq[:length]
        
    return (
        padded_traces,
        torch.stack(trace_lens),
        padded_tokens,
        torch.stack(word_lens)
    )

# --- Main Training Function ---
def main():
    # Entry point for the training script.
    cfg = DictConfig(CONFIG)
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Load Vocabulary
    # The vocabulary maps each character to a unique integer ID.
    vocab = {}
    with open(cfg.data.vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    vocab_size = len(vocab)
    logging.info(f"Vocabulary loaded with {vocab_size} tokens.")

    # 2. Setup DataLoaders
    # PyTorch DataLoaders handle batching, shuffling, and multi-threaded data loading.
    train_dataset = SwipeDataset(
        manifest_path=cfg.data.train_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        max_word_len=cfg.data.max_word_len
    )
    val_dataset = SwipeDataset(
        manifest_path=cfg.data.val_manifest,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len,
        max_word_len=cfg.data.max_word_len
    )
    
    # The collate function is passed to the DataLoader to handle padding.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=lambda b: collate_fn(b, cfg.data.max_trace_len, cfg.data.max_word_len),
        num_workers=cfg.training.num_workers,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=lambda b: collate_fn(b, cfg.data.max_trace_len, cfg.data.max_word_len),
        num_workers=cfg.training.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # 3. Configure the Conformer-Transducer Model using NeMo's config system.
    # This defines the entire architecture, from the encoder to the loss function.
    model_cfg = DictConfig({
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
            'dropout_pre_encoder': 0.1,
            'dropout_emb': 0.0,
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
            '_target_': 'nemo.collections.asr.losses.transducer.TransducerLoss',
            'loss_name': 'warprnnt_numba',
            'blank_index': 0, # <blank> token must be at index 0
        }
    })

    # 4. Instantiate the Model and Trainer
    # We use NeMo's EncDecRNNTBPEModel, which is a generic class for Transducer models.
    model = nemo_asr.models.EncDecRNNTBPEModel(cfg=model_cfg)
    model.setup_training_data(train_data_config=None) # We will manage dataloaders manually
    model.setup_validation_data(val_data_config=None)
    
    # PyTorch Lightning Trainer handles the training loop, checkpointing, and hardware acceleration.
    trainer = pl.Trainer(
        devices=cfg.training.devices,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        log_every_n_steps=100,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    # 5. Start Training
    logging.info("Starting model training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logging.info("Training complete.")

    # 6. Save the final model
    # The trained model is saved as a.nemo file, which is a tarball containing the config and weights.
    save_path = "swipe_conformer_transducer.nemo"
    model.save_to(save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()