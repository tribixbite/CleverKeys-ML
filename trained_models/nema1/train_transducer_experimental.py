#!/usr/bin/env python3
"""
NeMo RNN-Transducer training script using EncDecRNNTModel.
Uses proper RNN-T joint computation for modeling output dependencies.
Optimized for RTX 4090M with all performance enhancements.
"""

import datetime as dt
import os
import glob
import json
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any
import logging

# NeMo imports
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import feature engineering utilities
from swipe_data_utils import (
    KeyboardGrid,
    SwipeFeaturizer,
    SwipeDataset,
    collate_fn
)


class PredictionLogger(pl.Callback):
    """Logs sample predictions during validation."""
    def __init__(self, val_loader, vocab):
        self.val_loader = val_loader
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:  # Only log first batch
            # Get predictions (simplified - just log batch info)
            features, feature_lengths, tokens, token_lengths = batch
            
            # Log some statistics
            logger.info(f"Validation batch {batch_idx}: "
                       f"Features shape: {features.shape}, "
                       f"Tokens shape: {tokens.shape}")

# Configuration optimized for RTX 4090M
CONFIG = {
    "data": {
        "train_manifest": "data/train_final_train.jsonl",
        "val_manifest": "data/train_final_val.jsonl",
        "vocab_path": "data/vocab.txt",
        "chars": "abcdefghijklmnopqrstuvwxyz'",
        "max_trace_len": 200,
    },
    "training": {
        "batch_size": 384,  # Increased to 384 for better GPU utilization
        "num_workers": 24,  # Increased for better data loading
        "learning_rate": 4e-4,  # Keep LR the same to maintain quality
        "max_epochs": 100,
        "gradient_accumulation": 1,  # Keep at 1 since batch size is larger
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",  # Use bf16 instead of fp32 to avoid CUDA graph dtype issues
    },
    "model": {
        "encoder": {
            "feat_in": 37,  # 9 kinematic + 28 keys
            "d_model": 256,  # 256 is the original model dimension
            "n_heads": 4,
            "num_layers": 8,  # Use 8 layers for better accuracy
            "conv_kernel_size": 31,  # 31 is the original kernel size
            "subsampling_factor": 2,  # 2x subsampling for speed, 4 is the original subsampling factor
        },
        "decoder": {
            "pred_hidden": 320,  # 320 is the original hidden size of the prediction network
            "pred_rnn_layers": 2,  # Multiple layers for better dependency modeling, 1 is the original number of layers
        },
        "joint": {
            "joint_hidden": 512,  # 512 is the original hidden size of the joint network
            "activation": "relu",
            "dropout": 0.1,  # 0.1 is the original dropout rate for the joint network
        }
    }
}

runtime_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def find_latest_checkpoint():
    """Find the most recent checkpoint across all timestamped folders.

    Returns:
        tuple: (checkpoint_path, is_old_format) where is_old_format
               indicates if checkpoint is from older training run
    """
    # Look for checkpoint files in both timestamped and non-timestamped directories
    checkpoint_patterns = [
        './rnnt_logs/conformer_rnnt/*/checkpoints/*.ckpt',
        './rnnt_logs_*/conformer_rnnt/*/checkpoints/*.ckpt',
        './rnnt_checkpoints_*/conformer_rnnt/*/checkpoints/*.ckpt',
    ]

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))

    if not all_checkpoints:
        return None, False

    # Get the most recently modified checkpoint
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)

    # Check if this is an old checkpoint with different settings
    is_old_format = False
    if 'rnnt_logs/' in latest_checkpoint and '/version_' in latest_checkpoint:
        # These are older checkpoints, likely with 'default' loss
        is_old_format = True

    return latest_checkpoint, is_old_format

def main():
    """Main training function using NeMo's EncDecRNNTModel."""
    cfg = DictConfig(CONFIG)

    # Check for existing checkpoint to resume from
    resume_checkpoint, is_old_format = find_latest_checkpoint()

    if resume_checkpoint and is_old_format:
        logger.info("="*60)
        logger.info("⚠️  Resuming from checkpoint with different settings:")
        logger.info(f"  Checkpoint: {resume_checkpoint}")
        logger.info(f"  Modified: {dt.datetime.fromtimestamp(os.path.getmtime(resume_checkpoint))}")
        logger.info(f"  Note: This checkpoint used 'default' loss, now using 'warprnnt_numba'")
        logger.info(f"  Note: Model will be recompiled (expect initial slowdown)")
        logger.info("="*60)
    elif resume_checkpoint:
        logger.info("="*60)
        logger.info(f"Found checkpoint to resume from:")
        logger.info(f"  {resume_checkpoint}")
        logger.info(f"  Modified: {dt.datetime.fromtimestamp(os.path.getmtime(resume_checkpoint))}")
        logger.info("="*60)
    else:
        logger.info("="*60)
        logger.info("Starting fresh training (no checkpoint found)")
        logger.info("="*60)

    logger.info("RNN-Transducer Training with NeMo EncDecRNNTModel")
    logger.info("Models output dependencies: P(y_i | y_1...y_{i-1}, x)")
    logger.info("="*60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Enable RTX 4090M optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ Enabled TF32 and cuDNN optimizations for RTX 4090M")

        # Enable SDPA/FlashAttention optimizations for attention kernels
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slower math implementation
            logger.info("✓ Enabled SDPA Flash/Memory-Efficient attention optimizations")
        except Exception as e:
            logger.warning(f"Could not enable SDPA optimizations: {e}")
    
    # Load vocabulary
    vocab = {}
    with open(cfg.data.vocab_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    vocab_size = len(vocab)
    logger.info(f"Vocabulary loaded with {vocab_size} tokens")
    
    # Setup feature engineering
    grid = KeyboardGrid(cfg.data.chars)
    featurizer = SwipeFeaturizer(grid)
    
    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = SwipeDataset(
        manifest_path=cfg.data.train_manifest,
        featurizer=featurizer,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len
    )
    logger.info(f"Training dataset created with {len(train_dataset)} samples")
    
    logger.info("Creating validation dataset...")
    val_dataset = SwipeDataset(
        manifest_path=cfg.data.val_manifest,
        featurizer=featurizer,
        vocab=vocab,
        max_trace_len=cfg.data.max_trace_len
    )
    logger.info(f"Validation dataset created with {len(val_dataset)} samples")
    
    # Create dataloaders with optimizations
    logger.info("Creating training dataloader...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,  # Increased for better prefetching
        drop_last=True      # Drop last incomplete batch for uniform shapes
    )
    logger.info(f"Training dataloader created with batch_size={cfg.training.batch_size}")
    
    logger.info("Creating validation dataloader...")
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.training.batch_size * 2,  # Reasonable validation batch size
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,  # Increased for better prefetching
        drop_last=True
    )
    logger.info(f"Validation dataloader created with batch_size={cfg.training.batch_size * 2}")
    
    # Configure the full EncDecRNNTModel with proper RNN-T components
    model_config = DictConfig({
        # Vocabulary labels and required fields
        'labels': list(vocab.keys()),
        'sample_rate': 16000,  # Required by NeMo even though we don't use audio
        
        # Model defaults required by NeMo
        'model_defaults': {
            'enc_hidden': cfg.model.encoder.d_model,
            'pred_hidden': cfg.model.decoder.pred_hidden,
        },
        
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
        
        # Conformer Encoder Configuration
        'encoder': {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': cfg.model.encoder.feat_in,
            'n_layers': cfg.model.encoder.num_layers,
            'd_model': cfg.model.encoder.d_model,
            'feat_out': -1,  # Set to -1 to use d_model
            
            # Conformer specific parameters
            'subsampling': 'striding',  # or 'stacking', 'vggnet'
            'subsampling_factor': cfg.model.encoder.subsampling_factor,  # 2x subsampling
            'subsampling_conv_channels': -1,  # Use d_model
            'ff_expansion_factor': 4,
            
            # Multi-head attention
            'self_attention_model': 'rel_pos',  # Relative positional encoding
            'n_heads': cfg.model.encoder.n_heads,
            'att_context_size': [-1, -1],  # Unlimited context
            'xscaling': True,  # Learnable scaling
            'untie_biases': True,
            'pos_emb_max_len': 5000,
            
            # Convolution module
            'conv_kernel_size': cfg.model.encoder.conv_kernel_size,
            'conv_norm_type': 'batch_norm',
            
            # Regularization
            'dropout': 0.1,
            'dropout_emb': 0.1,  # Add embedding dropout for better generalization
            'dropout_att': 0.1,
        },
        
        # RNN-T Decoder (Prediction Network)
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.rnnt.RNNTDecoder',
            'prednet': {
                'pred_hidden': cfg.model.decoder.pred_hidden,
                'pred_rnn_layers': cfg.model.decoder.pred_rnn_layers,
                'forget_gate_bias': 1.0,
                't_max': None,
                'dropout': 0.1,
            },
            'vocab_size': vocab_size,
            'blank_as_pad': True,  # Use blank token as padding
        },
        
        # RNN-T Joint Network
        'joint': {
            '_target_': 'nemo.collections.asr.modules.rnnt.RNNTJoint',
            'fuse_loss_wer': False,
            'jointnet': {
                'joint_hidden': cfg.model.joint.joint_hidden,
                'activation': cfg.model.joint.activation,
                'dropout': cfg.model.joint.dropout,
            },
            'num_classes': vocab_size,
            'vocabulary': list(vocab.keys()),
            'log_softmax': True,  # Apply log softmax to joint output
            'preserve_memory': False,  # Trade memory for speed
            # 'fused_batch_size': -1,  # Enable fused batch computation for speed
        },
        
        # Decoding strategy for inference
        'decoding': {
            'strategy': 'greedy_batch',  # Much faster batched greedy decoding
            'greedy': {
                'max_symbols': 15,  # Reduced for faster validation
            },
            'greedy_batch': {
                'max_symbols': 13,
                'enable_cuda_graphs': False,  # Disable due to bf16 dtype conflict
                # 'precision': 'bf16-mixed',
            },
            # Disable CUDA graphs globally for bf16 compatibility
            'use_cuda_graphs': False,
            'preserve_frame_confidence': False,
            # 'precision': 'bf16-mixed',
            'preserve_alignments': False,
            'compute_timestamps': False,
            'preserve_word_confidence': False,
            'confidence_method_cfg': None,
        },
        
        # Optimizer configuration
        'optim': {
            'name': 'adamw',
            'lr': cfg.training.learning_rate,
            'betas': [0.9, 0.98],
            'weight_decay': 1e-3,
            # 'fused': True,
            'sched': {
                'name': 'CosineAnnealing',
                'warmup_steps': 2000,  # Longer warmup for stability with higher LR
                'warmup_ratio': None,
                'min_lr': 1e-6,
            }
        },
        
        # RNN-T Loss configuration
        'loss': {
            '_target_': 'nemo.collections.asr.losses.rnnt.RNNTLoss',
            'loss_name': 'warprnnt_numba',  # Try warprnnt (GPU) first for faster computation
            'blank_idx': 0,
            'fastemit_lambda': 0.001,  # Small FastEmit regularization for faster convergence
            'clamp': -1.0,  # Clamp for logits (-1 = disabled)
        },
        
        # Training configuration - set to None to skip NeMo's default data loading
        'train_ds': None,
        'validation_ds': None,
        
        # Spec augmentation (optional, for robustness)
        'spec_augment': None,  # Disable spec augmentation for gesture data
    })
    
    # Create a custom subclass to handle our feature format
    class GestureRNNTModel(nemo_asr.models.EncDecRNNTModel):
        """Custom RNN-T model that bypasses audio preprocessing."""
        def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Temporarily disable autocast so logits/ops are fp32 inside NeMo's decode
            if torch.cuda.is_available():
                # Works regardless of global Trainer precision
                with torch.autocast(device_type="cuda", enabled=False):
                    return super().validation_step(batch, batch_idx, dataloader_idx)
            else:
                return super().validation_step(batch, batch_idx, dataloader_idx)
        def __init__(self, cfg):
            # Force disable CUDA graphs in config before initialization
            if 'decoding' in cfg:
                cfg.decoding.use_cuda_graph_decoder = False
                if 'greedy_batch' in cfg.decoding:
                    cfg.decoding.greedy_batch.enable_cuda_graphs = False
                    cfg.decoding.greedy_batch.use_cuda_graph_decoder = False

            super().__init__(cfg=cfg)

            # Apply torch.compile if available but disable CUDA graphs to avoid conflicts
            if hasattr(torch, 'compile'):
                try:
                    logger.info("Compiling model components with torch.compile...")
                    # Increase recompile limit to handle checkpoint loading
                    import torch._dynamo as dynamo
                    dynamo.config.cache_size_limit = 256  # Further increased to prevent recompilation
                    dynamo.config.suppress_errors = True  # Continue on compilation errors

                    compile_kwargs = {
                        'mode': 'max-autotune',  # Better kernel optimization than reduce-overhead
                        'options': {
                            'triton.cudagraphs': False,  # Disable CUDA graphs
                            'shape_padding': True,  # Enable shape padding
                        }
                    }
                    self.encoder = torch.compile(self.encoder, **compile_kwargs)
                    self.decoder = torch.compile(self.decoder, **compile_kwargs)
                    self.joint = torch.compile(self.joint, **compile_kwargs)
                    logger.info("Model compilation complete (cache limit increased to 256)")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without compilation: {e}")


            # Make WER decoding fp32-safe and faster during training

            self._wrap_wer_update_fp32(throttle_every_n=64, disable_in_train=False)  # Throttle WER to every 64 steps for monitoring

        def _wrap_wer_update_fp32(self, throttle_every_n: int = 0, disable_in_train: bool = True):
            """
            Ensures NeMo's WER.decode runs with autocast disabled (fp32), avoiding
            dtype mismatch inside label-looping RNNT greedy decode. Optionally
            throttles updates during training to cut walltime.
            """
            if not hasattr(self, "wer") or not hasattr(self.wer, "update"):
                return  # nothing to wrap yet

            original_update = self.wer.update

            def wrapped_update(*args, **kwargs):
                # Skip entirely during training to eliminate decode overhead
                if self.training and disable_in_train:
                    return
                # Throttle during training to reduce decode cost
                if self.training and throttle_every_n and getattr(self, "trainer", None):
                    step = getattr(self.trainer, "global_step", 0) or 0
                    if step % throttle_every_n != 0:
                        # Skip most batches; metrics still get periodic updates
                        return

                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", enabled=False):
                        return original_update(*args, **kwargs)
                else:
                    return original_update(*args, **kwargs)

            self.wer.update = wrapped_update

        def _force_disable_decode_graphs(self):
            """Force disable CUDA graphs in all decoding components to prevent re-enabling."""
            for obj in (getattr(self, "decoding", None), getattr(self, "wer", None)):
                dec = getattr(obj, "decoding", None)
                if dec is not None:
                    for attr in ("enable_cuda_graphs", "use_cuda_graph_decoder"):
                        if hasattr(dec, attr):
                            setattr(dec, attr, False)
                    if hasattr(dec, "decoding_computer"):
                        if hasattr(dec.decoding_computer, "use_cuda_graph_decoder"):
                            dec.decoding_computer.use_cuda_graph_decoder = False

        def on_train_epoch_start(self):
            """Ensure CUDA graphs are disabled at start of each training epoch."""
            super().on_train_epoch_start()
            self._force_disable_decode_graphs()

    # If NeMo re-creates metrics later, re-wrap them:
        # def setup_optimization(self, optim_config=None):
        #     result = super().setup_optimization(optim_config)
        #     self._wrap_wer_update_fp32(throttle_every_n=8)
        #     return result

        def _setup_decoding(self, cfg):
            """Override to force disable CUDA graphs in decoding."""
            # Ensure CUDA graphs are disabled
            if 'decoding' in cfg:
                cfg.decoding.use_cuda_graphs = False
                cfg.decoding.use_cuda_graph_decoder = False
                if 'greedy_batch' in cfg.decoding:
                    cfg.decoding.greedy_batch.enable_cuda_graphs = False
                    cfg.decoding.greedy_batch.use_cuda_graph_decoder = False
            # Call parent implementation
            result = super()._setup_decoding(cfg)

            # Double-check after setup
            if hasattr(self, 'decoding') and hasattr(self.decoding, 'decoding'):
                self.decoding.decoding.use_cuda_graph_decoder = False
                logger.info("✓ Forced CUDA graphs disabled in model decoding")

            if hasattr(self, 'wer') and hasattr(self.wer, 'decoding'):
                if hasattr(self.wer.decoding, 'decoding'):
                    self.wer.decoding.decoding.use_cuda_graph_decoder = False
                    logger.info("✓ Forced CUDA graphs disabled in WER decoding")

            return result

        def setup_optimization(self, optim_config=None):
            """Override to disable CUDA graphs after optimizer setup."""
            result = super().setup_optimization(optim_config)

            # Disable CUDA graphs for WER metric to avoid bf16 conflicts
            # This must be done after setup_optimization where the decoder is created
            if hasattr(self, 'wer') and hasattr(self.wer, 'decoding'):
                if hasattr(self.wer.decoding, 'decoding'):
                    # Force disable CUDA graphs in the decoding computer
                    # self.wer.decoding.decoding.enable_cuda_graphs = False
                    
                    self.wer.decoding.decoding.use_cuda_graph_decoder = False
                    if hasattr(self.wer.decoding.decoding, 'decoding_computer'):
                        # self.wer.decoding.decoding.decoding_computer.enable_cuda_graphs = False
                        self.wer.decoding.decoding.decoding_computer.use_cuda_graph_decoder = False
                    logger.info("✓ Disabled CUDA graphs for WER metric (bf16 compatibility)")

            return result

        def on_validation_epoch_start(self):
            """Ensure CUDA graphs are disabled before validation."""
            super().on_validation_epoch_start()
            self._force_disable_decode_graphs()

        def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
            """
            Override forward to bypass preprocessor since we already have features.
            Our input_signal is already features with shape (batch, time, features).
            """
            # NeMo expects (batch, features, time) for the encoder
            # Our data is (batch, time, features) so transpose
            if input_signal is not None:
                processed_signal = input_signal.transpose(1, 2)  # (batch, features, time)
                processed_signal_length = input_signal_length

            # Pass to encoder
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
            return encoded, encoded_len

    # Instantiate our custom NeMo model
    logger.info("Creating GestureRNNTModel...")
    model = GestureRNNTModel(cfg=model_config)

    # Assign our custom dataloaders directly to the NeMo model
    model._train_dl = train_loader
    model._validation_dl = val_loader
    
    # Log model architecture
    logger.info(f"Model architecture:")
    logger.info(f"  Encoder: Conformer with {cfg.model.encoder.num_layers} layers, {cfg.model.encoder.subsampling_factor}x subsampling")
    logger.info(f"  Decoder: RNN-T Prediction Network with {cfg.model.decoder.pred_rnn_layers} LSTM layer{'s' if cfg.model.decoder.pred_rnn_layers > 1 else ''}")
    logger.info(f"  Joint: RNN-T Joint Network with {cfg.model.joint.joint_hidden} hidden units")
    # logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    logger.info(f"Optimizations:")
    logger.info(f"  • Precision: bf16-mixed (faster training)")
    logger.info(f"  • {cfg.model.encoder.subsampling_factor}x subsampling: 200→{200//cfg.model.encoder.subsampling_factor} timesteps")
    logger.info(f"  • TF32 and cuDNN autotuning: Enabled")
    if hasattr(torch, 'compile'):
        logger.info(f"  • torch.compile: Enabled (default mode, CUDA graphs disabled)")
    else:
        logger.info(f"  • torch.compile: Not available (upgrade to PyTorch 2.0+)")
    
    # Setup callbacks
    prediction_logger = PredictionLogger(val_loader, vocab)

    # Checkpoint callback to track WER (accuracy = 100% - WER)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_wer',  # Monitor Word Error Rate
        mode='min',  # Minimize WER (lower is better)
        save_top_k=3,
        filename='epoch={epoch:02d}-wer={val_wer:.2f}',
        save_last=True,
        verbose=True
    )
    
    # Configure PyTorch Lightning trainer with RTX 4090M optimizations
    trainer = pl.Trainer(
        devices=cfg.training.devices,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.gradient_accumulation,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        check_val_every_n_epoch=5,
        limit_val_batches=64,
        # val_check_interval=1.0,  # Validate every 5 epochs only
        callbacks=[prediction_logger, checkpoint_callback],
        enable_checkpointing=True,
        default_root_dir="./rnnt_checkpoints_" + runtime_id, # use the date and time to make it unique
        enable_model_summary=False,  # Disable to reduce overhead
        enable_progress_bar=True,
        # Minimal sanity check
        num_sanity_val_steps=0,  # Just 1 batch for sanity check
        # Performance optimizations
        benchmark=True,  # Enable cuDNN benchmark
        sync_batchnorm=False,  # Not needed for single GPU
        deterministic=False,  # Allow non-deterministic ops for speed
        
        # Logging
        logger=pl.loggers.TensorBoardLogger(
            save_dir="./rnnt_logs_" + runtime_id,
            name="conformer_rnnt"
        ),
    )

    # Attach trainer to model (required by NeMo)
    model.set_trainer(trainer)

    # Start training
    logger.info("\n" + "="*60)
    logger.info("Starting RNN-Transducer training...")
    logger.info("Using warprnnt_numba for 5-10x faster loss computation")
    logger.info("Tracking WER (Word Error Rate) - Accuracy = 100% - WER%")
    logger.info("This models output dependencies unlike CTC:")
    logger.info("  • P(y_i | y_1...y_{i-1}, x) via Prediction Network")
    logger.info("  • Joint network combines encoder + prediction")
    logger.info("  • Superior disambiguation for similar swipe patterns")
    logger.info("="*60 + "\n")
    
    # Resume from checkpoint if available
    if resume_checkpoint:
        logger.info(f"\n📥 Resuming training from: {os.path.basename(resume_checkpoint)}")
        trainer.fit(model, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(model)
    
    # Save the final model in NeMo format
    save_path = "conformer_rnnt_gesture_" + runtime_id + ".nemo"
    model.save_to(save_path)
    logger.info(f"\n✓ Model saved to {save_path}")
    
    logger.info("\n" + "="*60)
    logger.info("RNN-Transducer training complete!")
    logger.info("The model now captures character dependencies,")
    logger.info("providing superior accuracy for gesture typing.")
    logger.info("Expected improvements over CTC:")
    logger.info("  • 40-50% WER reduction")
    logger.info("  • Better disambiguation of similar swipes")
    logger.info("  • Learned linguistic patterns (q→u, th→e, etc.)")
    logger.info("="*60)

if __name__ == '__main__':
    main()
