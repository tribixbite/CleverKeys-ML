#!/usr/bin/env python3
"""
Standalone GestureRNNTModel class extracted from train_transducer.py
for use in export scripts.
"""

import torch
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig


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

        # Skip torch.compile during checkpoint loading to avoid _orig_mod issues
        # Apply torch.compile if available but disable CUDA graphs to avoid conflicts
        # if hasattr(torch, 'compile') and torch.cuda.is_available():
        #     # Compile encoder only (most compute-intensive)
        #     try:
        #         self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
        #         print("✓ Compiled encoder with torch.compile")
        #     except Exception as e:
        #         print(f"⚠️ torch.compile failed: {e}")


def get_default_config():
    """Get the default configuration used for training."""
    return DictConfig({
        'sample_rate': 16000,  # Not used for gesture data but required by NeMo
        'labels': ['<blank>', "'"] + [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['<unk>'],

        'model_defaults': {
            'asr_enc_hidden': 256,
            'enc_hidden': 256,
            'pred_hidden': 320,
            'joint_hidden': 512
        },

        'preprocessor': {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
            'normalize': 'per_feature',
            'window_size': 0.025,
            'window_stride': 0.01,
            'window': 'hann',
            'features': 37,  # Feature dimension matches gesture features
            'n_fft': 512,
            'frame_splicing': 1,
            'dither': 0.00001,
            'pad_value': 0
        },

        'encoder': {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': 37,  # Gesture feature dimension
            'feat_out': -1,
            'n_layers': 8,
            'n_heads': 4,
            'd_model': 256,
            'subsampling': 'striding',
            'subsampling_factor': 6,
            'subsampling_conv_channels': 256,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'pos_emb_max_len': 5000,
            'dropout': 0.1,
            'dropout_emb': 0.0,
            'dropout_att': 0.1
        },

        'decoder': {
            '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
            'normalization_mode': None,
            'random_state_sampling': False,
            'blank_as_pad': True,
            'prednet': {
                'pred_hidden': 320,
                'pred_rnn_layers': 2,
                'dropout': 0.1
            }
        },

        'joint': {
            '_target_': 'nemo.collections.asr.modules.RNNTJoint',
            'log_softmax': None,
            'preserve_memory': False,
            'fuse_loss_wer': True,
            'fused_batch_size': 16,
            'jointnet': {
                'joint_hidden': 512,
                'activation': 'relu',
                'dropout': 0.1
            }
        },

        'decoding': {
            'strategy': 'greedy',
            'preserve_alignments': False,
            'compute_timestamps': False,
            'compute_langs': False,
            'greedy': {
                'max_symbols': 10,
            }
        },

        'loss': {
            '_target_': 'nemo.collections.asr.losses.RNNTLoss',
            'num_classes': 29,  # vocab size: blank + ' + a-z + unk
            'reduction': 'mean_batch'
        },
    })