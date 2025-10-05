#!/usr/bin/env python3
"""
The Ultimate Robust Training Script for Gesture Typing Models.

This script trains a Squeezeformer-CTC model for gesture typing on a fixed-layout
keyboard. It is designed for maximum automation and robustness, incorporating an
advanced curriculum learning strategy and best practices for modern hardware.

-------------------------------
-- ARCHITECTURE: SQUEEZEFORMER-CTC --
-------------------------------
This model uses a Squeezeformer encoder with a CTC (Connectionist Temporal
Classification) decoder. This architecture was chosen for several key reasons:

1.  **Efficiency (Squeezeformer):** The Squeezeformer encoder is a highly
    optimized version of the Conformer, designed to provide state-of-the-art
    accuracy with lower computational cost, making it ideal for on-device deployment.
2.  **Simplicity (CTC):** Unlike RNN-T, CTC is non-autoregressive. This means
    inference is a single, stateless forward pass. This completely eliminates the
    need for managing decoder state (h, c) and feedback loops in production,
    dramatically simplifying your web and Android implementation.
3.  **Performance:** This combination is a state-of-the-art choice for tasks
    requiring efficient sequence-to-sequence modeling.

------------------------
-- CURRICULUM LEARNING --
------------------------
The script automates a multi-stage training curriculum to help the model
converge better and achieve higher accuracy:

1.  **Initial Frequency Balancing (Epochs 0-50):** The training starts by
    aggressively downsampling swipe samples for common words. This forces the
    model to pay attention to rarer words from the very beginning, preventing it
    from getting stuck in a local minimum where it only predicts common words.

2.  **Automated Profile Progression:** After the initial 50 epochs, a custom
    callback monitors the validation Word Error Rate (val_wer). When the model's
    improvement flattens for a set number of epochs, the callback automatically
    switches to the next, more challenging data sampling profile in the
    curriculum (e.g., from focusing on medium words to long words, then rare words).

------------------------
--       USAGE        --
------------------------
# Start a new training run (will store everything in 12-02-squeeze/)
python train_squeezeformer_ctc.py

# The script automatically finds and resumes from the latest checkpoint
# within the run directory if it's interrupted.
python train_squeezeformer_ctc.py

# Force a fresh run, ignoring any previous checkpoints
python train_squeezeformer_ctc.py --ignore-checkpoint

# Resume from a specific checkpoint file
python train_squeezeformer_ctc.py --resume-from-checkpoint path/to/your.ckpt

# Override the batch size for a run
python train_squeezeformer_ctc.py --batch-size 128

-------------------------------------
-- DEPLOYMENT: EXPORT AND PRODUCTION --
-------------------------------------
After training, follow these steps to deploy your model to web and Android.

**Step 1: Export to ONNX**
Run the model's built-in `export()` method. Because this is a stateless CTC model,
it will produce a single, simple `model.onnx` file.

```python
# Create an export.py script
import torch
from train_squeezeformer_ctc import GestureCTCModel

# Load your best checkpoint
model = GestureCTCModel.load_from_checkpoint("12-02-squeeze/.../best.ckpt")
model.eval()

# Export to ONNX
model.export("gesture_model.onnx")
```

**Step 2: Quantize with Hugging Face Optimum**
Install `optimum` and `onnxruntime`. Use the Optimum CLI to apply robust 8-bit
quantization. This solves the quantization issues you faced before.

```bash
pip install optimum[onnxruntime]

# Apply dynamic INT8 quantization
optimum-cli onnxruntime quantize --onnx_path gesture_model.onnx \
  --output gesture_model_quant.onnx --quantization_approach dynamic
```

**Step 3: Web Implementation (Transformers.js)**
Use the quantized `gesture_model_quant.onnx` file with Transformers.js. The
inference logic is a simple, single forward pass.

```javascript
import { AutoProcessor, AutoModelForCTC } from '@xenova/transformers';

// Load model and processor (you'll need to create a simple processor.json)
const processor = await AutoProcessor.from_pretrained('./model_path/');
const model = await AutoModelForCTC.from_pretrained('./model_path/'); // Loads gesture_model_quant.onnx

// Your 37D feature tensor (shape: [1, sequence_length, 37])
const features = ...;

// Single forward pass to get logits
const { logits } = await model({ inputs: features });
// logits shape: [1, sequence_length, vocab_size]

// Decode the logits. `processor.decode` handles the CTC decoding logic,
// including beam search if a decoder is configured.
const transcription = processor.decode(logits[0]);
console.log(transcription);
```

**Step 4: Android Implementation (ONNX Runtime Mobile)**
Use the same `gesture_model_quant.onnx` file in your Android app. The logic is
again a simple, stateless forward pass with easy hardware acceleration.

```kotlin
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.util.Collections

// 1. Initialize environment and session options
val ortEnvironment = OrtEnvironment.getEnvironment()
val sessionOptions = OrtSession.SessionOptions()

// 2. Enable hardware acceleration (NOT DEPRECATED)
// This lets Android choose the best available hardware (GPU, NPU, etc.)
sessionOptions.addNnapi()

// 3. Load the .onnx model from your app's assets
val modelBytes = resources.openRawResource(R.raw.gesture_model_quant).readBytes()
val session = ortEnvironment.createSession(modelBytes, sessionOptions)

// 4. Prepare your feature tensor
// Shape: [1, sequence_length, 37] -> flattened FloatBuffer
val inputTensor = OnnxTensor.createTensor(ortEnvironment, yourFeatureBuffer, longArrayOf(1, 96, 37))

// 5. Run inference in a single, stateless call
val results = session.run(Collections.singletonMap("features", inputTensor))
val logits = results[0].value as Array<Array<FloatArray>>

// 6. Apply a simple greedy or beam search CTC decoder to the logits
val decodedText = yourLocalCtcDecoder(logits)
```

"""

import argparse
import json
import logging
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    raise ImportError(
        "NeMo ASR toolkit is required. Please install with: pip install nemo_toolkit[asr]"
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SAMPLING_PROFILES = {
    "initial_downsample": {
        "strategy": "frequency_cap",
        "target_freq": 0.0002,
        "description": "Aggressively downsample common words to force rare word learning",
    },
    "medium_words": {
        "min_word_length": 5,
        "max_word_length": 7,
        "freq_power": 0.5,
        "rare_word_boost": 2.5,
        "description": "Focus on medium-length words with balanced frequency",
    },
    "long_words": {
        "min_word_length": 8,
        "freq_power": 0.6,
        "rare_word_boost": 3.0,
        "length_power": 0.8,
        "description": "Emphasize longer words which are harder to predict",
    },
    "rare_focused": {
        "max_frequency": 1000,
        "freq_power": 0.7,
        "rare_frequency_threshold": 100,
        "rare_word_boost": 4.0,
        "description": "Heavy focus on rare words for comprehensive coverage",
    },
    "production_balanced": {
        "min_word_length": 2,
        "freq_power": 0.55,
        "rare_frequency_threshold": 100,
        "rare_word_boost": 2.5,
        "description": "Final balanced profile for production deployment",
    },
}

CURRICULUM_STAGES = [
    "initial_downsample",
    "medium_words",
    "long_words",
    "rare_focused",
    "production_balanced",
]

CONFIG = {
    "run_name": f"12-02-squeeze",
    "data": {
        "train_manifest": "data/train_final_train.jsonl",
        "val_manifest": "data/train_final_val.jsonl",
        "vocab": list("abcdefghijklmnopqrstuvwxyz'"),
    },
    "training": {
        "batch_size": 1024,
        "num_workers": 12,  # Set to 0 to avoid multiprocessing issues
        "learning_rate": 5e-4,
        "max_epochs": 500,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "precision": "bf16-mixed" if torch.cuda.is_available() else "32",
        "warmup_steps": 8000,
        "accumulate_grad_batches": 1,
    },
    "model": {
        "encoder": {
            "feat_in": 37,  # From our custom featurizer
            "n_layers": 12,
            "d_model": 256,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "conv_kernel_size": 31,
            "dropout": 0.1,
            "dropout_att": 0.1,
        },
    },
    "preprocess": {
        "resample_short_target": 56,
        "resample_long_target": 96,
        "resample_short_threshold": 48,
        "resample_long_threshold": 112,
    },
    "curriculum": {
        "profiles": CURRICULUM_STAGES,
        "initial_stage_epochs": 50,
        "switch_patience": 5,  # Switch profile if val_wer doesn't improve for 5 epochs
        "switch_min_delta": 0.005,  # Minimum improvement to reset patience
    },
    "early_stopping": {
        "target_wer": 0.005,  # More realistic target: 5% WER
        "patience": 500,  # Much more patience since we're far from target
    },
    "validation": {
        "log_predictions": 5,  # Log 5 random correct/incorrect predictions per epoch
    },
}


# --- Feature Extraction ---
class PersonalizedSwipeFeaturizer:
    """
    Robust 37-dimensional feature extractor for swipe gestures.
    Features include position, velocity, acceleration, and keyboard proximity.
    """

    def __init__(self):
        self.feature_dim = 37
        # QWERTY keyboard layout positions (normalized)
        self.key_positions = self._get_keyboard_layout()

    def _get_keyboard_layout(self) -> Dict[str, Tuple[float, float]]:
        """Returns normalized positions for QWERTY keyboard layout."""
        layout = {}
        rows = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm",
        ]
        for row_idx, row in enumerate(rows):
            y = row_idx * 0.33  # Normalized y position
            for col_idx, char in enumerate(row):
                # Add slight offset for middle and bottom rows
                x_offset = 0.05 * row_idx if row_idx > 0 else 0
                x = (col_idx / 10.0) + x_offset
                layout[char] = (x, y)
        layout["'"] = (0.95, 0.33)  # Apostrophe position
        return layout

    def __call__(self, points: List[Dict]) -> np.ndarray:
        """
        Extract features from a sequence of swipe points.

        Args:
            points: List of dicts with 'x', 'y', 't' keys

        Returns:
            numpy array of shape (len(points), 37)
        """
        if not points:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        features = []
        for i in range(len(points)):
            features.append(self._compute_feature_vector(points, i))

        return np.stack(features).astype(np.float32)

    def _compute_feature_vector(self, points: List[Dict], idx: int) -> np.ndarray:
        """Compute 37D feature vector for a single point."""
        vec = np.zeros(self.feature_dim, dtype=np.float32)

        # Current point
        p_curr = points[idx]
        x, y, t = p_curr["x"], p_curr["y"], p_curr["t"]

        # Basic position (2D)
        vec[0] = x
        vec[1] = y

        # Velocity features (2D)
        if idx > 0:
            p_prev = points[idx - 1]
            dt = max((t - p_prev["t"]) / 1000.0, 1e-6)  # Convert ms to seconds
            vec[2] = (x - p_prev["x"]) / dt
            vec[3] = (y - p_prev["y"]) / dt

        # Acceleration features (2D)
        if idx > 0 and idx < len(points) - 1:
            p_next = points[idx + 1]
            p_prev = points[idx - 1]

            # Time delta for the next point
            dt_next = max((p_next["t"] - t) / 1000.0, 1e-6)

            # Velocity at the next point
            vx_next = (p_next["x"] - x) / dt_next
            vy_next = (p_next["y"] - y) / dt_next

            # Total time delta for central difference (t_next - t_prev)
            dt_total = max((p_next["t"] - p_prev["t"]) / 1000.0, 1e-6)

            # Correct acceleration using central difference
            vec[4] = (vx_next - vec[2]) / dt_total
            vec[5] = (vy_next - vec[3]) / dt_total

        # Speed and direction (2D)
        speed = np.hypot(vec[2], vec[3])
        vec[6] = speed
        vec[7] = np.arctan2(vec[3], vec[2]) if speed > 1e-6 else 0

        # Distance to each key (28D)
        for i, (char, (kx, ky)) in enumerate(self.key_positions.items()):
            dist = np.hypot(x - kx, y - ky)
            vec[8 + i] = np.exp(-dist * 5)  # Gaussian proximity

        # Trajectory curvature (1D)
        if idx > 1 and idx < len(points) - 1:
            p_prev = points[idx - 1]
            p_next = points[idx + 1]
            # Compute angle change
            v1 = np.array([p_curr["x"] - p_prev["x"], p_curr["y"] - p_prev["y"]])
            v2 = np.array([p_next["x"] - p_curr["x"], p_next["y"] - p_curr["y"]])
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                vec[36] = np.arccos(np.clip(cos_angle, -1, 1))

        return vec


# --- Dataset ---
class GestureDataset(Dataset):
    """Dataset for gesture typing with curriculum learning support."""

    def __init__(
        self,
        manifest_path: str,
        vocab: Dict[str, int],
        preprocess_cfg: DictConfig,
        sampling_cfg: Optional[DictConfig] = None,
        is_training: bool = True,
    ):
        super().__init__()
        self.vocab = vocab
        self.preprocess_cfg = preprocess_cfg
        self.featurizer = PersonalizedSwipeFeaturizer()
        self.is_training = is_training

        # Load data
        with open(manifest_path, "r", encoding="utf-8") as fh:
            self.all_samples = [json.loads(line) for line in fh if line.strip()]

        # Apply sampling strategy
        self.samples = self._apply_sampling(self.all_samples, sampling_cfg)

        logger.info(
            f"Loaded {len(self.samples)} samples from {manifest_path} "
            f"using sampling: {sampling_cfg.get('description', 'none') if sampling_cfg else 'none'}"
        )

    def _apply_sampling(
        self, samples: List[Dict], cfg: Optional[DictConfig]
    ) -> List[Dict]:
        """Apply curriculum learning sampling strategy."""
        if not cfg or cfg.get("strategy") == "none":
            return samples

        if cfg.get("strategy") == "frequency_cap":
            # Count word frequencies
            word_counts = Counter(s["word"] for s in samples)
            total_words = sum(word_counts.values())

            # Calculate target count for each word
            target_freq = cfg.get("target_freq", 0.0002)
            target_counts = {
                word: max(1, int(total_words * target_freq)) for word in word_counts
            }

            # Downsample
            downsampled = []
            current_counts = Counter()
            for s in samples:
                word = s["word"]
                if current_counts[word] < target_counts[word]:
                    downsampled.append(s)
                    current_counts[word] += 1

            return downsampled

        # Filter by word properties
        filtered = samples

        if "min_word_length" in cfg:
            filtered = [s for s in filtered if len(s["word"]) >= cfg.min_word_length]

        if "max_word_length" in cfg:
            filtered = [s for s in filtered if len(s["word"]) <= cfg.max_word_length]

        if "max_frequency" in cfg:
            word_counts = Counter(s["word"] for s in samples)
            filtered = [
                s for s in filtered if word_counts[s["word"]] <= cfg.max_frequency
            ]

        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Optional[Dict]:
        item = self.samples[index]
        raw_points = item.get("points", [])

        if len(raw_points) < 2:
            return None

        # Preprocessing pipeline
        norm_points = self._normalize_points(raw_points)
        target_len = self._determine_resample_target(len(norm_points))
        resampled_points = self._resample_points(norm_points, target_len)
        features = self.featurizer(resampled_points)

        # Tokenize word
        tokens = [self.vocab.get(c, -1) for c in item["word"].lower()]
        tokens = [t for t in tokens if t != -1]  # Filter unknown chars

        if len(tokens) == 0:
            return None

        return {
            "features": torch.from_numpy(features).float(),
            "feature_length": len(features),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "token_length": len(tokens),
            "word": item["word"],
        }

    def _normalize_points(self, points: List[Dict]) -> List[Dict]:
        """Convert from [0,1] to [-1,1] coordinate system."""
        if not points:
            return []

        # Normalize time to start at 0
        start_t = float(points[0].get("t", 0.0))

        return [
            {
                "x": p.get("x", 0.5) * 2.0 - 1.0,  # Convert [0,1] -> [-1,1]
                "y": p.get("y", 0.5) * 2.0 - 1.0,
                "t": max(0.0, float(p.get("t", 0.0)) - start_t),
            }
            for p in points
        ]

    def _determine_resample_target(self, length: int) -> int:
        """Determine adaptive resampling target based on trace length."""
        cfg = self.preprocess_cfg

        if length <= cfg.resample_short_threshold:
            return cfg.resample_short_target
        elif length >= cfg.resample_long_threshold:
            return cfg.resample_long_target
        else:
            # Linear interpolation
            progress = (length - cfg.resample_short_threshold) / (
                cfg.resample_long_threshold - cfg.resample_short_threshold
            )
            return int(
                cfg.resample_short_target
                + progress * (cfg.resample_long_target - cfg.resample_short_target)
            )

    def _resample_points(self, points: List[Dict], target_count: int) -> List[Dict]:
        """Resample points to target count using linear interpolation."""
        if not points or target_count <= 0:
            return []

        if len(points) == target_count:
            return points

        if len(points) == 1:
            return points * target_count

        # Calculate time step
        duration = max(points[-1]["t"] - points[0]["t"], 1.0)
        step = duration / max(target_count - 1, 1)

        resampled = []
        src_idx = 0

        for i in range(target_count):
            target_time = points[0]["t"] + step * i

            # Find surrounding points
            while src_idx < len(points) - 2 and points[src_idx + 1]["t"] < target_time:
                src_idx += 1

            p1 = points[src_idx]
            p2 = points[min(src_idx + 1, len(points) - 1)]

            # Linear interpolation
            if p2["t"] > p1["t"]:
                alpha = (target_time - p1["t"]) / (p2["t"] - p1["t"])
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                alpha = 0.0

            resampled.append(
                {
                    "x": p1["x"] + (p2["x"] - p1["x"]) * alpha,
                    "y": p1["y"] + (p2["y"] - p1["y"]) * alpha,
                    "t": target_time,
                }
            )

        return resampled


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """Custom collate function for variable-length sequences."""
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if not batch:
        return None

    # Stack features and tokens
    features = torch.nn.utils.rnn.pad_sequence(
        [item["features"] for item in batch], batch_first=True, padding_value=0.0
    )

    tokens = torch.nn.utils.rnn.pad_sequence(
        [item["tokens"] for item in batch],
        batch_first=True,
        padding_value=0,  # <-- FIX: Use a valid index like 0 for padding.
    )

    feature_lengths = torch.tensor(
        [item["feature_length"] for item in batch], dtype=torch.long
    )
    token_lengths = torch.tensor(
        [item["token_length"] for item in batch], dtype=torch.long
    )

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "tokens": tokens,
        "token_lengths": token_lengths,
        "words": [item["word"] for item in batch],
    }


# --- PyTorch Lightning Model ---
class GestureCTCModel(pl.LightningModule):
    """
    Lightning module for Squeezeformer-CTC gesture typing model.
    This is a self-contained pure PyTorch Lightning implementation.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.validation_outputs = []

        # Build vocab and determine blank ID
        self.vocab = {c: i for i, c in enumerate(cfg.data.vocab)}
        self.char_map = {i: c for c, i in self.vocab.items()}
        vocab_size = len(self.vocab)
        self.blank_id = vocab_size  # CRITICAL FIX: Blank is at index 27, AFTER the 27 characters (0-26).

        # Build Squeezeformer encoder from NeMo
        encoder_cfg = OmegaConf.to_container(cfg.model.encoder)
        self.encoder = nemo_asr.modules.SqueezeformerEncoder(**encoder_cfg)

        # Build the CTC decoder head
        self.decoder = torch.nn.Linear(
            encoder_cfg["d_model"],
            vocab_size + 1,  # Output size is vocab + 1 for the blank token
        )

        # Define the CTC loss function
        self.ctc_loss = torch.nn.CTCLoss(
            blank=self.blank_id, reduction="mean", zero_infinity=True
        )

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        # Transpose for encoder: (B, T, F) -> (B, F, T)
        features = features.transpose(1, 2)
        encoded, encoded_lengths = self.encoder(
            audio_signal=features, length=feature_lengths
        )

        # Squeezeformer outputs (B, D, T), transpose for decoder: (B, T, D)
        encoded = encoded.transpose(1, 2)

        logits = self.decoder(encoded)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs, encoded_lengths

    def training_step(self, batch: Optional[Dict], batch_idx: int):
        if batch is None:
            return None
        log_probs, encoded_lengths = self(batch["features"], batch["feature_lengths"])
        log_probs_t = log_probs.transpose(0, 1)  # (T, B, V) for CTC loss
        loss = self.ctc_loss(
            log_probs_t, batch["tokens"], encoded_lengths, batch["token_lengths"]
        )
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: Optional[Dict], batch_idx: int):
        if batch is None:
            return
        log_probs, encoded_lengths = self(batch["features"], batch["feature_lengths"])
        log_probs_t = log_probs.transpose(0, 1)
        loss = self.ctc_loss(
            log_probs_t, batch["tokens"], encoded_lengths, batch["token_lengths"]
        ).detach()  # CRITICAL FIX: Detach the loss to avoid NaNs/Infs

        # Greedy decoding for WER
        predictions = log_probs.argmax(dim=-1)
        decoded_preds, words = [], batch["words"]
        for i in range(predictions.shape[0]):
            pred_seq = predictions[i, : encoded_lengths[i]].cpu().numpy()
            # CRITICAL FIX: Check for the correct blank_id
            decoded = [
                token
                for idx, token in enumerate(pred_seq)
                if token != self.blank_id and (idx == 0 or token != pred_seq[idx - 1])
            ]
            pred_word = "".join([self.char_map.get(t, "") for t in decoded])
            decoded_preds.append(pred_word)

        if batch_idx < self.cfg.validation.log_predictions:
            for i in range(min(2, len(decoded_preds))):
                logger.info(f"[Val] REF: '{words[i]:<20}' | PRED: '{decoded_preds[i]}'")

        self.validation_outputs.append(
            {"predictions": decoded_preds, "references": words, "loss": loss}
        )
        return {"predictions": decoded_preds, "references": words, "loss": loss}

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        # Aggregate predictions and references from all validation batches
        all_predictions = []
        all_references = []
        for output in self.validation_outputs:
            all_predictions.extend(output["predictions"])
            all_references.extend(output["references"])

        # Calculate metrics on the aggregated lists
        total_words = len(all_references)
        total_correct = sum(
            1 for pred, ref in zip(all_predictions, all_references) if pred == ref
        )

        avg_loss = torch.stack([o["loss"] for o in self.validation_outputs]).mean()

        # This is actually Character Error Rate (CER), but we'll call it WER for consistency with the prompt
        wer = 1.0 - (total_correct / max(total_words, 1))

        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_wer", wer, on_epoch=True, prog_bar=True)
        logger.info(f"Validation WER: {wer:.4f}")
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.training.warmup_steps
            / self.trainer.estimated_stepping_batches
            if self.trainer.estimated_stepping_batches > 0
            else 0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def export(self, output_path: str):
        """Export model to ONNX format."""
        self.eval()

        # Create dummy input
        dummy_features = torch.randn(1, 96, 37)
        dummy_lengths = torch.tensor([96])

        # Export
        torch.onnx.export(
            self,
            (dummy_features, dummy_lengths),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["features", "feature_lengths"],
            output_names=["log_probs", "encoded_lengths"],
            dynamic_axes={
                "features": {0: "batch", 1: "time"},
                "feature_lengths": {0: "batch"},
                "log_probs": {0: "batch", 1: "time"},
                "encoded_lengths": {0: "batch"},
            },
        )

        logger.info(f"Model exported to {output_path}")


# --- Callbacks ---
class CurriculumCallback(Callback):
    """Callback for automated curriculum learning progression."""

    def __init__(self, cfg: DictConfig):
        self.curriculum_profiles = cfg.profiles
        self.initial_stage_epochs = cfg.initial_stage_epochs
        self.patience = cfg.switch_patience
        self.min_delta = cfg.switch_min_delta

        self.current_stage = 0
        self.val_wer_history = []
        self.epochs_since_improvement = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # After initial stage, check for curriculum switch
        if trainer.current_epoch > self.initial_stage_epochs:
            if self.epochs_since_improvement >= self.patience:
                self._advance_curriculum(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Monitor val_wer to decide when to switch
        metrics = trainer.callback_metrics

        if "val_wer" not in metrics:
            return

        current_wer = metrics["val_wer"].item()

        if len(self.val_wer_history) > 0:
            improvement = self.val_wer_history[-1] - current_wer
            if improvement > self.min_delta:
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

        self.val_wer_history.append(current_wer)

    def _advance_curriculum(self, trainer):
        if self.current_stage < len(self.curriculum_profiles) - 1:
            self.current_stage += 1
            new_profile_name = self.curriculum_profiles[self.current_stage]

            logger.info("=" * 80)
            logger.info(
                f"Advancing curriculum to stage {self.current_stage + 1}: '{new_profile_name}'"
            )
            profile = SAMPLING_PROFILES[new_profile_name]
            logger.info(f"Profile: {profile['description']}")
            logger.info("=" * 80)

            # Update the datamodule's config and reload
            if hasattr(trainer, "datamodule"):
                trainer.datamodule.sampling_profile_name = new_profile_name
                # Force recreation of train dataloader on next epoch
                trainer.datamodule.setup("fit")

            # Reset tracking
            self.epochs_since_improvement = 0
            self.val_wer_history = []


class GestureDataModule(pl.LightningDataModule):
    """Lightning DataModule for gesture typing data."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.sampling_profile_name = cfg.curriculum.profiles[0]

    def setup(self, stage: str):
        # Build vocab mapping
        vocab = {c: i for i, c in enumerate(self.cfg.data.vocab)}
        vocab["<blank>"] = len(vocab)  # Add blank token

        # Get current sampling config
        sampling_cfg = SAMPLING_PROFILES.get(self.sampling_profile_name, {})

        if stage == "fit" or stage is None:
            self.train_ds = GestureDataset(
                manifest_path=self.cfg.data.train_manifest,
                vocab=vocab,
                preprocess_cfg=self.cfg.preprocess,
                sampling_cfg=sampling_cfg,
                is_training=True,
            )

        self.val_ds = GestureDataset(
            manifest_path=self.cfg.data.val_manifest,
            vocab=vocab,
            preprocess_cfg=self.cfg.preprocess,
            sampling_cfg=None,  # No sampling for validation
            is_training=False,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.cfg.training.batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.cfg.training.batch_size
            * 2,  # Can use larger batch for validation
            collate_fn=collate_fn,
            num_workers=self.cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


# --- Main Execution ---
def main():
    pl.seed_everything(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Parse arguments
    parser = argparse.ArgumentParser(description="Robust Squeezeformer-CTC Training")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a .ckpt file to resume from.",
    )
    parser.add_argument(
        "--ignore-checkpoint",
        action="store_true",
        help="Start a fresh run, ignoring latest checkpoint.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Override batch size from config."
    )
    args = parser.parse_args()

    cfg = OmegaConf.create(CONFIG)
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    root_dir = Path(cfg.run_name)
    root_dir.mkdir(exist_ok=True)
    resume_path = args.resume_from_checkpoint
    if not resume_path and not args.ignore_checkpoint:
        ckpts = sorted(root_dir.rglob("*.ckpt"), key=os.path.getmtime, reverse=True)
        if ckpts:
            resume_path = str(ckpts[0])
            print(f"Auto-resuming from: {resume_path}")

    # --- Build Model ---
    # Pass the full config to the model
    model = GestureCTCModel(cfg=cfg)
    data_module = GestureDataModule(cfg=cfg)

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=1,
        precision=cfg.training.precision,
        log_every_n_steps=20,
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / "checkpoints",
                filename="{epoch:03d}-{val_wer:.3f}",
                monitor="val_wer",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                min_delta=0.001,
                patience=cfg.early_stopping.patience,
                stopping_threshold=cfg.early_stopping.target_wer,
            ),
            CurriculumCallback(cfg.curriculum),
        ],
    )

    # if not resume_path and hasattr(torch, "compile"):
    #     print("Attempting to compile model with torch.compile...")
    #     try:
    #         model = torch.compile(model)
    #         print("Model compiled!")
    #     except Exception as e:
    #         logger.warning(
    #             f"torch.compile failed: {e}. Continuing without compilation."
    #         )

    # Train
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info(f"Run name: {cfg.run_name}")
    logger.info(f"Device: {cfg.training.accelerator}")
    logger.info(f"Precision: {cfg.training.precision}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Max epochs: {cfg.training.max_epochs}")
    logger.info("=" * 80)

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    # NOTE: The PersonalizedSwipeFeaturizer is a placeholder.
    # Paste your full 37D feature calculation logic into it.
    main()
