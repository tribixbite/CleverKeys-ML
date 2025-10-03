# Training Script Analysis & Recommendations

## ✅ Correctly Implemented

### 1. Coordinate Transformation
The `_prepare_points` method correctly transforms coordinates:
```python
# Correct: Dataset [0,1] → Training [-1,1]
centered_x = raw_x * 2.0 - 1.0
centered_y = raw_y * 2.0 - 1.0
centered_x = clamp(centered_x, -1.5, 1.5)  # Allow slight out-of-bounds
```

### 2. Feature Extraction
- PersonalizedSwipeFeaturizer properly extracts 37 features
- Includes kinematic features (position, velocity, acceleration)
- Includes spatial features (distances to nearest keys)
- Correctly handles edge cases (first/last points)

### 3. Adaptive Resampling
- Uses linear interpolation between 56-96 frames
- Smooth transition prevents jarring changes in sequence length

## 🔴 Critical Issues

### 1. Model Underfitting (Root Cause of Poor Performance)
**Problem**: Model only learns first character mapping, WER 0.457 is misleading
**Evidence**: Frame 0 predicts 'h' correctly, all other frames output blanks
**Solution**: Need more aggressive training strategy

### 2. Learning Rate Too Conservative
**Current**: 2e-4 with AdamW
**Issue**: Too small for RNN-T which needs aggressive early learning
**Recommendation**: Start with 5e-4 or 1e-3, use warmup + cosine annealing

### 3. Batch Size Too Large
**Current**: 1000 samples
**Issue**: With 16GB VRAM, this likely causes gradient accumulation issues
**Recommendation**: Reduce to 256-512 for more frequent updates

## 🟡 Important Improvements

### 1. Add Curriculum Learning
```python
def get_curriculum_stage(epoch, max_epochs):
    """Progressive difficulty increase"""
    if epoch < max_epochs * 0.2:
        return "short_words"  # 2-4 chars
    elif epoch < max_epochs * 0.4:
        return "common_words"  # Top 1000
    elif epoch < max_epochs * 0.7:
        return "medium_words"  # 4-7 chars
    else:
        return "all_words"    # Full dataset
```

### 2. Implement Scheduled Sampling
```python
# During training, gradually reduce teacher forcing
teacher_forcing_ratio = max(0.1, 1.0 - epoch / 100)
if random.random() < teacher_forcing_ratio:
    use_ground_truth_prefix()
else:
    use_model_predictions()
```

### 3. Add RNN-T Specific Augmentations
```python
class RNNTAugmentation:
    def blank_insertion(self, labels, prob=0.1):
        """Insert blanks between characters during training"""
        augmented = []
        for label in labels:
            augmented.append(label)
            if random.random() < prob:
                augmented.append(blank_id)
        return augmented

    def prefix_training(self, word, min_len=1):
        """Train on prefixes to learn sequence generation"""
        prefix_len = random.randint(min_len, len(word))
        return word[:prefix_len]
```

### 4. Fix Validation Metrics
```python
# Current validation uses same data distribution
# Should use:
- Separate OOV (out-of-vocabulary) test set
- Per-length WER buckets (2-3, 4-5, 6-7, 8+ chars)
- First-char accuracy vs full-word accuracy metrics
```

### 5. Add Beam Search During Validation
```python
# Currently only using greedy decode for validation
# Add beam search validation every N epochs:
if epoch % 10 == 0:
    beam_wer = validate_with_beam_search(beam_size=10)
    if beam_wer < best_beam_wer:
        save_checkpoint("best_beam_model.ckpt")
```

## 🔧 Configuration Improvements

### 1. Optimizer Settings
```python
"optim": {
    "name": "adamw",
    "lr": 5e-4,  # Increased from 2e-4
    "betas": [0.9, 0.98],
    "weight_decay": 1e-4,  # Reduced from 1e-3
    # Add:
    "grad_clip": 5.0,  # Gradient clipping for stability
}
```

### 2. Learning Rate Schedule
```python
"scheduler": {
    "name": "CosineAnnealingWarmRestarts",
    "T_0": 50,  # Restart every 50 epochs
    "T_mult": 2,  # Double period after each restart
    "warmup_steps": 1000,  # Critical for RNN-T
}
```

### 3. Loss Modifications
```python
# Add label smoothing for RNN-T
"loss": {
    "_target_": "nemo.collections.asr.losses.rnnt_loss.RNNTLoss",
    "blank_weight": 0.1,  # Reduce blank preference
    "label_smoothing": 0.1,  # Add smoothing
}
```

## 📊 Training Strategy Recommendations

### Phase 1: Bootstrap (Epochs 0-50)
- Focus on short words (2-4 characters)
- High learning rate (1e-3) with warmup
- Heavy augmentation
- Prefix training enabled

### Phase 2: Expansion (Epochs 50-150)
- Gradually include longer words
- Reduce learning rate to 5e-4
- Add blank insertion augmentation
- Enable scheduled sampling

### Phase 3: Refinement (Epochs 150-300)
- Full dataset with frequency weighting
- Learning rate 1e-4 with cosine annealing
- Focus on rare words and edge cases
- Beam search validation

### Phase 4: Fine-tuning (Epochs 300+)
- Very low learning rate (1e-5)
- Focus on worst-performing examples
- Knowledge distillation if available
- Ensemble training

## 🚀 Quick Fixes for Immediate Improvement

1. **Increase learning rate**: Change from 2e-4 to 5e-4
2. **Reduce batch size**: From 1000 to 256
3. **Add warmup**: 1000 steps minimum
4. **Add gradient clipping**: Set to 5.0
5. **Reduce blank weight**: Add blank_weight=0.1 to loss

## 💡 Debugging Recommendations

1. **Log more metrics**:
   - Per-frame prediction distribution
   - Blank emission rate per epoch
   - Character transition probabilities
   - Attention weights if using attention

2. **Visualize predictions**:
   - Save sample predictions every 10 epochs
   - Plot confusion matrices for common errors
   - Track which words improve/degrade over time

3. **Add checkpointing**:
   - Save best model per character length
   - Save best model per frequency bucket
   - Keep models from each curriculum stage

## Example Modified Training Command

```bash
python train_transducer_personalized.py \
    --model-size tablet \
    --profile curriculum_stage1 \
    --augment \
    --learning-rate 5e-4 \
    --batch-size 256 \
    --warmup-steps 1000 \
    --grad-clip 5.0 \
    --blank-weight 0.1 \
    --max-epochs 300
```

## Summary

The main issue is that the model is stuck in a local minimum where it only learns to map starting positions to first characters. This requires:
1. More aggressive optimization (higher LR, smaller batches)
2. Curriculum learning (start simple, increase complexity)
3. RNN-T specific training techniques (prefix training, blank insertion)
4. Better metrics to track actual sequence generation capability

The coordinate transformation and feature extraction are correct. The problem is purely in the training dynamics.