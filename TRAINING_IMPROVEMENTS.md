# Training Script Improvements

## Critical Fix: Export Pipeline

The primary issue is that the model trains correctly but exports incorrectly. The validation logs show the model predicting full words ('fatal', 'nearby'), but exported ONNX only outputs first characters.

### Root Cause
The export script may not be preserving the decoder's hidden state correctly between time steps, causing it to reset after each character.

### Solution
```python
# In export_stateful_pair.py, ensure decoder state is properly maintained:
def export_decoder_with_state():
    # Ensure hidden/cell states are marked as outputs AND inputs
    # so they can be fed back between inference steps
    dynamic_axes = {
        'y_prev': {0: 'batch'},
        'h_in': {0: 'layers', 1: 'batch'},
        'c_in': {0: 'layers', 1: 'batch'},
        # Critical: output states must match input states
        'h_out': {0: 'layers', 1: 'batch'},
        'c_out': {0: 'layers', 1: 'batch'},
    }
```

## Training Configuration Improvements

### 1. Reduce Batch Size
```bash
# In run_comprehensive_training.sh, line 37-38
BATCH_SIZE=512           # Reduced from 1600 for better gradient updates
NUM_WORKERS=8            # Reduced from 12 to prevent CPU bottleneck
```

### 2. Increase Learning Rate with Warmup
```python
# In train_transducer_personalized.py
cfg['optim']['lr'] = 5e-4  # Increased from 2e-4
cfg['optim']['sched'] = {
    'name': 'CosineAnnealingWarmRestarts',
    'warmup_steps': 1000,  # Critical for stability
    'warmup_ratio': 0.1,
    'T_0': 50,
    'T_mult': 2
}
```

### 3. Add RNN-T Specific Training Techniques

#### Prefix Training
```python
def augment_with_prefixes(word, prob=0.3):
    """Train on partial words to improve sequence generation"""
    if random.random() < prob and len(word) > 2:
        prefix_len = random.randint(1, len(word) - 1)
        return word[:prefix_len]
    return word
```

#### Scheduled Sampling
```python
def get_teacher_forcing_ratio(epoch, max_epochs=300):
    """Gradually reduce teacher forcing"""
    return max(0.1, 1.0 - epoch / max_epochs)
```

#### Blank Insertion Augmentation
```python
def insert_blanks(labels, blank_id=0, prob=0.1):
    """Insert blanks between characters during training"""
    augmented = []
    for label in labels:
        augmented.append(label)
        if random.random() < prob:
            augmented.append(blank_id)
    return augmented
```

### 4. Curriculum Learning Enhancement

```python
# Better curriculum stages based on word complexity
CURRICULUM_STAGES = {
    'stage1': {
        'epochs': [0, 30],
        'focus': 'short_common',  # 2-4 chars, top 1000 words
        'lr': 1e-3,
        'augmentation': 'heavy'
    },
    'stage2': {
        'epochs': [30, 100],
        'focus': 'medium_balanced',  # 4-7 chars, balanced frequency
        'lr': 5e-4,
        'augmentation': 'moderate'
    },
    'stage3': {
        'epochs': [100, 200],
        'focus': 'all_words',  # Full dataset
        'lr': 2e-4,
        'augmentation': 'light'
    },
    'stage4': {
        'epochs': [200, 300],
        'focus': 'hard_negatives',  # Words with worst performance
        'lr': 1e-4,
        'augmentation': 'none'
    }
}
```

### 5. Loss Function Improvements

```python
# In train_transducer_personalized.py
cfg['loss'] = {
    '_target_': 'nemo.collections.asr.losses.rnnt.RNNTLoss',
    'blank_weight': 0.1,  # Reduce blank preference (was implicit 1.0)
    'fastemit_lambda': 0.001,  # Encourage faster emission
    'label_smoothing': 0.1,  # Improve generalization
}
```

### 6. Gradient Clipping
```python
cfg['optim']['grad_clip'] = 5.0  # Prevent gradient explosion
```

## Monitoring Improvements

### 1. Better Metrics Logging
```python
def log_detailed_metrics(self, outputs, batch_idx):
    # Log per-character accuracy
    first_char_acc = self.calculate_first_char_accuracy(outputs)
    self.log('train_first_char_acc', first_char_acc)

    # Log blank emission rate
    blank_rate = (outputs == self.blank_id).float().mean()
    self.log('train_blank_rate', blank_rate)

    # Log sequence length statistics
    pred_lens = (outputs != self.blank_id).sum(dim=1)
    self.log('train_avg_pred_len', pred_lens.float().mean())
```

### 2. Validation Enhancements
```python
# Add beam search validation every N epochs
if epoch % 10 == 0:
    beam_results = self.validate_with_beam_search(beam_size=10)
    self.log('val_beam_wer', beam_results['wer'])

# Track per-length WER
for length_bucket in [2, 3, 4, 5, 6, 7, 8]:
    bucket_wer = self.calculate_wer_for_length(predictions, targets, length_bucket)
    self.log(f'val_wer_len_{length_bucket}', bucket_wer)
```

## Quick Implementation Priority

1. **Fix export pipeline** - Ensure decoder states are properly maintained
2. **Reduce batch size** to 512 for better gradient updates
3. **Increase learning rate** to 5e-4 with 1000-step warmup
4. **Add gradient clipping** at 5.0
5. **Set blank_weight=0.1** in loss function

## Testing the Fix

After implementing export fixes, test with:
```bash
# Test checkpoint directly (bypassing ONNX)
python test/test_checkpoint_direct.py

# Re-export with fixed state handling
python new/export_stateful_pair.py --checkpoint epoch=71

# Test new export
python test/test_onnx_inference.py
```

## Expected Improvements

With these changes:
- Export should maintain decoder state and predict full words
- Training should converge faster with better learning dynamics
- Model should handle rare words better with curriculum learning
- Validation metrics will be more informative

The key insight is that your model IS learning correctly (as shown by validation logs), but the export process is broken. Focus on fixing the export pipeline first before changing training parameters.