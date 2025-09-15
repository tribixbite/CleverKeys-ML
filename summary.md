# Training Script Issues and Fixes Summary

## Problems Found in Original train.py

### 1. **KenLM Language Model Download Failure**
**Issue:** The script attempted to download a language model from `kensho/kenlm` repository that no longer exists or has been moved.
```python
lm_path = hf_hub_download(repo_id="kensho/kenlm", filename=f"lm/en_us/4-gram-small.arpa")
```
**Error:** `RepositoryNotFoundError: 404 Client Error`
**Impact:** Complete training failure - script couldn't proceed without the language model

### 2. **Deprecated PyTorch API Usage**
**Issue A:** Using deprecated `torch.cuda.amp.GradScaler` instead of new API
```python
scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["mixed_precision"])
```
**Warning:** `FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated`

**Issue B:** Using deprecated `torch.cuda.amp.autocast` instead of new API
```python
with torch.cuda.amp.autocast(enabled=CONFIG["mixed_precision"]):
```
**Warning:** `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated`

### 3. **Dimension Mismatch in Model Input**
**Issue:** Incorrect calculation of input dimensions for the model
```python
input_dim = 10 + grid.num_keys  # Wrong: assumed 10 base features
```
**Actual features:** Only 8 base features (x, y, dx, dy, vx, vy, ax, ay) + 27 one-hot keys = 35 total
**Error:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (48384x35 and 37x256)`
**Impact:** Model couldn't process input tensors

### 4. **Inconsistent Tensor Format in Transformer**
**Issue:** Mixing batch_first=True in TransformerEncoder but using seq_first tensor operations
```python
# Created encoder with batch_first=True
encoder_layer = nn.TransformerEncoderLayer(..., batch_first=True)
# But then permuted as if batch_first=False
x = self.pos_encoder(x.permute(1, 0, 2))
```
**Error:** `AssertionError: Expected key_padded_mask.shape[0] to be 177, but got 256`
**Impact:** Attention mask dimensions didn't match expected format

### 5. **PositionalEncoding Dimension Mismatch**
**Issue:** PositionalEncoding expected (seq_len, batch, d_model) but received (batch, seq_len, d_model)
```python
pe = torch.zeros(max_len, 1, d_model)  # Wrong shape for batch_first
```
**Impact:** Incorrect positional encoding application

### 6. **Learning Rate Scheduler Warning**
**Issue:** scheduler.step() called before first optimizer.step()
```python
scheduler.step()  # Called immediately in training loop
```
**Warning:** `Detected call of lr_scheduler.step() before optimizer.step()`
**Impact:** First learning rate value skipped

### 7. **NaN Loss Values**
**Issue:** CTC loss returning NaN due to sequence length constraints
- CTC requires input_lengths >= target_lengths
- No sequence length filtering
- No gradient clipping
**Impact:** Training doesn't converge, all losses are NaN

### 8. **Missing Sequence Length Filtering**
**Issue:** No maximum sequence length enforcement, allowing very long sequences that cause memory issues and NaN losses
**Impact:** Unstable training, potential out-of-memory errors

## Fixes Applied

### 1. **KenLM Model Fix**
```python
try:
    lm_path = hf_hub_download(repo_id="edugp/kenlm", filename="en.arpa.bin")
except:
    try:
        lm_path = hf_hub_download(repo_id="microsoft/unigram", filename="unigram_en.arpa")
    except:
        print("Warning: Could not download language model. Using vocabulary-only decoding.")
        lm_path = None
```

### 2. **Updated PyTorch APIs**
```python
# Updated GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["mixed_precision"])

# Updated autocast
with torch.amp.autocast('cuda', enabled=CONFIG["mixed_precision"]):
```

### 3. **Fixed Input Dimensions**
```python
input_dim = 8 + grid.num_keys  # Corrected to 8 base features
```

### 4. **Fixed Transformer Tensor Handling**
```python
def forward(self, src, src_key_padding_mask):
    x = self.input_projection(src)
    x = self.pos_encoder(x)  # Keep in batch_first format
    output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
    logits = self.ctc_head(output)
    log_probs = F.log_softmax(logits, dim=2)
    return log_probs.permute(1, 0, 2)  # Permute only at the end for CTC
```

### 5. **Fixed PositionalEncoding for batch_first**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Changed to (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Adjusted indexing for batch_first
        return self.dropout(x)
```

### 6. **Added Gradient Clipping and Length Constraints**
```python
# Ensure minimum sequence lengths for CTC
feature_lengths = torch.maximum(feature_lengths, target_lengths)

# Add gradient clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

### 7. **Added Sequence Length Filtering**
```python
class SwipeDataset(Dataset):
    def __init__(self, jsonl_path, featurizer, tokenizer, max_seq_length=200):
        self.max_seq_length = max_seq_length
        # ...
    
    def __getitem__(self, idx):
        # Filter out sequences that are too long
        if len(data['points']) > self.max_seq_length:
            return None
```

### 8. **Added Global Step Counter**
```python
global_step = 0  # Track global steps for scheduler
# This helps with proper scheduler initialization
```

## Areas for Improvement

### 1. **Data Quality Issues**
- **Problem:** Persistent NaN losses suggest data quality issues
- **Recommendations:**
  - Add data validation to check for:
    - Empty or very short sequences
    - Invalid coordinate values (NaN, Inf, or out of bounds)
    - Mismatched word/gesture pairs
  - Implement robust data preprocessing pipeline
  - Add statistics logging for debugging

### 2. **Better Error Handling**
- **Current:** Silent failures and NaN propagation
- **Recommendations:**
  ```python
  if torch.isnan(loss):
      logger.warning(f"NaN loss detected at step {global_step}")
      # Skip this batch or implement recovery strategy
  ```

### 3. **Improved Sequence Length Handling**
- **Current:** Hard filtering at max_seq_length
- **Recommendations:**
  - Dynamic batching by sequence length
  - Adaptive sequence truncation instead of dropping
  - Better padding strategies

### 4. **Enhanced Model Architecture**
- **Potential improvements:**
  - Layer normalization before CTC head
  - Dropout regularization in more places
  - Learnable positional encodings
  - Multi-scale feature extraction

### 5. **Training Stability**
- **Recommendations:**
  - Implement warm-up with smaller learning rate
  - Use gradient accumulation for larger effective batch size
  - Add learning rate reduction on plateau
  - Implement early stopping

### 6. **Better Logging and Monitoring**
```python
# Add comprehensive metrics
writer.add_histogram('gradients', grad_norm, global_step)
writer.add_scalar('Loss/train_non_nan_ratio', non_nan_ratio, epoch)
writer.add_scalar('Data/avg_seq_length', avg_seq_len, epoch)
```

### 7. **Configuration Management**
- Move CONFIG to external YAML/JSON file
- Add configuration validation
- Support multiple experiment configs

### 8. **Code Organization**
- Split into multiple modules (model.py, dataset.py, utils.py)
- Add unit tests for critical components
- Implement proper logging instead of print statements

### 9. **Performance Optimizations**
- **Current issues:**
  - No data prefetching optimization
  - Inefficient collate function
- **Recommendations:**
  - Use persistent_workers=True in DataLoader
  - Optimize feature extraction with vectorization
  - Cache preprocessed features

### 10. **Robustness Improvements**
```python
# Add checkpoint recovery
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
```

## Conclusion

The original training script had multiple critical issues that prevented it from running:
1. External dependency failures (KenLM model)
2. API deprecation warnings
3. Dimension mismatches in model architecture
4. Tensor format inconsistencies

After fixing these issues, the training runs but still exhibits NaN losses, indicating deeper data quality or numerical stability issues that require further investigation. The script would benefit from modularization, better error handling, and comprehensive data validation to make it production-ready.