# Export Architecture Documentation

## Overview

This document provides a comprehensive overview of the CleverKeys model export and quantization pipeline. The system supports multiple deployment targets with different precision levels and optimization strategies.

## Export Philosophy

The CleverKeys export architecture is designed around **deployment flexibility** and **performance optimization**:

- **Multi-Target**: Support for web (ONNX), Android (ExecuTorch PTE), and development (PyTorch)
- **Multi-Precision**: FP32, FP16, INT8 quantization with calibration
- **Component Isolation**: Separate encoder and decoder exports for modular deployment
- **Evolution-Aware**: Backward compatibility with initial exports while providing advanced optimizations

## Architecture Overview

```
Training Model (.ckpt/.nemo)
│
├── Encoder Export Pipeline
│   ├── ONNX Exports (Web/Server)
│   │   ├── export_onnx.py          (FP32 + INT8 Quantization)
│   │   └── export_optimized_onnx.py (Ultra-Optimized Multi-Precision)
│   │
│   └── ExecuTorch PTE Exports (Android)
│       ├── export_pte_fp32.py       (FP32 Baseline)
│       ├── export_pte.py            (PT2E + XNNPACK Quantization)
│       ├── export_pte_optimized.py  (Non-Quantized Optimized)
│       └── export_pte_ultra.py      (Ultra-Optimized Quantized)
│
└── Decoder Export Pipeline
    └── export_rnnt_step.py          (Single-Step RNN-T Decoder)
```

---

## File-by-File Analysis

### Core Encoder Exports

#### 1. `export_onnx.py` - Standard ONNX with Quantization
**Purpose**: Baseline ONNX export with optional INT8 quantization for web/server deployment.

**Key Features**:
- **Target**: Web browsers, ONNX Runtime inference
- **Precision**: FP32 baseline + INT8 quantization via ONNX Runtime
- **Quantization Method**: Static quantization with calibration dataset
- **Format**: QDQ (Quantize-Dequantize) format for hardware acceleration
- **Model Wrapping**: `EncoderWrapper` isolates encoder for export

**Technical Implementation**:
```python
class EncoderWrapper(torch.nn.Module):
    """Wraps the ConformerEncoder: forward(B,F,T), lengths(B) -> (encoded, encoded_len)"""
    def forward(self, feats_bft: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(audio_signal=feats_bft, length=lengths)

# Custom calibration data reader for swipe gesture data
class SwipeCalibrationDataReader(CalibrationDataReader):
    def get_next(self):
        feats_bft, lens = _to_encoder_inputs(batch)
        return {
            "features_bft": feats_bft.cpu().numpy(),
            "lengths": lens.cpu().numpy(),
        }
```

**Quantization Strategy**:
- **Static Quantization**: Uses representative calibration dataset
- **INT8 QDQ**: Per-channel quantization for optimal accuracy
- **ONNX Runtime**: Leverages ORT's quantization infrastructure

**Usage**:
```bash
python export_onnx.py --nemo_model checkpoint.nemo \
                      --val_manifest data/val.jsonl \
                      --vocab vocab.txt \
                      --fp32_onnx encoder_fp32.onnx \
                      --output encoder_int8_qdq.onnx
```

**Evolution**: This was one of the initial export scripts, enhanced with proper calibration data handling and QDQ format support.

---

#### 2. `export_optimized_onnx.py` - Ultra-Optimized Multi-Precision
**Purpose**: Advanced ONNX export with multiple optimization levels and precision formats.

**Key Features**:
- **Multi-Precision**: FP32, FP16, INT8 with advanced quantization
- **Ultra-Optimization**: Graph optimization passes, operator fusion
- **Calibration**: Representative dataset-based quantization calibration
- **Web-Optimized**: Smaller models, faster initialization for browsers
- **Deployment Analysis**: Built-in size analysis and compression estimation

**Technical Implementation**:
```python
class OptimizedEncoderWrapper(torch.nn.Module):
    """Optimized encoder wrapper for ONNX export"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval()

class FastCalibrationReader(CalibrationDataReader):
    """Fast calibration data reader with minimal data"""
    def __init__(self, data_loader, max_samples=16):
        self.max_samples = max_samples  # Reduced for speed

def audit_onnx_initializers(onnx_path):
    """Audit ONNX initializer sizes and types to verify quantization"""
    # Verifies quantization by checking for int8 dtypes
    # Reports model size breakdown and optimization effectiveness
```

**Advanced Features**:
- **ONNX Runtime Graph Optimization**: All optimization passes enabled
- **External Data Storage**: Optional for smaller file sizes
- **Quantization Verification**: Audits initializers to confirm INT8 presence
- **Compression Analysis**: Estimates APK compressed size using zlib
- **Platform-Specific**: Web (parallel execution) vs Android (sequential) optimization

**Multi-Target Export**:
```bash
# Create both web and Android optimized versions
python export_optimized_onnx.py --checkpoint checkpoint.ckpt \
                                --val_manifest data/val.jsonl \
                                --vocab vocab.txt \
                                --web_onnx encoder_web_quant.onnx \
                                --android_onnx encoder_android_quant.onnx \
                                --create_ort_optimized \
                                --compression_test
```

**Evolution**: Latest generation export with comprehensive optimization analysis, multi-target support, and built-in verification tools.

---

### ExecuTorch PTE Exports (Android)

#### 3. `export_pte_fp32.py` - FP32 Baseline PTE
**Purpose**: Non-quantized ExecuTorch export for Android development and testing.

**Key Features**:
- **Target**: Android via ExecuTorch runtime
- **Precision**: FP32 (no quantization)
- **Backend**: XNNPACK acceleration
- **Use Case**: Development, debugging, accuracy baseline

**Usage**:
```bash
python export_pte_fp32.py --nemo_model checkpoint.nemo --out encoder_fp32.pte
```

---

#### 4. `export_pte.py` - PT2E + XNNPACK Quantization
**Purpose**: Standard quantized ExecuTorch export using PT2E (PyTorch 2.0 Export) pipeline.

**Key Features**:
- **Target**: Production Android deployment
- **Quantization**: PT2E quantization flow with XNNPACK
- **Backend**: XNNPACK acceleration for ARM
- **Calibration**: Representative data calibration

**Technical Implementation**:
```python
# PT2E quantization workflow
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

# Export -> Prepare -> Calibrate -> Convert -> ExecuTorch
exported_model = torch.export.export(wrapped_encoder, (example_feats_bft, example_lens))
prepared_encoder = prepare_pt2e(exported_model.module(), quantizer)

# Calibration with gesture data
@torch.no_grad()
def calibrate_encoder(prepared_encoder, val_manifest, vocab_path, num_batches=64):
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        feats_bft, lens = _to_encoder_inputs(batch)
        prepared_encoder(feats_bft, lens)

quant_encoder = convert_pt2e(prepared_encoder).eval()
```

**Quantization Pipeline**:
1. **torch.export**: Export model to ExportedProgram
2. **prepare_pt2e**: Insert observers for calibration
3. **Calibration**: Run representative data through prepared model
4. **convert_pt2e**: Convert to quantized version
5. **to_edge**: Lower to ExecuTorch Edge IR
6. **XNNPACK Backend**: Partition for ARM acceleration

**Usage**:
```bash
python export_pte.py --nemo_model checkpoint.nemo \
                     --val_manifest data/val.jsonl \
                     --vocab vocab.txt \
                     --output encoder_quantized.pte \
                     --calib_batches 64
```

**Evolution**: Enhanced from initial version with proper PT2E workflow, comprehensive calibration, and XNNPACK optimization.

---

#### 5. `export_pte_optimized.py` - Non-Quantized Optimized
**Purpose**: Highly optimized FP32/FP16 ExecuTorch export without quantization.

**Key Features**:
- **Precision**: FP32/FP16 (no quantization)
- **Optimization**: Graph optimization, operator fusion
- **Performance**: Optimized for speed without quantization artifacts
- **Debugging**: Maintains full precision for analysis

**Usage**:
```bash
python export_pte_optimized.py --checkpoint checkpoint.ckpt \
                               --precision fp16
```

---

#### 6. `export_pte_ultra.py` - Ultra-Optimized Quantized
**Purpose**: Maximum performance ExecuTorch export with aggressive optimization and quantization.

**Key Features**:
- **Performance**: Blazing fast Android performance
- **Quantization**: Advanced quantization with multiple calibration methods
- **Optimization**: All available optimization passes enabled
- **Size**: Minimal model size for mobile deployment

**Advanced Technical Features**:
```python
class OptimizedEncoderWrapper(torch.nn.Module):
    """Ultra-optimized encoder wrapper for maximum Android performance"""
    def forward(self, audio_signal, length):
        # Direct forward without extra processing for maximum speed
        return self.encoder(audio_signal=audio_signal, length=length)

def create_calibration_data(model, num_samples=32, use_realistic_patterns=True):
    """Create diverse calibration data for robust quantization"""
    # Simulate realistic gesture-like patterns
    # Position features (smoother, gesture-like movement)
    base_signal = torch.sin(torch.linspace(0, 2*torch.pi, T)) * 0.3
    # Key features (binary-ish, some keys activated)
    audio_signal[0, key_idx, activation_start:activation_start+activation_len] = \
        torch.rand(activation_len) * 0.8 + 0.2

def verify_quantization(program, program_name="Program"):
    """Verify that the program contains quantization ops"""
    # Analyzes graph for quantize/dequantize ops
    # Reports quantization effectiveness and optimization ratio

def validate_xnnpack_partition(edge_program):
    """Validate XNNPACK partitioning effectiveness"""
    # Measures percentage of ops running on XNNPACK vs fallback
    # Warns if partition ratio is suboptimal
```

**Optimization Pipeline**:
1. **Realistic Calibration**: Creates gesture-like data patterns for robust quantization
2. **Ultra-Aggressive Quantization**: Symmetric per-channel INT8 with XNNPACK
3. **Quantization Verification**: Confirms INT8 ops are present in the graph
4. **XNNPACK Validation**: Ensures high partition ratio for ARM optimization
5. **Compression Analysis**: Estimates APK size with zlib compression
6. **Fallback Support**: Graceful degradation to FP32 if quantization fails

**Usage**:
```bash
python export_pte_ultra.py --checkpoint checkpoint.ckpt \
                           --output encoder_ultra_quant.pte \
                           --calibration_samples 32 \
                           --fallback_to_fp32
```

**Evolution**: Latest generation with comprehensive quantization analysis, validation tools, and bulletproof fallback mechanisms.

---

### Decoder Export

#### 7. `export_rnnt_step.py` - RNN-T Single-Step Decoder
**Purpose**: Export the RNN-T decoder+joint for single-step inference.

**Key Features**:
- **Component**: Decoder prediction network + joint network
- **Format**: Both ONNX and ExecuTorch PTE
- **Precision**: FP32 (decoder typically kept full precision)
- **Use Case**: Streaming inference, autoregressive decoding

**Technical Implementation**:
```python
class RNNTStep(torch.nn.Module):
    """
    One decoding step:
      Inputs: y_prev_ids(B,), h0/c0(L,B,H), enc_t(B,D)
      Outputs: logits(B,V), h1/c1(L,B,H)
    """
    def __init__(self, model):
        # Extract decoder components
        self.embedding = self._find_embedding(model.decoder)
        self.lstm = self._find_lstm(model.decoder)
        self.joint = model.joint

    def forward(self, y_prev_ids, h0, c0, enc_t):
        # Embedding lookup
        emb = self.embedding(y_prev_ids)
        x = emb.unsqueeze(0)  # (1,B,E) for LSTM

        # LSTM forward
        out, (h1, c1) = self.lstm(x, (h0, c0))
        pred_t = out.squeeze(0)  # (B,H)

        # Joint network (NeMo format)
        enc_t_3d = enc_t.unsqueeze(-1)    # (B,D,1)
        pred_t_3d = pred_t.unsqueeze(-1)  # (B,H,1)
        logits = self.joint(encoder_outputs=enc_t_3d,
                          decoder_outputs=pred_t_3d)
        return logits.squeeze(-1), h1, c1
```

**Architecture Discovery**:
- **Automatic Component Detection**: Finds embedding and LSTM modules in NeMo decoder
- **Vocabulary Support**: Derives blank_id from provided vocabulary file
- **Dimension Inference**: Automatically determines encoder dimensions via sample forward pass

**Inputs/Outputs**:
```python
# Inputs:
y_prev_ids: (B,) int64      # Previous token IDs (starts with blank_id)
h0, c0: (L,B,H) float32     # LSTM hidden states
enc_t: (B,D) float32        # Encoder frame output

# Outputs:
logits: (B,V) float32       # Vocabulary logits
h1, c1: (L,B,H) float32     # Updated LSTM states
```

**Usage**:
```bash
python export_rnnt_step.py --nemo_model checkpoint.nemo \
                           --onnx_out decoder.onnx \
                           --pte_out decoder.pte \
                           --vocab vocab.txt \
                           --layers 2 --hidden 256
```

**Evolution**: Enhanced with automatic architecture discovery, vocabulary integration, and robust dimension inference.

---

## Export Evolution Timeline

### Initial Phase (Pre-Improvements)
- **Basic Exports**: Simple ONNX and PTE exports
- **Limited Optimization**: Minimal graph optimization
- **No Beam Search**: Basic character-level decoding only
- **Single Precision**: Primarily FP32 exports

### Enhanced Phase (Current)
- **Wordlist Integration**: 150K+ word vocabulary with beam search
- **Multi-Precision**: FP32, FP16, INT8 with proper calibration
- **Advanced Optimization**: Graph fusion, operator optimization
- **Deployment-Specific**: Tailored exports for web vs. mobile

---

## Deployment Target Matrix

| Target | Recommended Export | Precision | Features |
|--------|-------------------|-----------|----------|
| **Web Development** | `export_onnx.py` | FP32 | Full precision, debugging |
| **Web Production** | `export_optimized_onnx.py` | INT8 | Small size, fast loading |
| **Android Development** | `export_pte_fp32.py` | FP32 | Accuracy baseline |
| **Android Production** | `export_pte_ultra.py` | INT8 | Maximum performance |
| **Server Inference** | `export_optimized_onnx.py` | FP16 | GPU acceleration |
| **Edge Devices** | `export_pte_ultra.py` | INT8 | Minimal footprint |

---

## Key Components

### Model Wrapping Architecture
All exports use wrapper classes to isolate components:
```python
# Standard encoder wrapper pattern
class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval()
    def forward(self, feats_bft, lengths):
        return self.encoder(audio_signal=feats_bft, length=lengths)

# Optimized variants add performance enhancements
class OptimizedEncoderWrapper(EncoderWrapper):
    def forward(self, features_bft, lengths):
        # Direct forward without extra processing for maximum speed
        return self.encoder(audio_signal=features_bft, length=lengths)
```

### Quantization Strategies

#### 1. ONNX Runtime Quantization (QDQ Format)
- **Method**: Static quantization with calibration
- **Format**: QDQ (Quantize-Dequantize) for hardware acceleration
- **Backend**: ONNX Runtime's quantization infrastructure
- **Precision**: INT8 symmetric per-channel quantization

#### 2. PT2E + XNNPACK Quantization
```python
# PyTorch 2.0 Export quantization workflow
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

# Five-stage pipeline
exported_model = torch.export.export(model, example_inputs)
prepared_model = prepare_pt2e(exported_model.module(), quantizer)
# [Calibration phase]
quantized_model = convert_pt2e(prepared_model)
edge_program = to_edge(quantized_model)
```

#### 3. Calibration Data Generation
```python
def create_calibration_data(model, num_samples=32, use_realistic_patterns=True):
    """Creates gesture-like calibration patterns"""
    # Kinematic features (first 9 channels)
    base_signal = torch.sin(torch.linspace(0, 2*torch.pi, T)) * 0.3
    audio_signal[0, dim, :] = base_signal + noise

    # Key activation patterns (channels 9-37)
    active_keys = torch.randint(9, 37, (num_active_keys,))
    audio_signal[0, key_idx, start:end] = torch.rand(length) * 0.8 + 0.2
```

### Optimization Passes

#### Graph-Level Optimizations
- **ONNX Runtime**: All optimization levels enabled (`ORT_ENABLE_ALL`)
- **ExecuTorch**: Edge IR optimizations + backend-specific passes
- **Operator Fusion**: Combines sequential operations for efficiency

#### Backend-Specific Optimizations
```python
# Platform-specific session options
if target_platform == "web":
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # Web workers
else:  # android
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Mobile
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
```

#### XNNPACK Partitioning
- **ARM Optimization**: Leverages NEON instructions on mobile
- **Partition Validation**: Ensures high XNNPACK coverage (>50%)
- **Fallback Handling**: Graceful degradation for unsupported ops

### Verification and Analysis Tools

#### Quantization Verification
```python
def verify_quantization(program):
    """Analyzes graph for quantization effectiveness"""
    quant_ops = sum(1 for node in graph.nodes if 'quantize' in str(node.target))
    dequant_ops = sum(1 for node in graph.nodes if 'dequantize' in str(node.target))
    return quant_ops > 0 and dequant_ops > 0
```

#### Size Analysis
- **Parameter Counting**: Tracks model size reduction through quantization
- **Compression Estimation**: Predicts APK compressed size using zlib
- **Initializer Auditing**: Verifies INT8 dtype presence in ONNX models

---

## Best Practices

### For Web Deployment
1. Use `export_optimized_onnx.py` with INT8 for production
2. Include wordlist beam search for accuracy
3. Test with representative gesture data

### For Android Deployment
1. Start with `export_pte_fp32.py` for development
2. Use `export_pte_ultra.py` for production
3. Validate quantization accuracy with calibration data

### For Development
1. Use FP32 exports for debugging and accuracy analysis
2. Compare quantized vs. full precision outputs
3. Profile inference performance on target devices

---

## Migration Guide

### From Initial Exports to Current
1. **Add Wordlist**: Include wordlist path in export commands
2. **Enable Beam Search**: Use beam search integration for accuracy
3. **Update Calibration**: Use representative calibration datasets
4. **Test Quantization**: Validate INT8 accuracy vs. FP32 baseline

### Recommended Upgrade Path
1. Export baseline FP32 model
2. Test quantized version with same data
3. Validate accuracy meets requirements
4. Deploy optimized version to production

---

## Technical Troubleshooting

### Common Export Issues

#### 1. Quantization Failures
```bash
# Symptoms: "Quantization verification failed" or missing INT8 ops
# Solutions:
--fallback_to_fp32                    # Graceful degradation
--skip_quantization_check             # Skip verification (faster)
--calibration_samples 16              # Reduce calibration data
```

#### 2. XNNPACK Partition Issues
```bash
# Symptoms: Low partition ratio (<50%), large .pte files
# Solutions: Check model architecture compatibility
grep -r "unsupported" logs/           # Find unsupported ops
# Use export_pte_fp32.py for debugging  # Non-quantized baseline
```

#### 3. Memory Issues During Export
```bash
# Symptoms: OOM errors, slow export
# Solutions:
--max_trace_len 100                   # Reduce sequence length
--calib_batches 32                    # Fewer calibration batches
--calib_bs 2                          # Smaller batch size
```

#### 4. torch.compile Key Conflicts
```python
# Symptoms: "encoder._orig_mod." keys in state_dict
# Automatic cleanup in all export scripts:
for key, value in state_dict.items():
    if key.startswith("encoder._orig_mod."):
        new_key = key.replace("encoder._orig_mod.", "encoder.")
        new_state_dict[new_key] = value
```

### Performance Optimization Tips

#### ONNX Runtime Optimization
```python
# Web deployment (parallel execution)
so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Android deployment (sequential + memory optimization)
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.enable_mem_pattern = True
so.enable_cpu_mem_arena = True
```

#### ExecuTorch Optimization
```bash
# Aggressive optimization pipeline
python export_pte_ultra.py \
  --calibration_samples 32 \
  --skip_quantization_check \
  --fallback_to_fp32
```

### Deployment Size Analysis

#### Model Size Breakdown
- **FP32 Encoder**: ~15-50MB (depending on model size)
- **INT8 Quantized**: ~4-15MB (75% reduction)
- **APK Compressed**: ~2-8MB (additional 50% reduction)
- **Decoder+Joint**: ~1-5MB (kept FP32 for accuracy)

#### Compression Estimates
```python
# Built-in compression analysis
--compression_test                    # Estimates APK size
--create_ort_optimized               # Additional 10-30% reduction
--external_data                      # Splits large tensors to separate files
```

---

## Evolution Summary

### Phase 1: Initial Exports (Pre-Improvements)
- **Basic ONNX/PTE**: Simple exports without optimization
- **Limited Quantization**: Basic INT8 without proper calibration
- **No Beam Search**: Character-level decoding only
- **Single Target**: One-size-fits-all approach

### Phase 2: Enhanced Exports (Current)
- **Multi-Target**: Web, Android, server-specific optimizations
- **Wordlist Integration**: 150K+ word vocabulary with beam search
- **Advanced Quantization**: Proper calibration with gesture-like data
- **Comprehensive Analysis**: Built-in verification and size estimation
- **Bulletproof Deployment**: Fallback mechanisms and error handling

### Key Improvements Made
1. **Calibration Data**: Realistic gesture patterns instead of random noise
2. **Verification Tools**: Quantization effectiveness analysis
3. **Multi-Precision**: FP32/FP16/INT8 with automatic fallback
4. **Size Optimization**: Compression analysis and external data support
5. **Platform Targeting**: Web vs Android specific optimizations
6. **Robust Error Handling**: Graceful degradation and clear diagnostics

---

This architecture provides comprehensive export capabilities while maintaining backward compatibility with initial implementations. The modular design allows incremental adoption of optimizations and new deployment targets, with built-in analysis tools to verify quantization effectiveness and predict deployment sizes.