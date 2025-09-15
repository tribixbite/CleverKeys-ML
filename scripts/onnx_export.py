import os
import sys
import torch
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic as onnx_quantize_dynamic, QuantType

# Add parent directory to path to import from train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components from the training script
from train import CONFIG, CharTokenizer, SwipeFeaturizer, GestureConformerModel, KeyboardGrid

def quantize_for_onnx():
    """
    Performs Post-Training Dynamic Quantization on the trained model
    and exports it to the ONNX format for browser deployment.
    """
    print("--- Starting Post-Training Dynamic Quantization for ONNX ---")
    device = torch.device("cpu") # Quantization is a CPU-bound process
    print(f"Using device: {device}")

    # --- 1. Load the Trained Floating-Point Model ---
    print("Loading the best floating-point model...")
    tokenizer = CharTokenizer(CONFIG["chars"])
    grid = KeyboardGrid(CONFIG["chars"])
    featurizer = SwipeFeaturizer(grid)
    input_dim = 9 + grid.num_keys

    model_fp32 = GestureConformerModel(
        input_dim,
        tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_encoder_layers"],
        dim_ff=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"],
        conv_kernel=CONFIG["conformer_conv_kernel_size"]
    )
    
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    model_fp32.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_fp32.eval()
    print("Model loaded successfully.")

    # --- 2. Export the FP32 model to ONNX first ---
    # The ONNX Runtime quantization tools operate on an existing .onnx file.
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    onnx_fp32_path = os.path.join(export_dir, "model.fp32.onnx")
    
    print(f"Exporting FP32 model to ONNX at: {onnx_fp32_path}")
    dummy_input = torch.randn(1, 100, input_dim) # (batch, seq_len, features)
    dummy_mask = torch.zeros(1, 100, dtype=torch.bool) # No padding
    
    try:
        torch.onnx.export(
            model_fp32,
            (dummy_input, dummy_mask),
            onnx_fp32_path,
            input_names=['features', 'padding_mask'],
            output_names=['log_probs'],
            opset_version=14,
            dynamic_axes={'features' : {0 : 'batch_size', 1 : 'sequence_length'},
                          'padding_mask': {0: 'batch_size', 1: 'sequence_length'},
                          'log_probs' : {1 : 'batch_size', 0 : 'sequence_length'}}
        )
        print("FP32 ONNX export successful.")
    except Exception as e:
        print(f"Error exporting FP32 to ONNX: {e}")
        return

    # --- 3. Apply Dynamic Quantization using ONNX Runtime ---
    onnx_quant_path = os.path.join(export_dir, "model.quant.onnx")
    print(f"Applying dynamic quantization. Output: {onnx_quant_path}")
    
    try:
        onnx_quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_quant_path,
            weight_type=QuantType.QInt8
        )
        print("ONNX dynamic quantization successful.")
    except Exception as e:
        print(f"Error during ONNX quantization: {e}")
        return

    # --- 4. Verify the Quantized Model (Optional but Recommended) ---
    print("\nVerifying the quantized ONNX model...")
    try:
        ort_session = onnxruntime.InferenceSession(onnx_quant_path)
        
        # Prepare dummy inputs for ONNX Runtime
        ort_inputs = {
            'features': dummy_input.numpy(),
            'padding_mask': dummy_mask.numpy()
        }
        
        ort_outs = ort_session.run(None, ort_inputs)
        print("Verification successful: Quantized ONNX model can be loaded and run.")
        print(f"Output shape: {ort_outs[0].shape}")
    except Exception as e:
        print(f"Error verifying quantized ONNX model: {e}")

    print(f"\n--- Export Complete ---")
    print(f"The final deployable file is: '{onnx_quant_path}'")
    print("This file is ready to be used with a web runtime like ONNX.js or ONNX Runtime Web.")

if __name__ == "__main__":
    quantize_for_onnx()
