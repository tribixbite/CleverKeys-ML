import os
import torch
import torch.quantization as tq
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import necessary components from the training script
# This assumes train_production.py is in the same directory
from train_production import CONFIG, CharTokenizer, KeyboardGrid, SwipeFeaturizer, SwipeDataset, collate_fn, GestureCTCModel

def quantize_for_pte():
    """
    Performs Post-Training Static Quantization (PTSQ) on the trained model
    and exports it to the ExecuTorch (.pte) format for Android deployment.
    """
    print("--- Starting Post-Training Static Quantization for PTE ---")
    device = torch.device("cpu") # Quantization is a CPU-bound process
    print(f"Using device: {device}")

    # --- 1. Load the Trained Floating-Point Model ---
    print("Loading the best floating-point model...")
    tokenizer = CharTokenizer(CONFIG["chars"])
    grid = KeyboardGrid(CONFIG["chars"])
    input_dim = 10 + grid.num_keys
    
    model_fp32 = GestureCTCModel(
        input_dim=input_dim,
        num_classes=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"]
    )
    
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
        
    model_fp32.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_fp32.eval()
    print("Model loaded successfully.")

    # --- 2. Prepare Model for Static Quantization ---
    print("Preparing model for quantization...")
    # We need to create a new model instance to attach quantization specifics.
    # We add QuantStub and DeQuantStub to explicitly mark where quantization
    # should begin and end.
    class QuantizableGestureModel(GestureCTCModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
            src = self.quant(src)
            # The base model forward path
            x = self.input_projection(src)
            x = self.pos_encoder(x.permute(1, 0, 2))
            output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            logits = self.ctc_head(output)
            log_probs = F.log_softmax(logits, dim=2)
            log_probs = self.dequant(log_probs)
            return log_probs

    model_to_quantize = QuantizableGestureModel(
        input_dim=input_dim, num_classes=tokenizer.vocab_size, d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"], num_encoder_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"], dropout=CONFIG["dropout"]
    )
    model_to_quantize.load_state_dict(model_fp32.state_dict())
    model_to_quantize.eval()
    
    # Specify quantization configuration. 'fbgemm' is the recommended backend for x86 CPUs.
    model_to_quantize.qconfig = tq.get_default_qconfig('fbgemm')
    print("Default qconfig set.")
    
    # Fuse modules for better performance (Conv-BN-ReLU, etc.). 
    # Our model is simpler, but this is best practice.
    # Example: tq.fuse_modules(model_to_quantize, [['conv', 'relu']], inplace=True)
    
    tq.prepare(model_to_quantize, inplace=True)
    print("Model prepared with quantization observers.")

    # --- 3. Calibrate the Model ---
    print("Calibrating the model with validation data...")
    # Static quantization requires feeding a small amount of representative data
    # through the model to observe the range of activations.
    featurizer = SwipeFeaturizer(grid)
    val_dataset = SwipeDataset(CONFIG["val_data_path"], featurizer, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=10, collate_fn=collate_fn)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=min(100, len(val_loader)), desc="Calibration"):
            if batch is None or i >= 100: # Calibrate on ~100 batches
                break
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            src_key_padding_mask = (torch.arange(features.shape[1])[None, :].to(device) >= feature_lengths[:, None])
            model_to_quantize(features, src_key_padding_mask)
    print("Calibration complete.")

    # --- 4. Convert to a Quantized Model ---
    print("Converting the model to a quantized integer representation...")
    model_int8 = tq.convert(model_to_quantize, inplace=True)
    model_int8.eval()
    print("Model converted to INT8.")

    # --- 5. Export for ExecuTorch ---
    print("\n--- Exporting Quantized Model for Production (PTE) ---")
    export_dir = CONFIG["export_dir"]
    os.makedirs(export_dir, exist_ok=True)
    
    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 100, input_dim) # (batch, seq_len, features)
    dummy_mask = torch.zeros(1, 100, dtype=torch.bool) # No padding

    # 1. Export to TorchScript
    try:
        # Important: Use torch.jit.script for quantized models, not trace
        scripted_model = torch.jit.script(model_int8)
        torchscript_path = os.path.join(export_dir, "model.quant.pt")
        scripted_model.save(torchscript_path)
        print(f"Successfully exported quantized model to TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")
        return

    # 2. Provide final instructions for ExecuTorch conversion
    print("\n--- Next Steps: Convert to .pte using ExecuTorch toolchain ---")
    print("Run the following commands in your terminal (with ExecuTorch SDK environment):")
    print(f"1. Lower to EDGE Dialect:\n   python -m executorch.exir.capture --model_path {torchscript_path} --output_path {os.path.join(export_dir, 'model.quant.edge.pt')}")
    print(f"2. Convert to PTE format:\n   executorch-compiler -m {os.path.join(export_dir, 'model.quant.edge.pt')} -o {os.path.join(export_dir, 'model.quant.pte')}")
    print("\nThe final deployable file will be 'exports/model.quant.pte'")


if __name__ == "__main__":
    quantize_for_pte()
