#!/usr/bin/env python3
"""
Convert PyTorch Lightning checkpoint to .nemo format for export
"""
import torch
from model_class import GestureRNNTModel, get_default_config

def main():
    ckpt_path = "9_15_val_09.ckpt"
    nemo_path = "conformer_rnnt_gesture_9_15.nemo"

    print(f"Loading checkpoint: {ckpt_path}")

    # Load the config and create the model
    cfg = get_default_config()
    model = GestureRNNTModel(cfg)

    # Load the checkpoint state
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle torch.compile keys by removing _orig_mod prefix
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder._orig_mod."):
            new_key = key.replace("encoder._orig_mod.", "encoder.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)

    print(f"Saving as .nemo: {nemo_path}")
    model.save_to(nemo_path)
    print("✓ Conversion complete")

if __name__ == "__main__":
    main()