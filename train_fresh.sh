#!/bin/bash
# Script to start fresh training

echo "Starting fresh training run..."

# Option 1: Start completely fresh with new run name
python train_squeezeformer_ctc.py --ignore-checkpoint

# Option 2: Remove old checkpoints and start fresh
# rm -rf 12-02-squeeze/checkpoints/*.ckpt
# python train_squeezeformer_ctc.py

# Option 3: Start with smaller batch size for testing
# python train_squeezeformer_ctc.py --ignore-checkpoint --batch-size 256