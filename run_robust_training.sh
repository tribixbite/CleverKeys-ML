#!/bin/bash
# Robust training script with memory management and automatic resumption

# Configuration
CHECKPOINT_DIR="./rnnt_checkpoints_short_common_20250921_143403/lightning_logs/version_0/checkpoints"
BATCH_SIZE=512  # Reduced from 800
NUM_WORKERS=4   # Reduced from 10
MAX_EPOCHS_PER_RUN=10  # Limit epochs per run to prevent memory accumulation

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# Function to get the latest checkpoint
get_latest_checkpoint() {
    ls -t $CHECKPOINT_DIR/epoch=*.ckpt 2>/dev/null | head -1
}

# Function to run training with automatic restart
run_training() {
    while true; do
        # Get the latest checkpoint
        LATEST_CHECKPOINT=$(get_latest_checkpoint)

        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo "Starting fresh training..."
            CHECKPOINT_ARG=""
        else
            echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
            CHECKPOINT_ARG="--checkpoint $LATEST_CHECKPOINT"
        fi

        # Run training with memory limits
        echo "Starting training run at $(date)"

        # Use timeout to limit each run to 2 hours max
        timeout 2h uv run python new/train_transducer_personalized.py \
            $CHECKPOINT_ARG \
            --profile short_common \
            --val-profile validation_balanced \
            --batch-size $BATCH_SIZE \
            --num-workers $NUM_WORKERS \
            --max-epochs $MAX_EPOCHS_PER_RUN

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Training completed successfully"
            break
        elif [ $EXIT_CODE -eq 124 ]; then
            echo "Training timeout reached, restarting..."
        elif [ $EXIT_CODE -eq 137 ]; then
            echo "Training killed (likely OOM), restarting with cleanup..."
            # Clear cache and wait before restart
            sync
            echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
            sleep 10
        else
            echo "Training failed with exit code $EXIT_CODE"
            break
        fi

        # Small delay before restart
        sleep 5
    done
}

# Main execution
echo "=== Robust Training Script ==="
echo "Batch Size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Max Epochs per Run: $MAX_EPOCHS_PER_RUN"
echo "================================"

run_training