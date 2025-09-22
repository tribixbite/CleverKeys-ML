#!/bin/bash
# Multi-profile training script for comprehensive word coverage
# Cycles through different sampling profiles to ensure all word types are well-learned

# Configuration
BATCH_SIZE=400          # Conservative batch size
NUM_WORKERS=2           # Low worker count to prevent OOM
EPOCHS_PER_PROFILE=15   # Epochs to train on each profile before switching
TOTAL_CYCLES=10         # How many times to cycle through all profiles

# Training profiles array (from sampling_profiles.py)
# Each profile focuses on different aspects of the dataset
declare -a TRAIN_PROFILES=(
    "short_common"          # Short, high-frequency words (fastest training)
    "medium_common"         # Medium-length common words
    "long_balanced"         # Longer words with balanced frequency
    "rare_focus"           # Focus on rare/difficult words
    "validation_balanced"   # Balanced mix for comprehensive coverage
    "uniform"              # Uniform sampling across all words
)

# Validation profiles to pair with training profiles
# Mix it up to ensure model generalizes well
declare -a VAL_PROFILES=(
    "validation_balanced"   # Use balanced validation for short_common
    "validation_balanced"   # Standard validation
    "validation_balanced"   # Standard validation
    "validation_balanced"   # Standard validation
    "uniform"              # Different validation for balanced training
    "validation_balanced"   # Standard validation
)

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# Logging
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/multi_profile_training_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Function to get the latest checkpoint from any run
get_latest_checkpoint() {
    find . -name "epoch=*.ckpt" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1
}

# Function to get current epoch from checkpoint filename
get_epoch_from_checkpoint() {
    local checkpoint=$1
    if [[ $checkpoint =~ epoch=([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "0"
    fi
}

# Function to run training with specific profile
run_training_with_profile() {
    local train_profile=$1
    local val_profile=$2
    local max_epochs=$3
    local checkpoint=$4

    log_message "Starting training with profile: train=$train_profile, val=$val_profile"
    log_message "Max epochs for this run: $max_epochs"

    if [ -n "$checkpoint" ]; then
        log_message "Resuming from checkpoint: $checkpoint"
        CHECKPOINT_ARG="--checkpoint $checkpoint"
    else
        log_message "Starting fresh training"
        CHECKPOINT_ARG=""
    fi

    # Run training with timeout (4 hours per profile run)
    timeout 4h uv run python new/train_transducer_personalized.py \
        $CHECKPOINT_ARG \
        --profile "$train_profile" \
        --val-profile "$val_profile" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        2>&1 | tee -a $LOG_FILE

    local exit_code=$?

    if [ $exit_code -eq 124 ]; then
        log_message "Training timeout reached for profile $train_profile"
    elif [ $exit_code -eq 137 ]; then
        log_message "Training killed (OOM) for profile $train_profile, will continue with next profile"
        # Clear cache
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        sleep 10
    elif [ $exit_code -ne 0 ]; then
        log_message "Training failed with exit code $exit_code"
        return $exit_code
    else
        log_message "Training completed successfully for profile $train_profile"
    fi

    return 0
}

# Function to cycle through all profiles
run_profile_cycle() {
    local cycle_num=$1
    log_message "===== Starting training cycle $cycle_num ====="

    for i in ${!TRAIN_PROFILES[@]}; do
        local train_profile="${TRAIN_PROFILES[$i]}"
        local val_profile="${VAL_PROFILES[$i]}"

        # Get latest checkpoint
        local checkpoint=$(get_latest_checkpoint)
        local current_epoch=0
        if [ -n "$checkpoint" ]; then
            current_epoch=$(get_epoch_from_checkpoint "$checkpoint")
        fi

        # Calculate target epoch for this profile run
        local target_epoch=$((current_epoch + EPOCHS_PER_PROFILE))

        log_message "Profile $((i+1))/${#TRAIN_PROFILES[@]}: $train_profile (epochs $current_epoch -> $target_epoch)"

        # Run training with this profile
        run_training_with_profile "$train_profile" "$val_profile" "$target_epoch" "$checkpoint"

        # Check if we should continue
        if [ $? -ne 0 ]; then
            log_message "Critical error in training, stopping"
            return 1
        fi

        # Pause between profiles to let system stabilize
        log_message "Pausing before next profile..."
        sleep 30

        # Optional: Run garbage collection
        if command -v nvidia-smi &> /dev/null; then
            log_message "GPU Memory before cleanup: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
            python -c "import gc; import torch; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
            log_message "GPU Memory after cleanup: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
        fi
    done

    log_message "===== Completed training cycle $cycle_num ====="
    return 0
}

# Function to monitor system resources
monitor_resources() {
    while true; do
        if command -v nvidia-smi &> /dev/null; then
            echo "[$(date '+%H:%M:%S')] GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader)" >> "$LOG_DIR/resource_monitor.log"
        fi
        echo "[$(date '+%H:%M:%S')] RAM: $(free -h | grep Mem | awk '{print $3"/"$2}')" >> "$LOG_DIR/resource_monitor.log"
        sleep 60
    done
}

# Main execution
main() {
    log_message "========================================="
    log_message "Multi-Profile Training Script Started"
    log_message "========================================="
    log_message "Configuration:"
    log_message "  Batch Size: $BATCH_SIZE"
    log_message "  Workers: $NUM_WORKERS"
    log_message "  Epochs per Profile: $EPOCHS_PER_PROFILE"
    log_message "  Total Cycles: $TOTAL_CYCLES"
    log_message "  Training Profiles: ${TRAIN_PROFILES[*]}"
    log_message "========================================="

    # Start resource monitor in background
    monitor_resources &
    MONITOR_PID=$!

    # Run training cycles
    for cycle in $(seq 1 $TOTAL_CYCLES); do
        run_profile_cycle $cycle
        if [ $? -ne 0 ]; then
            log_message "Training stopped due to error"
            break
        fi

        # Longer pause between complete cycles
        log_message "Completed cycle $cycle/$TOTAL_CYCLES. Pausing before next cycle..."
        sleep 60
    done

    # Clean up
    kill $MONITOR_PID 2>/dev/null || true

    log_message "========================================="
    log_message "Multi-Profile Training Complete!"
    log_message "Final checkpoint: $(get_latest_checkpoint)"
    log_message "========================================="
}

# Handle interrupts gracefully
trap 'log_message "Training interrupted by user"; kill $MONITOR_PID 2>/dev/null; exit 1' INT TERM

# Run main function
main