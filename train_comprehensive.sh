#!/bin/bash
# Comprehensive frequency-aware training for CleverKeys RNNT model
# Addresses the massive frequency imbalance where common words dominate training

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============ Configuration ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Training script path (corrected from old script)
TRAIN_SCRIPT="new/train_transducer_personalized.py"

# Batch sizes for different GPU memory configurations
BATCH_SIZE_16GB=400    # RTX 4090M with 16GB
BATCH_SIZE_24GB=600    # RTX 4090 with 24GB
BATCH_SIZE_DEFAULT=320 # Conservative default

# Training parameters
NUM_WORKERS=8          # DataLoader workers
LEARNING_RATE=2e-4     # Default learning rate
MAX_EPOCHS_PER_RUN=50  # Epochs per profile before switching

# Checkpoint management
CHECKPOINT_BASE_DIR="./rnnt_checkpoints"
LOG_DIR="./training_logs"

# ============ Helper Functions ============

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

detect_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        local mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ -n "$mem_mb" ]; then
            local mem_gb=$((mem_mb / 1024))
            log_message "Detected GPU memory: ${mem_gb}GB"

            if [ $mem_gb -ge 24 ]; then
                echo $BATCH_SIZE_24GB
            elif [ $mem_gb -ge 16 ]; then
                echo $BATCH_SIZE_16GB
            else
                echo $BATCH_SIZE_DEFAULT
            fi
        else
            echo $BATCH_SIZE_DEFAULT
        fi
    else
        log_message "No GPU detected, using CPU mode"
        echo 32  # Small batch for CPU
    fi
}

find_best_checkpoint() {
    local pattern="${1:-*}"
    find "$CHECKPOINT_BASE_DIR" -path "*/${pattern}/checkpoints/*.ckpt" -type f 2>/dev/null | \
    while read -r ckpt; do
        if [[ $ckpt =~ wer=([0-9.]+) ]]; then
            echo "${BASH_REMATCH[1]} $ckpt"
        fi
    done | sort -n | head -1 | awk '{print $2}'
}

find_latest_checkpoint() {
    local pattern="${1:-*}"
    find "$CHECKPOINT_BASE_DIR" -path "*/${pattern}/checkpoints/*.ckpt" -type f 2>/dev/null | \
    xargs ls -t 2>/dev/null | head -1
}

get_checkpoint_info() {
    local ckpt="$1"
    local epoch=""
    local wer=""

    if [[ $ckpt =~ epoch=([0-9]+) ]]; then
        epoch="${BASH_REMATCH[1]}"
    fi
    if [[ $ckpt =~ wer=([0-9.]+) ]]; then
        wer="${BASH_REMATCH[1]}"
    fi

    echo "epoch=$epoch wer=$wer"
}

run_training_with_profile() {
    local profile="$1"
    local val_profile="${2:-validation_balanced}"
    local checkpoint="${3:-}"
    local max_epochs="${4:-$MAX_EPOCHS_PER_RUN}"

    log_message "════════════════════════════════════════"
    log_message "Training with profile: $profile"
    log_message "Validation profile: $val_profile"
    log_message "Max epochs: $max_epochs"

    if [ -n "$checkpoint" ] && [ -f "$checkpoint" ]; then
        log_message "Resuming from: $checkpoint"
        local info=$(get_checkpoint_info "$checkpoint")
        log_message "Checkpoint info: $info"
    else
        log_message "Starting fresh training (no checkpoint)"
    fi
    log_message "════════════════════════════════════════"

    # Build command
    local cmd="uv run python $TRAIN_SCRIPT"
    cmd="$cmd --profile $profile"
    cmd="$cmd --val-profile $val_profile"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --num-workers $NUM_WORKERS"
    cmd="$cmd --learning-rate $LEARNING_RATE"

    if [ -n "$checkpoint" ] && [ -f "$checkpoint" ]; then
        cmd="$cmd --checkpoint $checkpoint"
    fi

    # Add max epochs control (needs to be implemented in training script)
    # cmd="$cmd --max-epochs $max_epochs"

    log_message "Command: $cmd"

    # Run training
    if $cmd 2>&1 | tee -a "$LOG_FILE"; then
        log_message "✓ Training completed successfully"
    else
        local exit_code=$?
        log_message "✗ Training failed with exit code $exit_code"
        return $exit_code
    fi

    # Find and report new checkpoint
    local new_ckpt=$(find_latest_checkpoint "$profile*")
    if [ -n "$new_ckpt" ] && [ "$new_ckpt" != "$checkpoint" ]; then
        local info=$(get_checkpoint_info "$new_ckpt")
        log_message "New checkpoint created: $info"
        echo "$new_ckpt"  # Return new checkpoint path
    else
        echo "$checkpoint"  # Return original if no new one
    fi
}

# ============ Training Strategies ============

# Strategy 1: Curriculum Learning
# Gradually progress from common to rare words
run_curriculum_training() {
    log_message "╔════════════════════════════════════════╗"
    log_message "║    CURRICULUM LEARNING STRATEGY        ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint=""
    local stages=("curriculum_stage1" "curriculum_stage2" "curriculum_stage3" "curriculum_stage4")

    for stage in "${stages[@]}"; do
        checkpoint=$(run_training_with_profile "$stage" "validation_balanced" "$checkpoint" 50)
        sleep 10  # Brief pause between stages
    done

    log_message "Curriculum training complete!"
    return 0
}

# Strategy 2: Frequency Band Training
# Train separate "experts" for different frequency bands
run_frequency_band_training() {
    log_message "╔════════════════════════════════════════╗"
    log_message "║    FREQUENCY BAND TRAINING             ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint=""
    local bands=(
        "ultra_common_suppressed:30"  # Fewer epochs for common words
        "common_balanced:40"
        "medium_frequency:50"
        "rare_focused:60"              # More epochs for rare words
        "ultra_rare_boost:70"
    )

    for band_spec in "${bands[@]}"; do
        IFS=':' read -r band epochs <<< "$band_spec"
        checkpoint=$(run_training_with_profile "$band" "validation_balanced" "$checkpoint" "$epochs")
        sleep 10
    done

    log_message "Frequency band training complete!"
    return 0
}

# Strategy 3: Length-Based Training
# Focus on different word lengths
run_length_based_training() {
    log_message "╔════════════════════════════════════════╗"
    log_message "║    LENGTH-BASED TRAINING               ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint=""
    local lengths=("short_words" "medium_words" "long_words")

    for length_profile in "${lengths[@]}"; do
        checkpoint=$(run_training_with_profile "$length_profile" "validation_balanced" "$checkpoint" 40)
        sleep 10
    done

    log_message "Length-based training complete!"
    return 0
}

# Strategy 4: Cyclic Training
# Cycle through different profiles to prevent overfitting
run_cyclic_training() {
    log_message "╔════════════════════════════════════════╗"
    log_message "║    CYCLIC TRAINING                     ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint=""
    local cycles=3
    local profiles=(
        "sqrt_balanced"
        "rare_focused"
        "production_balanced"
    )

    for ((cycle=1; cycle<=cycles; cycle++)); do
        log_message "━━━━━━━ Cycle $cycle of $cycles ━━━━━━━"

        for profile in "${profiles[@]}"; do
            checkpoint=$(run_training_with_profile "$profile" "validation_balanced" "$checkpoint" 30)
            sleep 10
        done
    done

    log_message "Cyclic training complete!"
    return 0
}

# ============ Main Execution ============

main() {
    # Setup
    mkdir -p "$LOG_DIR"
    mkdir -p "$CHECKPOINT_BASE_DIR"

    # Create timestamped log file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/training_${timestamp}.log"

    log_message "╔════════════════════════════════════════╗"
    log_message "║  CleverKeys Comprehensive Training     ║"
    log_message "╚════════════════════════════════════════╝"

    # Detect and set batch size based on GPU memory
    BATCH_SIZE=$(detect_gpu_memory)
    log_message "Using batch size: $BATCH_SIZE"

    # Check for existing checkpoint
    local start_checkpoint=$(find_best_checkpoint)
    if [ -n "$start_checkpoint" ]; then
        log_message "Found existing checkpoint: $start_checkpoint"
        local info=$(get_checkpoint_info "$start_checkpoint")
        log_message "Checkpoint info: $info"
    else
        log_message "No existing checkpoint found, starting fresh"
    fi

    # Parse command-line arguments
    local strategy="${1:-curriculum}"

    case "$strategy" in
        curriculum)
            run_curriculum_training
            ;;
        frequency)
            run_frequency_band_training
            ;;
        length)
            run_length_based_training
            ;;
        cyclic)
            run_cyclic_training
            ;;
        all)
            # Run all strategies in sequence
            run_curriculum_training
            run_frequency_band_training
            run_length_based_training
            run_cyclic_training
            ;;
        test)
            # Quick test with one profile
            run_training_with_profile "sqrt_balanced" "validation_balanced" "" 2
            ;;
        *)
            log_message "Unknown strategy: $strategy"
            log_message "Available strategies: curriculum, frequency, length, cyclic, all, test"
            exit 1
            ;;
    esac

    # Final summary
    log_message ""
    log_message "╔════════════════════════════════════════╗"
    log_message "║    TRAINING COMPLETE - SUMMARY         ║"
    log_message "╚════════════════════════════════════════╝"

    local final_checkpoint=$(find_best_checkpoint)
    if [ -n "$final_checkpoint" ]; then
        local info=$(get_checkpoint_info "$final_checkpoint")
        log_message "Best checkpoint: $final_checkpoint"
        log_message "Performance: $info"
    fi

    log_message "Log saved to: $LOG_FILE"
    log_message "Checkpoints in: $CHECKPOINT_BASE_DIR"
}

# Signal handlers for graceful shutdown
trap 'log_message "Training interrupted!"; exit 130' INT TERM

# Show usage if --help is provided
if [[ "${1:-}" == "--help" ]]; then
    cat << EOF
Usage: $0 [strategy]

Comprehensive training script for CleverKeys RNNT model with frequency-aware sampling.

Strategies:
  curriculum  - Gradual progression from common to rare words (default)
  frequency   - Train separate experts for different frequency bands
  length      - Focus on different word lengths
  cyclic      - Cycle through profiles to prevent overfitting
  all         - Run all strategies in sequence
  test        - Quick test run with minimal epochs

The script automatically detects GPU memory and adjusts batch size accordingly.

Examples:
  $0                  # Run default curriculum strategy
  $0 curriculum       # Explicitly run curriculum learning
  $0 frequency        # Run frequency band training
  $0 all              # Run all strategies
  $0 test             # Quick test run

Environment variables:
  BATCH_SIZE_OVERRIDE - Override auto-detected batch size
  NUM_WORKERS         - Number of DataLoader workers (default: 8)
  LEARNING_RATE       - Learning rate (default: 2e-4)

EOF
    exit 0
fi

# Check for required files
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    echo "Please ensure you're running from the CleverKeys root directory"
    exit 1
fi

if [ ! -f "new/sampling_profiles.py" ]; then
    echo "Error: Sampling profiles not found at new/sampling_profiles.py"
    exit 1
fi

# Override batch size if environment variable is set
if [ -n "${BATCH_SIZE_OVERRIDE:-}" ]; then
    BATCH_SIZE=$BATCH_SIZE_OVERRIDE
    echo "Using overridden batch size: $BATCH_SIZE"
fi

# Start training
main "$@"