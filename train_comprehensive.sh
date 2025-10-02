#!/bin/bash
# Comprehensive frequency-aware training for CleverKeys RNNT model
# Addresses the massive frequency imbalance where common words dominate training
# Fully resumable after system restarts - just run the script again!

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
MAX_EPOCHS_PER_RUN=100 # Epochs per profile before switching (increased for longer runs)
TOTAL_EPOCHS_TARGET=500 # Total epochs to aim for across all profiles

# Checkpoint management (all artifacts under 9292025script)
BASE_DIR="./9292025script/20251002"
CHECKPOINT_BASE_DIR="$BASE_DIR"
LOG_DIR="$BASE_DIR/training_logs"
STATE_FILE="$BASE_DIR/training_state.json" # Persistent state for resumption

# ============ Helper Functions ============

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save training state for resumption after restart
save_state() {
    local current_strategy="$1"
    local current_profile="$2"
    local current_checkpoint="$3"
    local current_epoch="$4"

    cat > "$STATE_FILE" <<EOF
{
    "strategy": "$current_strategy",
    "profile": "$current_profile",
    "checkpoint": "$current_checkpoint",
    "epoch": $current_epoch,
    "timestamp": "$(date -Iseconds)",
    "batch_size": $BATCH_SIZE
}
EOF
    log_message "State saved to $STATE_FILE"
}

# Load training state if exists
load_state() {
    if [ -f "$STATE_FILE" ]; then
        log_message "Found previous training state in $STATE_FILE"
        # Parse JSON manually (basic parsing for portability)
        RESUME_STRATEGY=$(grep '"strategy"' "$STATE_FILE" | sed 's/.*: *"\([^"]*\)".*/\1/')
        RESUME_PROFILE=$(grep '"profile"' "$STATE_FILE" | sed 's/.*: *"\([^"]*\)".*/\1/')
        RESUME_CHECKPOINT=$(grep '"checkpoint"' "$STATE_FILE" | sed 's/.*: *"\([^"]*\)".*/\1/')
        RESUME_EPOCH=$(grep '"epoch"' "$STATE_FILE" | sed 's/.*: *\([0-9]*\).*/\1/')
        return 0
    else
        return 1
    fi
}

detect_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        local mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ -n "$mem_mb" ]; then
            local mem_gb=$((mem_mb / 1024))
            # Print detection info to stderr to avoid contaminating command substitution
            echo "Detected GPU memory: ${mem_gb}GB" >&2

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
        echo "No GPU detected, using CPU mode" >&2
        echo 32  # Small batch for CPU
    fi
}

find_best_checkpoint() {
    local pattern="${1:-*}"
    # Prefer .ckpt files by best WER if present
    local best_ckpt
    best_ckpt=$(find "$BASE_DIR" -path "$BASE_DIR/rnnt_checkpoints_*/checkpoints/*.ckpt" -type f 2>/dev/null | \
    while read -r ckpt; do
        if [[ $ckpt =~ wer=([0-9.]+) ]]; then
            echo "${BASH_REMATCH[1]} $ckpt"
        fi
    done | sort -n | head -1 | awk '{print $2}')
    if [ -n "$best_ckpt" ]; then
        echo "$best_ckpt"
        return
    fi
    # Fallback to latest .nemo (for reporting only, not resume)
    local best_nemo
    best_nemo=$(find "$BASE_DIR" -path "$BASE_DIR/rnnt_checkpoints_*/model_epoch*.nemo" -type f 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [ -n "$best_nemo" ] && echo "$best_nemo"
}

find_latest_checkpoint() {
    local pattern="${1:-*}"
    # Prefer latest .ckpt
    local latest_ckpt
    latest_ckpt=$(find "$BASE_DIR" -path "$BASE_DIR/rnnt_checkpoints_${pattern}/checkpoints/*.ckpt" -type f 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    if [ -n "$latest_ckpt" ]; then
        echo "$latest_ckpt"
        return
    fi
    # Fallback to latest .nemo
    local latest_nemo
    latest_nemo=$(find "$BASE_DIR" -path "$BASE_DIR/rnnt_checkpoints_*/model_epoch*.nemo" -type f 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [ -n "$latest_nemo" ] && echo "$latest_nemo"
}

get_checkpoint_info() {
    local ckpt="$1"
    local epoch=""
    local wer=""

    if [[ $ckpt =~ epoch=([0-9]+) ]]; then
        epoch="${BASH_REMATCH[1]}"
    fi
    if [[ $ckpt =~ val_wer=([0-9.]+) ]]; then
        wer="${BASH_REMATCH[1]}"
    elif [[ $ckpt =~ wer=([0-9.]+) ]]; then
        wer="${BASH_REMATCH[1]}"
    fi

    echo "epoch=$epoch wer=$wer"
}

# Extract epoch number from a checkpoint or nemo filename
get_epoch_from_checkpoint() {
    local ckpt="$1"
    local epoch="0"
    if [[ $ckpt =~ epoch=([0-9]+) ]]; then
        epoch="${BASH_REMATCH[1]}"
    elif [[ $ckpt =~ model_epoch([0-9]+)\.nemo$ ]]; then
        epoch="${BASH_REMATCH[1]}"
    fi
    echo "$epoch"
}

run_training_with_profile() {
    local profile="$1"
    local val_profile="${2:-validation_balanced}"
    local checkpoint="${3:-}"
    local max_epochs="${4:-$MAX_EPOCHS_PER_RUN}"
    local strategy="${5:-unknown}"

    log_message "════════════════════════════════════════"
    log_message "Training with profile: $profile"
    log_message "Validation profile: $val_profile"

    local current_epoch=0
    if [ -n "$checkpoint" ] && [ -f "$checkpoint" ]; then
        log_message "Resuming from: $checkpoint"
        local info=$(get_checkpoint_info "$checkpoint")
        log_message "Checkpoint info: $info"
        current_epoch=$(get_epoch_from_checkpoint "$checkpoint")
    else
        log_message "Starting fresh training (no checkpoint)"
    fi

    # Calculate target epochs (current + requested)
    local target_epochs=$((current_epoch + max_epochs))
    log_message "Training from epoch $current_epoch to $target_epochs"
    log_message "════════════════════════════════════════"

    # Save state before training
    save_state "$strategy" "$profile" "$checkpoint" "$current_epoch"

    # Build command (export base dir for python run)
    local cmd="CKS_RUN_BASE=\"$BASE_DIR\" DISABLE_COMPILE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_CUDAGRAPHS=0 uv run python $TRAIN_SCRIPT"
    cmd="$cmd --profile $profile"
    cmd="$cmd --val-profile $val_profile"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --num-workers $NUM_WORKERS"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --max-epochs $target_epochs"  # Set explicit epoch target

    if [ -n "$checkpoint" ] && [ -f "$checkpoint" ] && [[ "$checkpoint" == *.ckpt ]]; then
        cmd="$cmd --checkpoint $checkpoint"  # only pass .ckpt to trainer
    fi

    log_message "Command: $cmd"

    # Run training
    if bash -lc "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
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

        # Update saved state with new checkpoint
        local new_epoch=$(get_epoch_from_checkpoint "$new_ckpt")
        save_state "$strategy" "$profile" "$new_ckpt" "$new_epoch"

        echo "$new_ckpt"  # Return new checkpoint path
    else
        echo "$checkpoint"  # Return original if no new one
    fi
}

# ============ Training Strategies ============

# Strategy 1: Curriculum Learning
# Gradually progress from common to rare words
run_curriculum_training() {
    local checkpoint_init="${1:-}"
    log_message "╔════════════════════════════════════════╗"
    log_message "║    CURRICULUM LEARNING STRATEGY        ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint="$checkpoint_init"
    local stages=("curriculum_stage1" "curriculum_stage2" "curriculum_stage3" "curriculum_stage4")

    # If resuming, skip stages until we reach the recorded profile
    local start_idx=0
    if [ -n "${RESUME_PROFILE:-}" ]; then
        local i=0
        for st in "${stages[@]}"; do
            if [ "$st" = "$RESUME_PROFILE" ]; then
                start_idx=$i
                break
            fi
            i=$((i+1))
        done
    fi

    local idx=$start_idx
    while [ $idx -lt ${#stages[@]} ]; do
        local stage=${stages[$idx]}
        checkpoint=$(run_training_with_profile "$stage" "validation_balanced" "$checkpoint" 100 "curriculum")
        sleep 10  # Brief pause between stages
        idx=$((idx+1))
    done

    log_message "Curriculum training complete!"
    return 0
}

# Strategy 2: Frequency Band Training
# Train separate "experts" for different frequency bands
run_frequency_band_training() {
    local checkpoint_init="${1:-}"
    log_message "╔════════════════════════════════════════╗"
    log_message "║    FREQUENCY BAND TRAINING             ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint="$checkpoint_init"
    local bands=(
        "ultra_common_suppressed:30"  # Fewer epochs for common words
        "common_balanced:40"
        "medium_frequency:50"
        "rare_focused:60"              # More epochs for rare words
        "ultra_rare_boost:70"
    )

    # Skip until RESUME_PROFILE if present
    local pass=false
    [ -z "${RESUME_PROFILE:-}" ] && pass=true

    for band_spec in "${bands[@]}"; do
        IFS=':' read -r band epochs <<< "$band_spec"
        if [ "$pass" = false ]; then
            if [ "$band" = "$RESUME_PROFILE" ]; then
                pass=true
            else
                continue
            fi
        fi
        checkpoint=$(run_training_with_profile "$band" "validation_balanced" "$checkpoint" "$epochs")
        sleep 10
    done

    log_message "Frequency band training complete!"
    return 0
}

# Strategy 3: Length-Based Training
# Focus on different word lengths
run_length_based_training() {
    local checkpoint_init="${1:-}"
    log_message "╔════════════════════════════════════════╗"
    log_message "║    LENGTH-BASED TRAINING               ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint="$checkpoint_init"
    local lengths=("short_words" "medium_words" "long_words")

    local pass=false
    [ -z "${RESUME_PROFILE:-}" ] && pass=true
    for length_profile in "${lengths[@]}"; do
        if [ "$pass" = false ]; then
            if [ "$length_profile" = "$RESUME_PROFILE" ]; then
                pass=true
            else
                continue
            fi
        fi
        checkpoint=$(run_training_with_profile "$length_profile" "validation_balanced" "$checkpoint" 40)
        sleep 10
    done

    log_message "Length-based training complete!"
    return 0
}

# Strategy 4: Cyclic Training
# Cycle through different profiles to prevent overfitting
run_cyclic_training() {
    local checkpoint_init="${1:-}"
    log_message "╔════════════════════════════════════════╗"
    log_message "║    CYCLIC TRAINING                     ║"
    log_message "╚════════════════════════════════════════╝"

    local checkpoint="$checkpoint_init"
    local cycles=3
    local profiles=(
        "sqrt_balanced"
        "rare_focused"
        "production_balanced"
    )

    local resumed=false
    for ((cycle=1; cycle<=cycles; cycle++)); do
        log_message "━━━━━━━ Cycle $cycle of $cycles ━━━━━━━"

        for profile in "${profiles[@]}"; do
            if [ "$resumed" = false ] && [ -n "${RESUME_PROFILE:-}" ] && [ "$profile" != "$RESUME_PROFILE" ]; then
                continue
            fi
            resumed=true
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
    mkdir -p "$BASE_DIR"
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
    if [ -n "$start_checkpoint" ] && [ -f "$start_checkpoint" ]; then
        log_message "Found existing checkpoint: $start_checkpoint"
        local info=$(get_checkpoint_info "$start_checkpoint")
        log_message "Checkpoint info: $info"
    else
        log_message "No existing checkpoint found, starting fresh"
    fi

    # Determine CLI strategy (if provided)
    local cli_strategy="${1:-}"

    # Check for resume state first
    if load_state; then
        log_message "╔════════════════════════════════════════╗"
        log_message "║   RESUMING INTERRUPTED TRAINING        ║"
        log_message "╚════════════════════════════════════════╝"
        log_message "Previous strategy: $RESUME_STRATEGY"
        log_message "Previous profile: $RESUME_PROFILE"
        log_message "Previous checkpoint: $RESUME_CHECKPOINT"
        log_message "Previous epoch: $RESUME_EPOCH"

        # Strategy selection: prefer CLI if provided, else resume strategy
        local strategy
        if [ -n "$cli_strategy" ]; then
            strategy="$cli_strategy"
        else
            strategy="$RESUME_STRATEGY"
        fi
        local resume_checkpoint="$RESUME_CHECKPOINT"
        if [ -n "$resume_checkpoint" ]; then
            start_checkpoint="$resume_checkpoint"
        fi
    else
        # Parse command-line arguments for fresh start
        local strategy="${cli_strategy:-curriculum}"
        local resume_checkpoint=""
    fi

    case "$strategy" in
        curriculum)
            run_curriculum_training "$start_checkpoint"
            ;;
        frequency)
            run_frequency_band_training "$start_checkpoint"
            ;;
        length)
            run_length_based_training "$start_checkpoint"
            ;;
        cyclic)
            run_cyclic_training "$start_checkpoint"
            ;;
        all)
            # Run all strategies in sequence
            run_curriculum_training "$start_checkpoint"
            start_checkpoint=$(find_latest_checkpoint "curriculum*")
            run_frequency_band_training "$start_checkpoint"
            start_checkpoint=$(find_latest_checkpoint "common_*")
            run_length_based_training "$start_checkpoint"
            start_checkpoint=$(find_latest_checkpoint "short_*")
            run_cyclic_training "$start_checkpoint"
            ;;
        test)
            # Quick test with one profile
            run_training_with_profile "sqrt_balanced" "validation_balanced" "" 2 "test"
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
    cat << 'EOF'
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
