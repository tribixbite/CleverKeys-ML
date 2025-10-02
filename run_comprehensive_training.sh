#!/bin/bash
set -euo pipefail
# Comprehensive multi-profile training for complete word coverage
# Designed to run for days, cycling through different sampling strategies

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Determine Python command to use
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
    echo "Using uv for Python execution"
else
    PYTHON_CMD="python"
    echo "uv not found, using system python"
fi

# Parse command line arguments
DRY_RUN=false
FAST_DEV_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run|-n)
            DRY_RUN=true
            echo "===== DRY RUN MODE - No actual training will occur ====="
            ;;
        --fast-dev-run)
            FAST_DEV_RUN=true
            echo "===== FAST DEV RUN MODE - Single batch testing ====="
            ;;
    esac
done

# ============ Configuration ============
: "${CKS_RUN_BASE:=./9292025script/20251002}"
BATCH_SIZE=400          # Conservative to prevent OOM
NUM_WORKERS=2           # Low worker count
EPOCHS_PER_PROFILE=10   # Train each profile for 10 epochs before switching

# Define training stages with specific goals
declare -A TRAINING_STAGES=(
    ["stage1"]="Foundation: Common words and basic patterns"
    ["stage2"]="Expansion: Rare and challenging words"
    ["stage3"]="Refinement: Balanced and production-ready"
)

# Stage 1: Foundation (build strong base with common patterns)
declare -a STAGE1_PROFILES=(
    "short_common"          # Master 3-5 char high-freq words first
    "medium_balanced"       # Expand to 5-8 char balanced words
    "base_random"          # Uniform sampling for general coverage
)

# Stage 2: Expansion (tackle difficult cases)
declare -a STAGE2_PROFILES=(
    "rare_words"           # Focus on low-frequency words
    "long_words"           # Master 8+ character words
    "very_rare"            # Ultra-rare words and proper nouns
    "high_confusion"       # Words with similar swipe paths
)

# Stage 3: Refinement (polish for production)
declare -a STAGE3_PROFILES=(
    "production_current"    # Current production settings
    "validation_balanced"   # Comprehensive balanced mix
    "medium_balanced"      # Re-visit medium words
    "base_random"          # Final uniform polish
)

# Validation profiles for each stage
declare -A VALIDATION_PROFILES=(
    ["stage1"]="validation_balanced"
    ["stage2"]="validation_current"
    ["stage3"]="validation_balanced"
)

# ============ Setup ============
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

LOG_DIR="$CKS_RUN_BASE/training_logs"
mkdir -p "$CKS_RUN_BASE" "$LOG_DIR"
mkdir -p $LOG_DIR
MAIN_LOG="$LOG_DIR/comprehensive_training_$(date +%Y%m%d_%H%M%S).log"
METRICS_LOG="$LOG_DIR/metrics_$(date +%Y%m%d_%H%M%S).csv"

# Initialize metrics CSV
echo "timestamp,stage,profile,epoch,checkpoint,wer,gpu_mem,ram_used" > $METRICS_LOG

# ============ Helper Functions ============
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $MAIN_LOG
}

get_latest_checkpoint() {
    find "$CKS_RUN_BASE" -path "*/lightning_logs/*/checkpoints/epoch=*.ckpt" -type f 2>/dev/null |
    xargs -r ls -t 2>/dev/null | head -1
}

get_best_checkpoint() {
    # Find checkpoint with lowest WER
    find "$CKS_RUN_BASE" -path "*/lightning_logs/*/checkpoints/epoch=*wer=*.ckpt" -type f 2>/dev/null |
    sed 's/.*wer=val_wer=//' | sed 's/.ckpt//' |
    sort -n | head -1 | xargs -r -I {} find "$CKS_RUN_BASE" -name "*wer=val_wer={}.ckpt" | head -1
}

extract_wer_from_checkpoint() {
    local checkpoint=$1
    if [[ $checkpoint =~ wer=val_wer=([0-9.]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "N/A"
    fi
}

get_epoch_from_checkpoint() {
    local checkpoint=$1
    if [[ $checkpoint =~ epoch=epoch=([0-9]+) ]]; then
        local e="${BASH_REMATCH[1]}"
        e=$((10#$e))
        echo "$e"
    else
        echo "0"
    fi
}

log_metrics() {
    local stage=$1
    local profile=$2
    local checkpoint=$3

    local epoch=$(get_epoch_from_checkpoint "$checkpoint")
    local wer=$(extract_wer_from_checkpoint "$checkpoint")
    local gpu_mem="N/A"
    local ram_used=$(free -m | grep Mem | awk '{print $3}')

    if command -v nvidia-smi &> /dev/null; then
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S'),$stage,$profile,$epoch,$checkpoint,$wer,$gpu_mem,$ram_used" >> $METRICS_LOG
}

cleanup_memory() {
    log_message "Performing memory cleanup..."

    # Python garbage collection
    python3 -c "
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print('Python/CUDA cleanup complete')
" 2>/dev/null || true

    # System cache drop (skip if not permitted)
    sync || true
    if [ -w /proc/sys/vm/drop_caches ]; then
        echo 3 > /proc/sys/vm/drop_caches || true
    fi

    sleep 5
}

# ============ Training Functions ============
run_single_training() {
    local profile=$1
    local val_profile=$2
    local checkpoint=$3
    local target_epochs=$4

    log_message "Training with profile: $profile (validation: $val_profile)"

    # Ignore non-ckpt paths (e.g., accidental filenames)
    if [ -n "$checkpoint" ] && [[ "$checkpoint" != *.ckpt ]]; then
        checkpoint=""
    fi

    local checkpoint_arg=""
    # Don't use checkpoint for FAST_DEV_RUN to avoid max_epochs conflict
    if [ "$FAST_DEV_RUN" = true ]; then
        log_message "FAST_DEV_RUN mode - skipping checkpoint resume"
        checkpoint_arg=""
    elif [ -n "$checkpoint" ] && [ -f "$checkpoint" ] && [[ "$checkpoint" == *.ckpt ]]; then
        checkpoint_arg="--checkpoint $checkpoint"
        log_message "Resuming from: $checkpoint"
    fi

    if [ "$DRY_RUN" = true ]; then
        log_message "DRY RUN: Would execute:"
        log_message "  timeout 6h $PYTHON_CMD new/train_transducer_personalized.py \\"
        log_message "    $checkpoint_arg \\"
        log_message "    --profile $profile \\"
        log_message "    --val-profile $val_profile \\"
        log_message "    --batch-size $BATCH_SIZE \\"
        log_message "    --num-workers $NUM_WORKERS"
        sleep 1  # Small delay to simulate execution
        return 0
    fi

    # If compile is requested and resume should be ignored, drop the checkpoint to allow compile on cold start
    if [ "${ENABLE_COMPILE:-0}" = "1" ] && [ "${IGNORE_RESUME:-0}" = "1" ]; then
        if [ -n "$checkpoint_arg" ]; then
            log_message "ENABLE_COMPILE+IGNORE_RESUME set – starting cold (no ckpt) to enable torch.compile"
        fi
        checkpoint_arg=""
    fi

    # Set environment variable for fast_dev_run
    if [ "$FAST_DEV_RUN" = true ]; then
        export FAST_DEV_RUN=1
        log_message "Running in FAST_DEV_RUN mode (single batch test)"
    fi

    # Run with timeout and memory monitoring
    # Compose command with stability env and max-epochs target (toggle via ENABLE_COMPILE)
    local stability_env="DISABLE_COMPILE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_CUDAGRAPHS=0"
    if [ "${ENABLE_COMPILE:-0}" = "1" ]; then
        stability_env=""
    fi
    local cmd="CKS_RUN_BASE=\"$CKS_RUN_BASE\" $stability_env $PYTHON_CMD new/train_transducer_personalized.py \\
        $checkpoint_arg \\
        --profile \"$profile\" \\
        --val-profile \"$val_profile\" \\
        --batch-size $BATCH_SIZE \\
        --num-workers $NUM_WORKERS \\
        --max-epochs $target_epochs"

    set +e
    timeout 6h bash -lc "$cmd" 2>&1 | tee -a "$MAIN_LOG"
    local exit_code=${PIPESTATUS[0]}
    set -e

    # Handle exit codes
    case $exit_code in
        0)
            log_message "✓ Training completed successfully"
            ;;
        124)
            log_message "⏱ Training timeout reached (expected)"
            ;;
        137)
            log_message "⚠ OOM kill detected - cleaning up"
            cleanup_memory
            ;;
        *)
            log_message "✗ Training failed with code $exit_code"
            return $exit_code
            ;;
    esac

    return 0
}

run_stage() {
    local stage_name=$1
    local stage_description="${TRAINING_STAGES[$stage_name]}"
    local -n profiles=$2  # nameref to array
    local val_profile="${VALIDATION_PROFILES[$stage_name]}"

    log_message "════════════════════════════════════════"
    log_message "Starting $stage_name: $stage_description"
    log_message "Profiles: ${profiles[*]}"
    log_message "════════════════════════════════════════"

    # For FAST_DEV_RUN, only test first profile
    local profiles_to_run=("${profiles[@]}")
    if [ "$FAST_DEV_RUN" = true ]; then
        profiles_to_run=("${profiles[0]}")
        log_message "FAST_DEV_RUN: Testing only first profile: ${profiles[0]}"
    fi

    for profile in "${profiles_to_run[@]}"; do
        # Get current best checkpoint
        local checkpoint=$(get_latest_checkpoint)
        local current_epoch=0

        if [ -n "$checkpoint" ]; then
            current_epoch=$(get_epoch_from_checkpoint "$checkpoint")
            log_metrics "$stage_name" "$profile" "$checkpoint"
        fi

        log_message "───────────────────────────────────"
        local current_epoch_dec=$((10#$current_epoch))
        local target_epoch=$((current_epoch_dec + EPOCHS_PER_PROFILE))
        log_message "Profile: $profile (epochs $current_epoch_dec → $target_epoch)"
        log_message "───────────────────────────────────"

        # Run training
        run_single_training "$profile" "$val_profile" "$checkpoint" $target_epoch

        if [ $? -ne 0 ]; then
            log_message "Critical error - stopping stage"
            return 1
        fi

        # Inter-profile cleanup
        cleanup_memory

        # Show progress
        local new_checkpoint=$(get_latest_checkpoint)
        if [ "$new_checkpoint" != "$checkpoint" ]; then
            local new_wer=$(extract_wer_from_checkpoint "$new_checkpoint")
            log_message "Progress: New checkpoint with WER=$new_wer"
        fi

        sleep 10
    done

    log_message "✓ Completed $stage_name"
    return 0
}

# ============ Monitoring Function ============
monitor_system() {
    if [ "$DRY_RUN" = true ] || [ "$FAST_DEV_RUN" = true ]; then
        log_message "Skipping system monitor (DRY_RUN or FAST_DEV_RUN mode)"
        return 0
    fi

    while true; do
        {
            echo -n "[$(date '+%H:%M:%S')] "

            # GPU stats
            if command -v nvidia-smi &> /dev/null; then
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                    --format=csv,noheader,nounits | tr ',' ' ' | \
                    awk '{printf "GPU: %s%% %s/%sMB %s°C ", $1, $2, $3, $4}'
            fi

            # RAM stats
            free -m | grep Mem | awk '{printf "RAM: %s/%sMB ", $3, $2}'

            # CPU load
            uptime | awk -F'load average:' '{print "Load:" $2}'
        } >> "$LOG_DIR/system_monitor.log"

        sleep 60
    done
}

# ============ Main Execution ============
main() {
    log_message "╔════════════════════════════════════════╗"
    log_message "║  Comprehensive Training System Started  ║"
    log_message "╚════════════════════════════════════════╝"
    log_message ""
    log_message "Configuration:"
    log_message "  Batch Size: $BATCH_SIZE"
    log_message "  Workers: $NUM_WORKERS"
    log_message "  Epochs per Profile: $EPOCHS_PER_PROFILE"
    log_message ""

    # Start system monitor
    monitor_system &
    MONITOR_PID=$!

    # Find and report starting checkpoint
    local start_checkpoint=$(get_latest_checkpoint)
    if [ -n "$start_checkpoint" ]; then
        log_message "Starting from existing checkpoint: $start_checkpoint"
        log_message "  Epoch: $(get_epoch_from_checkpoint "$start_checkpoint")"
        log_message "  WER: $(extract_wer_from_checkpoint "$start_checkpoint")"
    else
        log_message "Starting fresh training (no existing checkpoint)"
    fi

    # Run training stages in cycles
    local cycle=1
    local max_cycles=10  # Run all stages 10 times (can run for days)

    # Limit to 1 cycle for FAST_DEV_RUN
    if [ "$FAST_DEV_RUN" = true ]; then
        max_cycles=1
        log_message "FAST_DEV_RUN: Limited to 1 cycle, first profile only"
    fi

    while [ $cycle -le $max_cycles ]; do
        log_message ""
        log_message "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
        log_message "     Training Cycle $cycle of $max_cycles"
        log_message "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"

        # Run each stage
        run_stage "stage1" STAGE1_PROFILES || break
        run_stage "stage2" STAGE2_PROFILES || break
        run_stage "stage3" STAGE3_PROFILES || break

        # End of cycle summary
        local best_checkpoint=$(get_best_checkpoint)
        log_message ""
        log_message "Cycle $cycle Complete!"
        log_message "Best checkpoint: $best_checkpoint"
        log_message "Best WER: $(extract_wer_from_checkpoint "$best_checkpoint")"

        cycle=$((cycle + 1))

        # Longer pause between cycles
        log_message "Pausing before next cycle..."
        cleanup_memory
        sleep 60
    done

    # Cleanup
    kill $MONITOR_PID 2>/dev/null || true

    # Final report
    log_message ""
    log_message "╔════════════════════════════════════════╗"
    log_message "║   Training Complete - Final Report      ║"
    log_message "╚════════════════════════════════════════╝"

    local final_checkpoint=$(get_latest_checkpoint)
    local best_checkpoint=$(get_best_checkpoint)

    log_message "Final checkpoint: $final_checkpoint"
    log_message "  Epoch: $(get_epoch_from_checkpoint "$final_checkpoint")"
    log_message "  WER: $(extract_wer_from_checkpoint "$final_checkpoint")"
    log_message ""
    log_message "Best checkpoint: $best_checkpoint"
    log_message "  WER: $(extract_wer_from_checkpoint "$best_checkpoint")"
    log_message ""
    log_message "Logs saved to: $LOG_DIR"
    log_message "Metrics CSV: $METRICS_LOG"
}

# Signal handlers
trap 'log_message "Interrupted!"; kill $MONITOR_PID 2>/dev/null; exit 1' INT TERM

# Start training
main
