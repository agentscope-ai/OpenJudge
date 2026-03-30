#!/bin/bash
# OpenJudge GRPO Pointwise Training Script
# Train judge models using GRPO reinforcement learning for scoring
# Support LoRA and full fine-tuning
#
# Usage:
#   ./run.sh                    # Run with default parameters
#   ./run.sh --dry-run          # Validate parameters without running training
#   ./run.sh -h|--help          # Show help message
#
# Environment Variables:
#   All parameters below can be overridden via environment variables.
#   Example: N_GPUS_PER_NODE=4 LR=2e-6 ./run.sh

set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# ============================================================================
# Parse Arguments
# ============================================================================
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help=true
            ;;
    esac
done

# ============================================================================
# Help Message
# ============================================================================
show_help_message() {
    cat << 'EOF'
OpenJudge GRPO Pointwise Training Script
=========================================
Train judge models using GRPO reinforcement learning for scoring.

USAGE:
    ./run.sh [OPTIONS]

OPTIONS:
    --dry-run       Validate parameters without running training
    -h, --help      Show this help message

ENVIRONMENT VARIABLES (with defaults):

  Ray Cluster:
    N_GPUS_PER_NODE      Number of GPUs per node (default: 8)
    N_NODES              Number of nodes (default: 1)

  Paths:
    MODEL_PATH           Model path or HuggingFace ID (default: Qwen/Qwen3-8B)
    TRAIN_FILE           JSON array of training file paths
    VAL_FILE             JSON array of validation file paths
    SAVE_PATH            Checkpoint save directory (default: ./checkpoints/grpo/pointwise)
    SCRIPT_DIR           Script directory (default: /grpo/pointwise/)

  Training:
    PROJECT_NAME         Project name for logging (default: OpenJudge)
    EXPERIMENT_NAME      Experiment name (default: openjudge-grpo-pointwise-<timestamp>)
    TRAIN_BATCH_SIZE     Training batch size (default: 48)
    VAL_BATCH_SIZE       Validation batch size (default: 48)
    MAX_PROMPT_LENGTH    Maximum prompt length (default: 4096)
    MAX_RESPONSE_LENGTH  Maximum response length (default: 2048)
    LR                   Learning rate (default: 1e-6)
    KL_LOSS_COEF         KL loss coefficient (default: 0.001)
    ROLLOUT_N            Number of rollouts (default: 4)
    TOTAL_EPOCHS         Total training epochs (default: 3)
    SAVE_FREQ            Save frequency (default: 2)
    TEST_FREQ            Test frequency (default: 2)
    VAL_BEFORE_TRAIN     Validate before training (default: False)

  Actor/Rollout:
    PPO_MINI_BATCH_SIZE       PPO mini batch size (default: 24)
    GPU_MEMORY_UTILIZATION    GPU memory utilization (default: 0.6)

  LoRA:
    LORA_RANK           LoRA rank, 0 to disable (default: 0)
    LORA_ALPHA          LoRA alpha (default: 32)
    LORA_TARGET_MODULES LoRA target modules (default: all-linear)

  Logging:
    LOGGER              Logger backends (default: ['console','swanlab'])

EXAMPLES:
    # Run with default settings
    ./run.sh

    # Run with custom learning rate and GPU count
    LR=2e-6 N_GPUS_PER_NODE=4 ./run.sh

    # Enable LoRA fine-tuning
    LORA_RANK=64 LORA_ALPHA=32 ./run.sh

    # Validate parameters only
    ./run.sh --dry-run
EOF
    exit 0
}

if [[ "$show_help" == "true" ]]; then
    show_help_message
fi

# ============================================================================
# Ray Cluster Configuration
# ============================================================================
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
N_NODES=${N_NODES:-1}

# ============================================================================
# Path Configuration
# ============================================================================
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
# Data: Multiple jsonl files for different evaluation dimensions
TRAIN_FILE=${TRAIN_FILE:-'["/data/text/correctness/correctness_eval_v1_train.jsonl","/data/text/hallucination/hallucination_eval_v1_train.jsonl","/data/text/relevance/relevance_eval_v1_train.jsonl","/data/text/harmlessness/harmlessness_eval_v1_train.jsonl","/data/text/instruction_following/instruction_following_eval_v1_train.jsonl"]'}
VAL_FILE=${VAL_FILE:-'["/data/text/correctness/correctness_eval_v1_val.jsonl","/data/text/hallucination/hallucination_eval_v1_val.jsonl","/data/text/relevance/relevance_eval_v1_val.jsonl","/data/text/harmlessness/harmlessness_eval_v1_val.jsonl","/data/text/instruction_following/instruction_following_eval_v1_val.jsonl"]'}

SAVE_PATH=${SAVE_PATH:-./checkpoints/grpo/pointwise}

# Get script directory for relative paths
SCRIPT_DIR=${SCRIPT_DIR:-"/grpo/pointwise/"}
GRPO_DIR="$(dirname "$SCRIPT_DIR")"

# Custom modules (relative to script location)
CUSTOM_REWARD_FUNCTION_PATH=${CUSTOM_REWARD_FUNCTION_PATH:-${SCRIPT_DIR}/grader_reward_fn.py}
CUSTOM_CHAT_RL_DATASET_PATH=${CUSTOM_CHAT_RL_DATASET_PATH:-${GRPO_DIR}/grader_rl_dataset.py}

# ============================================================================
# Training Configuration
# ============================================================================
PROJECT_NAME=${PROJECT_NAME:-OpenJudge}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-openjudge-grpo-pointwise-${TIMESTAMP}}

# ============================================================================
# Hyperparameters (all support env var override)
# ============================================================================
# Data settings
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-48}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-48}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

# Optimizer settings
LR=${LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

# GRPO settings
ROLLOUT_N=${ROLLOUT_N:-4}

# Training settings
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-2}
TEST_FREQ=${TEST_FREQ:-2}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# Actor/Rollout settings
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}

# ============================================================================
# LoRA Configuration
# ============================================================================
# Default lora rank is 0. Set to a positive integer to enable LoRA. e.g. 64
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

# ============================================================================
# Logger Configuration
# ============================================================================
LOGGER=${LOGGER:-"['console','swanlab']"}

# ============================================================================
# Dependency Check
# ============================================================================
check_dependencies() {
    local missing_deps=()

    # Check required commands
    for cmd in python3; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    # Check Python packages
    local required_packages=("json" "verl.trainer.main_ppo")
    for pkg in "${required_packages[@]}"; do
        if ! python3 -c "import ${pkg%%.*}" 2>/dev/null; then
            missing_deps+=("python package: ${pkg%%.*}")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "[ERROR] Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
}

# ============================================================================
# GPU Availability Check
# ============================================================================
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        echo "[INFO] Detected $gpu_count GPU(s)"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null | while read line; do
            echo "  GPU $line"
        done

        # Check if enough GPUs available
        local available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
        if [[ $available_gpus -lt $N_GPUS_PER_NODE ]]; then
            echo "[WARN] Requested $N_GPUS_PER_NODE GPUs but only $available_gpus available"
        fi
    else
        echo "[WARN] nvidia-smi not found, skipping GPU check"
    fi
}

# ============================================================================
# Disk Space Check
# ============================================================================
check_disk_space() {
    local save_dir="$SAVE_PATH"
    mkdir -p "$save_dir" 2>/dev/null

    if command -v df &> /dev/null; then
        local available_gb=$(df -BG "$save_dir" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
        if [[ -n "$available_gb" && "$available_gb" -lt 10 ]]; then
            echo "[WARN] Low disk space: ${available_gb}GB available in $save_dir"
        else
            echo "[INFO] Disk space: ${available_gb}GB available in $save_dir"
        fi
    fi
}

# ============================================================================
# Export Configuration to File
# ============================================================================
export_config() {
    local config_file="${SAVE_PATH}/${EXPERIMENT_NAME}/config_${TIMESTAMP}.json"
    cat > "$config_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "experiment_name": "$EXPERIMENT_NAME",
    "model_path": "$MODEL_PATH",
    "train_files": $TRAIN_FILE,
    "val_files": $VAL_FILE,
    "save_path": "$SAVE_PATH",
    "n_gpus_per_node": $N_GPUS_PER_NODE,
    "n_nodes": $N_NODES,
    "train_batch_size": $TRAIN_BATCH_SIZE,
    "val_batch_size": $VAL_BATCH_SIZE,
    "max_prompt_length": $MAX_PROMPT_LENGTH,
    "max_response_length": $MAX_RESPONSE_LENGTH,
    "lr": $LR,
    "kl_loss_coef": $KL_LOSS_COEF,
    "rollout_n": $ROLLOUT_N,
    "total_epochs": $TOTAL_EPOCHS,
    "save_freq": $SAVE_FREQ,
    "test_freq": $TEST_FREQ,
    "val_before_train": "$VAL_BEFORE_TRAIN",
    "ppo_mini_batch_size": $PPO_MINI_BATCH_SIZE,
    "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
    "lora_rank": $LORA_RANK,
    "lora_alpha": $LORA_ALPHA,
    "lora_target_modules": "$LORA_TARGET_MODULES",
    "logger": $LOGGER
}
EOF
    echo "[INFO] Configuration exported to: $config_file"
}

# ============================================================================
# Parameter Validation
# ============================================================================
validate_params() {
    local has_error=0

    # Validate numeric parameters
    if ! [[ "$N_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] N_GPUS_PER_NODE must be a positive integer, got: $N_GPUS_PER_NODE"
        has_error=1
    fi

    if ! [[ "$N_NODES" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] N_NODES must be a positive integer, got: $N_NODES"
        has_error=1
    fi

    if ! [[ "$LORA_RANK" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] LORA_RANK must be a non-negative integer, got: $LORA_RANK"
        has_error=1
    fi

    if ! [[ "$ROLLOUT_N" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ROLLOUT_N must be a positive integer, got: $ROLLOUT_N"
        has_error=1
    fi

    if ! [[ "$TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] TOTAL_EPOCHS must be a positive integer, got: $TOTAL_EPOCHS"
        has_error=1
    fi

    if ! [[ "$PPO_MINI_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] PPO_MINI_BATCH_SIZE must be a positive integer, got: $PPO_MINI_BATCH_SIZE"
        has_error=1
    fi

    if ! [[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] TRAIN_BATCH_SIZE must be a positive integer, got: $TRAIN_BATCH_SIZE"
        has_error=1
    fi

    if ! [[ "$VAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] VAL_BATCH_SIZE must be a positive integer, got: $VAL_BATCH_SIZE"
        has_error=1
    fi

    if ! [[ "$MAX_PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] MAX_PROMPT_LENGTH must be a positive integer, got: $MAX_PROMPT_LENGTH"
        has_error=1
    fi

    if ! [[ "$MAX_RESPONSE_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] MAX_RESPONSE_LENGTH must be a positive integer, got: $MAX_RESPONSE_LENGTH"
        has_error=1
    fi

    if ! [[ "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] SAVE_FREQ must be a positive integer, got: $SAVE_FREQ"
        has_error=1
    fi

    if ! [[ "$TEST_FREQ" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] TEST_FREQ must be a positive integer, got: $TEST_FREQ"
        has_error=1
    fi

    if ! [[ "$LORA_ALPHA" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] LORA_ALPHA must be a positive integer, got: $LORA_ALPHA"
        has_error=1
    fi

    # Validate learning rate (scientific notation or decimal)
    if ! python3 -c "import sys; sys.exit(0 if 0 < float('$LR') < 1 else 1)" 2>/dev/null; then
        echo "[ERROR] LR must be a valid float between 0 and 1, got: $LR"
        has_error=1
    fi

    # Validate KL_LOSS_COEF
    if ! python3 -c "import sys; sys.exit(0 if float('$KL_LOSS_COEF') >= 0 else 1)" 2>/dev/null; then
        echo "[ERROR] KL_LOSS_COEF must be a non-negative float, got: $KL_LOSS_COEF"
        has_error=1
    fi

    # Validate GPU_MEMORY_UTILIZATION
    if ! python3 -c "import sys; sys.exit(0 if 0 < float('$GPU_MEMORY_UTILIZATION') <= 1 else 1)" 2>/dev/null; then
        echo "[ERROR] GPU_MEMORY_UTILIZATION must be between 0 and 1, got: $GPU_MEMORY_UTILIZATION"
        has_error=1
    fi

    # Validate VAL_BEFORE_TRAIN (boolean)
    if ! [[ "$VAL_BEFORE_TRAIN" =~ ^(True|False|true|false|0|1)$ ]]; then
        echo "[ERROR] VAL_BEFORE_TRAIN must be True/False, got: $VAL_BEFORE_TRAIN"
        has_error=1
    fi

    # Validate train/val files (parse JSON array and check each file)
    if ! python3 -c "import json, os; files = json.loads('''$TRAIN_FILE'''); [open(f) for f in files]" 2>/dev/null; then
        echo "[ERROR] TRAIN_FILE contains invalid JSON or non-existent files"
        echo "        TRAIN_FILE: $TRAIN_FILE"
        has_error=1
    fi

    if ! python3 -c "import json, os; files = json.loads('''$VAL_FILE'''); [open(f) for f in files]" 2>/dev/null; then
        echo "[ERROR] VAL_FILE contains invalid JSON or non-existent files"
        echo "        VAL_FILE: $VAL_FILE"
        has_error=1
    fi

    # Validate custom reward function path
    if [[ ! -f "$CUSTOM_REWARD_FUNCTION_PATH" ]]; then
        echo "[ERROR] CUSTOM_REWARD_FUNCTION_PATH file not found: $CUSTOM_REWARD_FUNCTION_PATH"
        has_error=1
    fi

    # Validate custom dataset path
    if [[ ! -f "$CUSTOM_CHAT_RL_DATASET_PATH" ]]; then
        echo "[ERROR] CUSTOM_CHAT_RL_DATASET_PATH file not found: $CUSTOM_CHAT_RL_DATASET_PATH"
        has_error=1
    fi

    if [[ $has_error -eq 1 ]]; then
        echo ""
        echo "[FATAL] Parameter validation failed. Please check the errors above."
        exit 1
    fi

    echo "[INFO] All parameters validated successfully."
}

# ============================================================================
# Environment Setup
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# ============================================================================
# Signal Handling (Graceful Shutdown)
# ============================================================================
cleanup() {
    echo ""
    echo "[INFO] Received interrupt signal. Cleaning up..."
    # Add any cleanup commands here
    exit 130
}
trap cleanup SIGINT SIGTERM

# ============================================================================
# Create Save Directory
# ============================================================================
mkdir -p "${SAVE_PATH}/${EXPERIMENT_NAME}"

# ============================================================================
# Log File Setup
# ============================================================================
LOG_FILE="${SAVE_PATH}/${EXPERIMENT_NAME}/training_${TIMESTAMP}.log"
echo "[INFO] Logging to: $LOG_FILE"

# Function to tee output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Run validation
validate_params

# Check dependencies and system
check_dependencies
check_gpu
check_disk_space

# Export configuration
export_config

# Exit early for dry-run mode
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[INFO] Dry-run mode: Parameters validated successfully. Skipping training."
    exit 0
fi

# ============================================================================
# Start Timer
# ============================================================================
START_TIME=$(date +%s)
echo "[INFO] Training started at: $(date '+%Y-%m-%d %H:%M:%S')"

# ============================================================================
# Configuration Summary
# ============================================================================
echo -e "\n=== GRPO Pointwise Training Configuration ==="
echo "MODEL_PATH:           $MODEL_PATH"
echo "TRAIN_FILE:           $TRAIN_FILE"
echo "VAL_FILE:             $VAL_FILE"
echo "SAVE_PATH:            $SAVE_PATH"
echo "N_GPUS_PER_NODE:      $N_GPUS_PER_NODE"
echo "N_NODES:              $N_NODES"
echo "MAX_PROMPT_LENGTH:    $MAX_PROMPT_LENGTH"
echo "MAX_RESPONSE_LENGTH:  $MAX_RESPONSE_LENGTH"
echo "LR:                   $LR"
echo "KL_LOSS_COEF:         $KL_LOSS_COEF"
echo "ROLLOUT_N:            $ROLLOUT_N"
echo "TOTAL_EPOCHS:         $TOTAL_EPOCHS"
echo "PPO_MINI_BATCH_SIZE:  $PPO_MINI_BATCH_SIZE"
echo "GPU_MEMORY_UTIL:      $GPU_MEMORY_UTILIZATION"
echo ""
echo "=== LoRA Configuration ==="
echo "LORA_RANK:            $LORA_RANK"
echo "LORA_ALPHA:           $LORA_ALPHA"
echo "LORA_TARGET_MODULES:  $LORA_TARGET_MODULES"
echo ""
echo "=== Validation & Logging ==="
echo "VAL_BEFORE_TRAIN:     $VAL_BEFORE_TRAIN"
echo "SAVE_FREQ:            $SAVE_FREQ"
echo "TEST_FREQ:            $TEST_FREQ"
echo "LOGGER:               $LOGGER"
echo "==============================================\n"

# Change to GRPO directory for correct relative path resolution
cd "${GRPO_DIR}" || { echo "Failed to cd to $GRPO_DIR"; exit 1; }

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.prompt_key='input' \
    data.custom_cls.path="${CUSTOM_CHAT_RL_DATASET_PATH}" \
    data.custom_cls.name="PointwiseChatRLDataset" \
    reward_model.reward_manager='naive' \
    custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
    custom_reward_function.name='compute_score' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=$LOGGER \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.default_local_dir="${SAVE_PATH}/${EXPERIMENT_NAME}"

EXIT_CODE=$?

# ============================================================================
# End Timer
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "\n=============================================="
    echo "[SUCCESS] Training completed successfully!"
    echo "Checkpoints saved to: ${SAVE_PATH}/${EXPERIMENT_NAME}"
    echo "Log file: $LOG_FILE"
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "=============================================="
else
    echo -e "\n=============================================="
    echo "[FAILED] Training failed with exit code $EXIT_CODE"
    echo "Log file: $LOG_FILE"
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "=============================================="
fi

exit $EXIT_CODE
