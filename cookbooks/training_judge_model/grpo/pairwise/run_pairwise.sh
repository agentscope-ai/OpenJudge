#!/bin/bash
# OpenJudge GRPO Pairwise Training Script
# Train judge models using GRPO reinforcement learning for preference comparison (A/B)

set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# ============================================================================
# Ray Cluster Configuration
# ============================================================================
RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
N_NODES=${N_NODES:-1}

# ============================================================================
# Path Configuration
# ============================================================================
# Model: Use HuggingFace model ID or local path
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}

# Data: Download from HuggingFace or use local parquet files
TRAIN_FILE=${TRAIN_FILE:-./data/rewardbench2_pairwise_train.parquet}
VAL_FILE=${VAL_FILE:-./data/rewardbench2_pairwise_val.parquet}

# Output directory
SAVE_PATH=${SAVE_PATH:-./checkpoints/grpo/pairwise}

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_DIR="$(dirname "$SCRIPT_DIR")"

# Custom modules
CUSTOM_REWARD_FUNCTION_PATH=${SCRIPT_DIR}/reward_fn.py
CUSTOM_CHAT_RL_DATASET_PATH=${GRPO_DIR}/chat_rl_dataset.py
RUNTIME_ENV_PATH=${GRPO_DIR}/runtime_env.yaml

# ============================================================================
# Training Configuration
# ============================================================================
PROJECT_NAME=OpenJudge
EXPERIMENT_NAME=grpo-pairwise-${TIMESTAMP}

# ============================================================================
# Hyperparameters
# ============================================================================
# Data settings
TRAIN_BATCH_SIZE=96
VAL_BATCH_SIZE=192
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=2048

# Optimizer settings
LR=1e-6
KL_LOSS_COEF=0.001

# GRPO settings
ROLLOUT_N=4  # Number of samples per prompt

# Training settings
TOTAL_EPOCHS=30
SAVE_FREQ=20
TEST_FREQ=4

# ============================================================================
# Environment Setup
# ============================================================================
# Disable PyTorch expandable segments for vLLM compatibility
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

echo "=== GRPO Pairwise Training Configuration ==="
echo "RAY_ADDRESS: $RAY_ADDRESS"
echo "MODEL_PATH: $MODEL_PATH"
echo "TRAIN_FILE: $TRAIN_FILE"
echo "N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
echo "N_NODES: $N_NODES"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "GRPO_DIR: $GRPO_DIR"
echo "=============================================="

# Change to GRPO directory to ensure runtime_env.yaml working_dir resolves correctly
cd "${GRPO_DIR}"

ray job submit --address="${RAY_ADDRESS}" \
    --runtime-env="${RUNTIME_ENV_PATH}" \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.prompt_key='input' \
    data.custom_cls.path="${CUSTOM_CHAT_RL_DATASET_PATH}" \
    data.custom_cls.name="PairwiseChatRLDataset" \
    reward_model.reward_manager='naive' \
    custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
    custom_reward_function.name='compute_score' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=${SAVE_PATH}/${EXPERIMENT_NAME}

echo "Training completed! Checkpoints saved to: ${SAVE_PATH}/${EXPERIMENT_NAME}"