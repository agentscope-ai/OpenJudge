#!/bin/bash
# OpenJudge GRPO Pointwise Training Script
# Train judge models using GRPO reinforcement learning for scoring
# Support LoRA and full fine-tuning

set -x
TIMESTAMP=$(date "+%m%dT%H%M")

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
SCRIPT_DIR="/grpo/pointwise/"
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
# Environment Setup
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# ============================================================================
# Configuration Summary
# ============================================================================
echo -e "\n=== GRPO Pointwise Training Configuration ==="
echo "RAY_ADDRESS:          $RAY_ADDRESS"
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

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "\nTraining completed successfully!"
    echo "Checkpoints saved to: ${SAVE_PATH}/${EXPERIMENT_NAME}"
else
    echo -e "\nTraining failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
