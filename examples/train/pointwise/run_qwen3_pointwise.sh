# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

TIMESTAMP=$(date "+%m%dT%H%M")

TRAIN_FILE=./data/loaded_dataset_helpsteer2_output_train.parquet
VAL_FILE=./data/loaded_dataset_helpsteer2_output_test.parquet
MODEL_PATH=./models/Qwen3-14B

PROJECT_NAME=warmup_train
EXPERIMENT_NAME=rm-gallery-qwen3-${TIMESTAMP}

CUSTOM_REWARD_FUNCTION_PATH=./rm_gallery/examples/train/pointwise/reward_fn.py
CUSTOM_CHAT_RL_DATASET_PATH=./rm_gallery/examples/train/pointwise/dataset_helpsteer2.py
CUSTOM_CHAT_RL_DATASET_NAME=HelpSteer2TrainDataset
REWARD_MANAGER=naive
REWARD_FUNCTION_NAME=compute_score


DEFAULT_LOCAL_DIR=./checkpoints/${TIMESTAMP}

N_GPUS_PER_NODE=8
N_NODES=3

set -x

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=./rm_gallery/examples/train/pointwise/runtime_env.yaml \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=96 \
    data.val_batch_size=192 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.prompt_key='input' \
    data.custom_cls.path="${CUSTOM_CHAT_RL_DATASET_PATH}" \
    data.custom_cls.name="${CUSTOM_CHAT_RL_DATASET_NAME}" \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${REWARD_FUNCTION_NAME} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}