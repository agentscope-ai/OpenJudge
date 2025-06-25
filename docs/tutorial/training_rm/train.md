# VERL-based Pointwise Reward Model Training Guide

## 📖 Overview

This document provides a comprehensive guide for training Pointwise Reward Models using the VERL framework. Through this tutorial, you will learn how to configure the environment, prepare data, design reward functions, and execute the training pipeline.

> **Note**: This guide covers the general framework and common components. For a complete end-to-end example with HelpSteer2 dataset.

[HelpSteer2 Pointwise Training Guide](pointwise.md).\
[HelpSteer2 Pairwise Training Guide](pairwise.md).

## 🏗️ System Architecture

### Core Components

The VERL pointwise reward model training system consists of three core components:

#### 1. **Training Dataset** - Inherits from `BaseTrainDataset`
   - Supports 0-4 scale helpfulness scoring
   - Provides flexible conversation template system
   - Integrates custom reward functions

#### 2. **Prompt Template** - Based on `BasePromptTemplate`
   - Defines structured output format for scoring
   - Supports extensible scoring criteria
   - Adapts to pointwise evaluation tasks

#### 3. **Reward Function** - Customizable reward computation module
   - Supports exponential decay reward calculation
   - Provides flexible evaluation metric configuration
   - Real-time accuracy and MAE statistics

## 🔧 Environment Configuration

### System Requirements

| Component | Recommended Version |
|-----------|-------------------|
| Python    | ≥ 3.10 |
| CUDA      | ≥ 12.1 |
| PyTorch   | ≥ 2.1 |
| Ray       | ≥ 2.9 |
| VERL      | ≥ 0.4.0 |
| VLLM      | ≥ 0.8.4 |

### Runtime Configuration

Create a `runtime_env.yaml` configuration file:

```yaml
# runtime_env.yaml
excludes: ["/.git/"]
env_vars:
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments: False"
  WANDB_API_KEY: "your_wandb_api_key"
  WANDB_BASE_URL: "your_wandb_base_url"
  HYDRA_FULL_ERROR: "1"
```

### Dependency Installation

Ensure the following essential dependencies are installed:
- `verl==0.4.0` (core framework)
- `ray>=2.9` (distributed computing)
- `vllm>=0.8.4` (inference engine)
- `torch>=2.1` (deep learning framework)

---

## 🚀 Quick Start

### Step 1: Prepare Training Data

Training data should conform to the `DataSample` format specification. For detailed data loading and preprocessing steps, please refer to the data loading section.

### Step 2: Launch Ray Distributed Cluster

#### Single Node Setup
Example for a **single node with 8 × A100**:

```bash
ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --dashboard-host 0.0.0.0
```

#### Multi-Node Setup
**Master Node:**
```bash
ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8
```

**Worker Nodes:**
```bash
ray start --address=$MASTER_ADDR:6379 --num-gpus 8
```

### Step 3: Execute Training Pipeline

```bash
# Navigate to training directory
cd examples/train/pointwise

# Make script executable and run
chmod +x run_pointwise.sh
./run_pointwise.sh
```

### Data Format Description

- **Input Data Format**: All input data must conform to the `DataSample` format

## 🧩 Core Component Details

### Custom Training Dataset

Here's a complete implementation example of a custom training dataset:

```python
class CustomTrainDataset(BaseTrainDataset):
    def __init__(self, *args, **kwargs):
        # Initialize reward module
        self.reward_module = YourRewardModule(
            name="custom_reward",
            template=YourTemplate,
            examples=self._get_examples(),
            llm=None,
        )
        super().__init__(*args, **kwargs)

    def _build_messages(self, example):
        # Build formatted messages
        result = self.reward_module.format(sample=example)
        return [{"role": "user", "content": result}]
```

> **Important Note: Reasoning Model Configuration**
>
> When training reasoning reward models, pay attention to the following configuration:
> - For reasoning models (e.g., Qwen3):
>   - `apply_chat_template` with `enable_thinking=True`
>   - `format` with `enable_thinking=False`
> - For non-reasoning models:
>   - `apply_chat_template` with `enable_thinking=False`
>   - `format` with `enable_thinking=True`
>
> ```python
> # Reasoning model configuration example
> self.tokenizer.apply_chat_template(
>     messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
> )
>
> result = self.helpfulness_reward.format(sample=example, enable_thinking=False)
> ```

### Reward Function Design

The reward function is a key component for evaluating model performance

[HelpSteer2 Pointwise Reward Function Design](pointwise.md#94-reward-function-reward_fnpy).\
[HelpSteer2 Pairwise Reward Function Design](pairwise.md#94-reward-function-reward_fnpy).

### Prompt Template System

The template system defines the structured format for model input and output

[HelpSteer2 Pointwise Template Design](pointwise.md#93-prompt-template-pointwisetraintemplate).\
[HelpSteer2 Pairwise Template Design](pairwise.md#93-prompt-template-pairwisecomparisontemplate).

---

## 📊 Training Monitoring

### Logging and Metrics

The training process logs to both **Console** and **Weights & Biases**:

* **Console**: Use `ray job logs <job_id> -f` for real-time logs
* **WandB**: Set `WANDB_API_KEY` and `WANDB_BASE_URL` environment variables to upload metrics automatically

### Key Metrics to Monitor

| Metric | Meaning | Target Range |
|--------|---------|--------------|
| `reward/mean` | Mean reward of the current epoch | 0.6 - 1.0 |
| `accuracy` | Accuracy of score predictions | > 0.7 |
| `kl_loss` | KL divergence to the reference model | < 0.1 |

### Training Curves

Monitor the training progress through these key curves:

* **Training Reward Curve**: Shows model learning progression on training data
* **Validation Reward Curve**: Indicates generalization performance
* **Loss Curves**: Track convergence of different loss components

---

## ❓ FAQ & Troubleshooting

### Common Issues and Solutions

#### 1. **`num_samples=0` error**

   **Problem**: The dataset is empty after filtering.

   **Solution**: Check whether `_build_messages` parses rows correctly:
   ```python
   from examples.train.pointwise.dataset import PointwiseTrainDataset
   ds = PointwiseTrainDataset(...)
   print(len(ds))
   ```

#### 2. **Ray connection issues**

   **Problem**: Ray can't connect to `127.0.0.1:8265`

   **Solution**:
   - Ensure `ray start --head` has been run
   - Check that port 8265 is reachable
   - Update `--address` parameter in the training script

#### 3. **Out of Memory Errors**

   **Problem**: CUDA out of memory during training

   **Solutions**:
   - Lower `actor_rollout_ref.rollout.gpu_memory_utilization`
   - Reduce `data.train_batch_size` or `ppo_micro_batch_size_per_gpu`
   - Use gradient checkpointing if available

#### 4. **Reasoning Model Configuration Issues**

   **Problem**: Incorrect thinking/reasoning token handling

   **Solution**: For reasoning models (e.g., Qwen3):
   - `apply_chat_template` with `enable_thinking=True`
   - `format` with `enable_thinking=False`

   For non-reasoning models:
   - `apply_chat_template` with `enable_thinking=False`
   - `format` with `enable_thinking=True`

### Performance Optimization Tips

1. **Batch Size Tuning**: Start with smaller batch sizes and gradually increase
2. **Memory Management**: Monitor GPU memory usage with `nvidia-smi`
3. **Ray Resource Allocation**: Ensure proper CPU/GPU resource allocation across Ray workers
4. **Data Loading**: Use efficient data formats (Parquet) and appropriate chunk sizes

---

### PPO + GRPO Pipeline

1. **Data Loading**: Ray workers read the dataset and build prompts + ground truth scores
2. **Generation**: **Actor** uses VLLM to generate score predictions in batches
3. **Reward Calculation**: **RewardManager** calls the custom reward_fn to get scalar rewards
4. **Advantage Estimation**: **GRPO Estimator** computes advantages & targets
5. **Policy Update**: **PPO** updates the actor parameters, while the critic learns the value function
---

## 🔗 Related Resources

### Tutorial Documentation
- **[Data Processing Tutorial](../data/)** - Comprehensive data handling techniques

### Framework Documentation
- **[VERL Framework](https://github.com/volcengine/verl)**: Core training framework
- **[Ray Distributed](https://docs.ray.io/)**: Distributed computing platform
- **[VLLM Inference](https://docs.vllm.ai/)**: High-performance inference engine

### Dataset Resources
- **[HelpSteer2](https://huggingface.co/datasets/nvidia/helpsteer2)**: Human preference dataset
