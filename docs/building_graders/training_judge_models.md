# Train Judge Models

Train judge models using three approaches: **SFT** for foundation learning, **Bradley-Terry** for scalar preference scoring, and **GRPO** for generative evaluation with reasoning.

!!! info "Terminology: Judge Model vs Reward Model"
    In OpenJudge, we use **judge model** to refer to models trained for evaluation. This is the same concept as **reward model** commonly used in RLHF literature. Both terms describe models that assess and score AI outputs—we prefer "judge model" to emphasize the evaluation and assessment role.


## Overview

OpenJudge provides training pipelines for building custom judge models. Each method serves different use cases:

| Method | Output Type | Training Data | Interpretable | Best For |
|--------|-------------|---------------|---------------|----------|
| **SFT** | Generative (text) | Demonstrations | ✅ Yes | Model initialization, response generation |
| **Bradley-Terry** | Scalar score | Preference pairs | ❌ No | RLHF judge modeling, ranking |
| **GRPO** | Generative (text) | Labeled responses | ✅ Yes | Interpretable evaluation with reasoning |

**Common Requirements:**

```bash
pip install verl==0.6.1
```


## Datasets

All training datasets are available on HuggingFace:

| Method | Dataset | Link |
|--------|---------|------|
| SFT | HelpSteer2 high-quality responses | [🔗 train_rm/sft](https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main/train_rm/sft) |
| Bradley-Terry | HelpSteer2 preference pairs | [🔗 train_rm/bradley_terry](https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main/train_rm/bradley_terry) |
| GRPO Pointwise | RewardBench2 for scoring | [🔗 train_rm/grpo/pointwise](https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main/train_rm/grpo/pointwise) |
| GRPO Pairwise | RewardBench2 for comparison | [🔗 train_rm/grpo/pairwise](https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main/train_rm/grpo/pairwise) |


---


## SFT Training

Supervised Fine-Tuning learns from high-quality demonstration data. Use SFT to initialize models before preference training or when you have expert-labeled responses.

### Training Objective

$$\mathcal{L} = -\sum_{t} \log P(y_t | y_{<t}, x)$$

### Quick Start

```bash
cd cookbooks/training_judge_model/sft
bash run_sft_rm.sh
```

### Data Format

Parquet files with `messages` column (compatible with `tokenizer.apply_chat_template`):

```python
import pandas as pd

messages = [
    {"role": "user", "content": "What are the benefits of exercise?"},
    {"role": "assistant", "content": "Regular exercise improves cardiovascular health..."}
]

df = pd.DataFrame({"messages": [messages]})
df.to_parquet("train.parquet")
```

### Configuration

Key parameters in `run_sft_rm.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./models/Qwen3-14B` | Base model path |
| `TRAIN_BATCH_SIZE` | `96` | Global batch size |
| `MICRO_BATCH_SIZE` | `12` | Per-GPU micro batch |
| `MAX_LENGTH` | `4096` | Maximum sequence length |
| `SP_SIZE` | `8` | Sequence parallel size |
| `TOTAL_EPOCHS` | `1` | Training epochs |

**Data configuration:**

```yaml
data:
  train_batch_size: 96
  micro_batch_size: 12
  max_length: 4096
  truncation: right
  multiturn:
    enable: true
    messages_key: messages
```

### Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss |
| `val/loss` | Validation loss |

→ **[Full Documentation](https://github.com/agentscope-ai/OpenJudge/tree/main/cookbooks/training_judge_model/sft)**


---


## Bradley-Terry Training

Bradley-Terry training learns to rank responses by modeling preference probability. Use when you have binary preference data (chosen vs. rejected).

### Training Objective

$$\mathcal{L} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$$

### Quick Start

```bash
cd cookbooks/training_judge_model/bradley-terry
bash run_bt_rm.sh
```

### Data Format

Parquet files with `chosen` and `rejected` columns (JSON strings of message lists):

```python
import json
import pandas as pd

chosen = json.dumps([
    {"role": "user", "content": "What are the benefits of exercise?"},
    {"role": "assistant", "content": "Regular exercise improves cardiovascular health..."}
])
rejected = json.dumps([
    {"role": "user", "content": "What are the benefits of exercise?"},
    {"role": "assistant", "content": "Exercise is good for you."}
])

df = pd.DataFrame({"chosen": [chosen], "rejected": [rejected]})
df.to_parquet("train.parquet")
```

### Configuration

Key parameters in `run_bt_rm.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./models/Qwen3-8B` | Base model path |
| `TRAIN_BATCH_SIZE` | `256` | Global batch size |
| `MICRO_BATCH_SIZE` | `1` | Per-GPU micro batch |
| `MAX_LENGTH` | `4096` | Maximum sequence length |
| `LR` | `2e-6` | Learning rate |
| `TOTAL_EPOCHS` | `3` | Training epochs |
| `STRATEGY` | `fsdp2` | FSDP strategy (fsdp or fsdp2) |

**Optimizer configuration:**

```yaml
optim:
  lr: 2e-6
  weight_decay: 0.001
  warmup_steps_ratio: 0.03
  clip_grad: 2.0
  lr_scheduler: cosine
```

### Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Bradley-Terry loss |
| `train/accuracy` | Preference prediction accuracy |
| `val/loss` | Validation loss |
| `val/accuracy` | Validation accuracy |

→ **[Full Documentation](https://github.com/agentscope-ai/OpenJudge/tree/main/cookbooks/training_judge_model/bradley-terry)**


---


## GRPO Training (Reinforcement Learning)

Group Relative Policy Optimization trains generative judges that output structured evaluations with reasoning. No separate critic model required.

### Training Objective

$$\mathcal{L} = -\mathbb{E}\left[\sum_{i=1}^{G} A_i \log \pi_\theta(y_i|x)\right]$$

### Training Modes

=== "Pointwise (Absolute Scoring)"

    Rate individual responses on a 0-4 helpfulness scale.

    **Output Format:**
    ```
    <think>Analysis of response quality...</think>
    <score>3</score>
    ```

=== "Pairwise (Preference Comparison)"

    Compare two responses and select the better one.

    **Output Format:**
    ```
    <think>Comparison of Response A vs B...</think>
    <better>A</better>
    ```

### Prerequisites

GRPO requires a Ray cluster:

```bash
# Start Ray head node
ray start --head --port=6379 --dashboard-port=8265

# Verify cluster
ray status
```

### Quick Start

=== "Pointwise"

    ```bash
    cd cookbooks/training_judge_model/grpo
    bash pointwise/run_pointwise.sh
    ```

=== "Pairwise"

    ```bash
    cd cookbooks/training_judge_model/grpo
    bash pairwise/run_pairwise.sh
    ```

### Configuration

Override defaults with environment variables:

```bash
MODEL_PATH=Qwen/Qwen3-32B \
N_GPUS_PER_NODE=8 \
RAY_ADDRESS=http://localhost:8265 \
bash pointwise/run_pointwise.sh
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen3-8B` | Base model |
| `RAY_ADDRESS` | `http://127.0.0.1:8265` | Ray dashboard |
| `N_GPUS_PER_NODE` | `8` | GPUs per node |
| `TRAIN_BATCH_SIZE` | `96` | Global batch size |
| `ROLLOUT_N` | `4` | Samples per prompt |
| `KL_LOSS_COEF` | `0.001` | KL divergence coefficient |

### Metrics

| Metric | Description |
|--------|-------------|
| `train/reward_mean` | Average reward |
| `train/kl_divergence` | KL from reference model |
| `train/policy_loss` | Policy gradient loss |

→ **[Full Documentation](https://github.com/agentscope-ai/OpenJudge/tree/main/cookbooks/training_judge_model/grpo)**


---


## Troubleshooting

### OOM (Out of Memory)

- Reduce `MICRO_BATCH_SIZE` or `micro_batch_size_per_gpu`
- Enable `enable_gradient_checkpointing`
- Reduce `max_length`
- Enable `cpu_offload` (SFT/BT) or `param_offload` (GRPO)

### Training Instability

- Lower learning rate
- Increase `clip_grad` value
- Check data quality and format

### Ray Connection Issues (GRPO)

- Verify Ray is running: `ray status`
- Check `RAY_ADDRESS` is correct
- Ensure firewall allows ports 6379 and 8265


## Next Steps

- [Create Custom Graders](create_custom_graders.md) — Build graders from trained models
- [Validate on RewardBench2](../validating_graders/rewardbench2.md) — Evaluate grader quality

