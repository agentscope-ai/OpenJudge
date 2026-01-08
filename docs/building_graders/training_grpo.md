# Train with GRPO (Reinforcement Learning)

Train generative judge models using Group Relative Policy Optimization (GRPO). This approach produces models that output structured evaluations with reasoning, making them ideal for interpretable AI assessment.


## Overview

GRPO is a reinforcement learning method that trains models to generate evaluation outputs by optimizing against rule-based reward signals. Unlike scalar reward models, GRPO-trained judges produce interpretable outputs with explicit reasoning.

**Key Advantages:**

- **Interpretable:** Outputs include thinking process and final judgment
- **Flexible:** Supports both absolute scoring and comparative evaluation
- **No Critic:** Simpler training without separate value network
- **Reasoning:** Leverages model's chain-of-thought capabilities


## Training Modes

### Pointwise (Absolute Scoring)

Train models to rate individual responses on a 0-4 helpfulness scale.

**Output Format:**
```
<think>Analysis of response quality...</think>
<score>3</score>
```

**Best For:**
- Quality filtering pipelines
- Response ranking
- Reward signals for RLHF

### Pairwise (Preference Comparison)

Train models to compare two responses and select the better one.

**Output Format:**
```
<think>Comparison of Response A vs B...</think>
<better>A</better>
```

**Best For:**
- Preference data generation
- Model comparison evaluation
- Arena-style benchmarking


## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install verl==0.6.1

# Start Ray cluster
ray start --head --port=6379 --dashboard-port=8265
```

### Run Training

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


## Configuration

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


## Dataset

Download pre-processed datasets from HuggingFace:

| Mode | Link |
|------|------|
| Pointwise | [agentscope-ai/OpenJudge/.../grpo/pointwise](https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main/train_rm/grpo/pointwise) |
| Pairwise | [agentscope-ai/OpenJudge/.../grpo/pairwise](https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main/train_rm/grpo/pairwise) |


## Full Documentation

For complete configuration options, data format specifications, Ray cluster setup, and troubleshooting:

→ **[See Cookbook README](https://github.com/modelscope/OpenJudge/tree/main/cookbooks/training_judge_model/grpo)**


## Comparison with Other Methods

| Method | Output Type | Training Data | Interpretable | Best For |
|--------|-------------|---------------|---------------|----------|
| **GRPO** | Generative (text) | Labeled responses | ✅ Yes | Interpretable evaluation |
| **Bradley-Terry** | Scalar score | Preference pairs | ❌ No | Reward modeling |
| **SFT** | Generative (text) | Demonstrations | ✅ Yes | Response generation |


## Next Steps

- [Bradley-Terry Training](https://github.com/modelscope/OpenJudge/tree/main/cookbooks/training_judge_model/bradley-terry) — Train scalar reward models
- [SFT Training](https://github.com/modelscope/OpenJudge/tree/main/cookbooks/training_judge_model/sft) — Supervised fine-tuning

