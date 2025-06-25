# HelpSteer2 Pointwise Reward Model – End-to-End Training Guide

> This tutorial walks you through the full workflow **Download data ➜ Convert to Parquet ➜ Launch Ray cluster ➜ Run the training script ➜ Monitor & tune** using the HelpSteer2 dataset as a concrete example.

> **📋 For general framework concepts, system architecture, and universal troubleshooting, please refer to the [Train Reward Model General Guide](train.md). This document focuses on the HelpSteer2-specific implementation details.**

<details>
<summary><strong>📑 Table of Contents</strong></summary>

1. [Environment Setup](#1-environment-setup)
2. [Data Download](#2-data-download)
3. [Data Conversion](#3-data-conversion)
   1. [Prepare the YAML Config](#31-prepare-the-yaml-config)
   2. [Run the Conversion](#32-run-the-conversion)
4. [Launch the Ray Cluster](#4-launch-the-ray-cluster)
5. [Run the Training Script](#5-run-the-training-script)
6. [Monitoring](#6-monitoring)
7. [Inference & Evaluation](#7-inference--evaluation)
8. [FAQ](#8-faq)
9. [Technical Deep Dive](#9-technical-deep-dive)

</details>

---

## 1. Environment Setup

> **📋 For detailed environment configuration, system requirements, and runtime setup, please refer to the [Environment Configuration section](train.md#environment-configuration) in the general guide.**

Quick summary of key requirements:
- Python ≥ 3.10, CUDA ≥ 12.1, VERL ≥ 0.4.0
- Create `runtime_env.yaml` with proper environment variables
- Ensure WANDB integration for monitoring

---

## 2. Data Download

HelpSteer2 dataset: <https://huggingface.co/datasets/nvidia/helpsteer2>

The dataset contains helpfulness annotations with 0-4 scale scoring. Download and prepare it:

```bash
# Create a data directory
mkdir -p ~/data/HelpSteer2 && cd ~/data/HelpSteer2

# Download the dataset files
# You can use huggingface-hub or git clone
git clone https://huggingface.co/datasets/nvidia/helpsteer2
```

---

## 3. Data Conversion

A generic conversion script is provided at `examples/data/data_from_yaml.py`.

### 3.1 Prepare the YAML Config

`examples/train/pointwise/data_config.yaml` (feel free to copy & modify):

```yaml
# examples/train/pointwise/data_config.yaml

dataset:
  name: helpsteer2_pointwise               # Custom dataset name
  configs:
    type: local                           # Local file
    source: helpsteer2_pointwise          # Converter registered in rm_gallery.gallery.data.load
    path: ~/data/HelpSteer2/helpsteer2    # Path to downloaded dataset
  export:
    output_dir: ./examples/data/exports   # Where to put Parquet files
    formats: ["parquet"]
    preserve_structure: true
    split_ratio: {train: 0.8, test: 0.2} # Train / test split
```

### 3.2 Run the Conversion

```bash
python examples/data/data_from_yaml.py \
       --config examples/train/pointwise/data_config.yaml
```

> After it finishes, you should see the following files in `examples/data/exports/`:
>
> * `helpsteer2_train.parquet`
> * `helpsteer2_test.parquet`

Quick peek:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_parquet('examples/data/exports/helpsteer2_train.parquet').head())
PY
```

---

## 4. Launch the Ray Cluster

> **📋 For detailed Ray cluster setup instructions (single-node and multi-node configurations), please refer to the [Ray Cluster Setup section](train.md#launch-ray-distributed-cluster) in the general guide.**

Quick start for single node:
```bash
ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --dashboard-host 0.0.0.0
```

---

## 5. Run the Training Script

The script `examples/train/pointwise/run_pointwise.sh` contains common hyper-parameters. Check the following lines:

```bash
TRAIN_FILE=./examples/data/exports/helpsteer2_train.parquet
VAL_FILE=./examples/data/exports/helpsteer2_test.parquet
MODEL_PATH=/path/to/your/base/model  # e.g., Qwen3-8B
```

Make it executable (once) and launch:

```bash
chmod +x examples/train/pointwise/run_pointwise.sh
cd examples/train/pointwise
./run_pointwise.sh
```

**Key arguments**

| Argument | Description |
|----------|-------------|
| `data.custom_cls.path`          | Dataset script (`dataset.py`) |
| `custom_reward_function.path`   | Reward function (`reward_fn.py`) |
| `algorithm.adv_estimator`       | Use the GRPO estimator |
| `actor_rollout_ref.rollout.name`| Inference backend (default VLLM) |
| `trainer.total_epochs`          | Number of training epochs |

---

## 6. Monitoring

> **📋 For comprehensive monitoring setup, key metrics explanation, and troubleshooting tips, please refer to the [Training Monitoring section](train.md#training-monitoring) in the general guide.**

Quick monitoring checklist:
* **Console**: Use `ray job logs <job_id> -f` for real-time logs
* **WandB**: Ensure `WANDB_API_KEY` is set for automatic metric upload
* **Key Metrics**: Monitor `reward/mean`, `accuracy`, `kl_loss`

---

## 7. Inference & Evaluation

After training, look for **LoRA** or full weights in `checkpoints/<TIMESTAMP>/actor_latest`.

> For evaluation examples, check `external/verl/tests/e2e` or just load the weights for inference.

---

## 8. FAQ

> **📋 For comprehensive troubleshooting and common issues, please refer to the [FAQ & Troubleshooting section](train.md#faq--troubleshooting) in the general guide.**

### HelpSteer2-Specific Issues

1. **Data format validation**

   Verify HelpSteer2 dataset structure:
   ```python
   import pandas as pd
   df = pd.read_parquet('helpsteer2_train.parquet')
   print(df.columns)
   print(df['helpfulness'].value_counts())
   ```

2. **Score range configuration**

   HelpSteer2 uses 0-4 helpfulness scale. Ensure your reward function `max_error = 4` is set correctly in `reward_fn.py`.

---

## 9. Technical Deep Dive

> **📋 For comprehensive technical details, system architecture, and implementation deep dive, please refer to the [Technical Deep Dive section](train.md#technical-deep-dive) in the general guide.**

### HelpSteer2-Specific Implementation

#### 9.1 Data Converter

| Path | Class | Purpose |
|------|-------|---------|
| `rm_gallery/gallery/data/load/helpsteer2_pointwise.py` | `HelpSteer2PointwiseConverter` | Convert raw HelpSteer2 dataset to `DataSample` with helpfulness scores |

**HelpSteer2 Logic**:
1. Read `prompt`, `response`, `helpfulness` score from HelpSteer2 format
2. Create individual samples with ground truth scores (0-4 scale)
3. Handle missing or invalid helpfulness scores appropriately

#### 9.2 HelpSteer2 Dataset Configuration

Key files for HelpSteer2 implementation:
- `examples/train/pointwise/dataset.py` - Dataset loader
- `examples/train/pointwise/reward_fn.py` - Reward function with 0-4 scale
- `examples/train/pointwise/data_config.yaml` - Data configuration


#### 9.3 Prompt Template `PointwiseTrainTemplate`
```python
class PointwiseTrainTemplate(BasePromptTemplate):
    """
    The PrincipleTemplate class inherits from BasePromptTemplate and is used to define the template for principles reasoning.
    """

    score: int = Field(default=..., description="score of helpfulness from 0 to 4")
```


#### 9.4 Reward Function `reward_fn.py`

| Path | Key Function |
|------|--------------|
| `examples/train/pointwise/reward_fn.py` | `compute_score()` |

```python
# HelpSteer2-specific reward function
def pointwise_reward(predicted_score, true_score):
    """Reward function optimized for HelpSteer2's 0-4 helpfulness scale"""
    if true_score is None:
        return 0.0

    abs_error = abs(predicted_score - true_score)
    max_error = 4  # HelpSteer2 scale: 0-4

    k = 2.0  # Decay coefficient
    error_ratio = abs_error / max_error
    reward = math.exp(-k * error_ratio)

    return float(reward)
```
