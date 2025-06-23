
# HelpSteer2 Pairwise Reward Model – End-to-End Training Guide

> This tutorial walks you through the full workflow **Download data ➜ Convert to Parquet ➜ Launch Ray cluster ➜ Run the training script ➜ Monitor & tune**. You can get started even without prior experience.

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

| Component | Recommended Version |
|-----------|--------------------|
| Python    | ≥ 3.10 |
| CUDA      | ≥ 12.1 |
| PyTorch   | ≥ 2.1 |
| Ray       | ≥ 2.9 |
| VERL      | ≥ 0.4.0 |
| VLLM      | ≥ 0.8.4 |

---

## 2. Data Download

HelpSteer2 preference dataset: <https://huggingface.co/datasets/nvidia/HelpSteer2/tree/main/preference>

The dataset is released as **one archive file `preference.jsonl.gz`** (the field `split` already contains `train` / `test`). Download and extract it:

```bash
# Create a data directory
mkdir -p ~/data/HelpSteer2 && cd ~/data/HelpSteer2

# Download the archive
wget -c https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/preference/preference.jsonl.gz

# Extract (keep the original gz file)
gunzip -k preference.jsonl.gz
```

---

## 3. Data Conversion

A generic conversion script is provided at `examples/data/data_from_yaml.py`.

### 3.1 Prepare the YAML Config

`examples/train/pairwise/data_config.yaml` (feel free to copy & modify):

```yaml
# examples/train/pairwise/data_config.yaml

dataset:
  name: helpsteer2_pairwise                # Custom dataset name
  configs:
    type: local                           # Local file
    source: helpsteer2_pairwise           # Converter registered in rm_gallery.gallery.data.load
    path: ~/data/HelpSteer2/preference/preference.jsonl
  export:
    output_dir: ./examples/data/exports   # Where to put Parquet files
    formats: ["parquet"]
    preserve_structure: true
    split_ratio: {train: 0.8, test: 0.2} # Train / test split
```

### 3.2 Run the Conversion

```bash
python examples/data/data_from_yaml.py \
       --config examples/train/pairwise/data_config.yaml
```

> After it finishes, you should see the following files in `examples/data/exports/`:
>
> * `preference_train.parquet`
> * `preference_test.parquet`

Quick peek:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_parquet('examples/data/exports/preference_train.parquet').head())
PY
```

---

## 4. Launch the Ray Cluster

Example for a **single node with 8 × A100**:

```bash
ray start --head --num-gpus 8 --dashboard-host 0.0.0.0
```

> For multi-node setups, replace `--head` with `--address=<master_ip>:6379`. See the Ray docs for details.

---

## 5. Run the Training Script

The script `examples/train/pairwise/run_pairwise.sh` contains common hyper-parameters. Just check the following lines:

```bash
TRAIN_FILE=./examples/data/exports/preference_train.parquet
VAL_FILE=./examples/data/exports/preference_test.parquet
MODEL_PATH=/path/to/your/base/model
```

Make it executable (once) and launch:

```bash
chmod +x examples/train/pairwise/run_pairwise.sh
./examples/train/pairwise/run_pairwise.sh
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

The script logs to both **Console** and **Weights & Biases**:

* **Console**: `ray job logs <job_id> -f` for real-time logs.
* **WandB**: set `WANDB_API_KEY / WANDB_BASE_URL` to upload metrics automatically.

Important metrics:

| Metric | Meaning |
|--------|---------|
| `reward/mean` | Mean reward of the current epoch |
| `accuracy`    | Ratio of correct preference predictions |
| `kl_loss`     | KL divergence to the reference model |

---

## 7. Inference & Evaluation

After training, look for **LoRA** or full weights in `checkpoints/<TIMESTAMP>/actor_latest`.

> For evaluation examples, check `external/verl/tests/e2e` or just load the weights for inference.

---

## 8. FAQ

1. **`num_samples=0` error**
   The dataset is empty after filtering. Check whether `_build_messages` parses rows correctly:
   ```python
   from examples.train.pairwise.dataset import HelpfulnessPairwiseTrainDataset
   ds = HelpfulnessPairwiseTrainDataset(...)
   print(len(ds))
   ```
2. **Ray can't connect to `127.0.0.1:8265`**
   Make sure you have run `ray start --head` and that port 8265 is reachable, or update `--address` in the script.
3. **Out-of-memory**
   Lower `actor_rollout_ref.rollout.gpu_memory_utilization` or reduce `data.train_batch_size / ppo_micro_batch_size_per_gpu`.

---

## 9. Technical Deep Dive

> The following sections are for readers interested in how the **RLHF Pairwise Pipeline** is implemented under the hood.

### 9.1 Converter

| Path | Class | Purpose |
|------|-------|---------|
| `rm_gallery/gallery/data/load/helpsteer2_pairwise.py` | `HelpSteer2PairwiseConverter` | Convert raw JSONL to `DataSample` and create **both forward & reverse pairs** |

**Logic**
1. Read `prompt`, `response_1`, `response_2`, `preference_strength`.
2. Determine the preference (>0 → `response_2` is better, <0 → `response_1` is better, 0 → tie).
3. Emit two samples (forward + reverse order).

### 9.2 Dataset `HelpfulnessPairwiseTrainDataset`

| Path | Notes |
|------|-------|
| `examples/train/pairwise/dataset.py` | Dataset actually loaded by Ray workers |

* `_normalize_row`: deserialize JSON strings read by Pandas.
* `_build_messages`: parse each row and call `PairwiseComparisonTemplate.format(...)` to build the prompt.
* `_extract_ground_truth`: read `metadata.preferred`.

### 9.3 Prompt Template `PairwiseComparisonTemplate`

```python
class PairwiseComparisonTemplate(BasePromptTemplate):
    think: str       # (optional) chain-of-thought
    preference: str  # A / B / tie
```

Example:
```text
# Query
Why are apples healthy?

# Response A
...

# Response B
...

# Output Requirement
<think>...</think>
<preference>A</preference>
```

### 9.4 Reward Function `reward_fn.py`

| Path | Key Function |
|------|--------------|
| `examples/train/pairwise/reward_fn.py` | `compute_score()` |

1. Parse model output with `extract_preference_from_response(solution_str)`.
2. Compare with `metadata.preferred` to produce the reward: 1.0 (correct) / 0.0 (wrong).

### 9.5 Reward Base Class Changes

* File: `rm_gallery/gallery/rm/alignment/base.py`
* Added `BaseHelpfulnessPairWiseReward` (inherits `BaseListWisePrincipleReward`) and overrides `_evaluate` to output only the prompt—no external LLM calls required. The dataset can therefore instantiate it without errors.

### 9.6 PPO + GRPO Pipeline

1. Ray workers read the dataset and build prompts + ground truth.
2. **Actor** uses VLLM to generate `solution_str` in batches.
3. **RewardManager** calls the custom `compute_score` to get scalar rewards.
4. **GRPO Estimator** computes advantages & targets.
5. **PPO** updates the actor parameters, while the critic learns the value function.

