# OpenJudge Docker Guide

This document describes how to deploy OpenJudge using Docker. We provide two images:

| Image | Purpose | Dockerfile |
|-------|---------|------------|
| **Base Image** | Running evaluation tasks (API calls) | `Dockerfile.base` |
| **Training Image** | Judge model SFT/RL training | `Dockerfile.train` |

---

## Option 1: Base Installation (Evaluation)

For scenarios using OpenJudge for evaluation (calling LLMs via API).

### 1.1 Build Image

```bash
cd OpenJudge
docker build -f docker/Dockerfile.base -t openjudge:latest .
```

### 1.2 Start Container

```bash
docker run -it \
    -v $(pwd):/workspace/OpenJudge \
    -e OPENAI_API_KEY=your_api_key \
    --name openjudge \
    openjudge:latest
```

---

## Option 2: Training Environment

For scenarios using the [verl](https://github.com/volcengine/verl) framework for Judge model SFT/RL training.

### Environment Details

- **Base Image**: PAI PyTorch 2.8.0 + CUDA 12.8 + Python 3.12
- **Training Framework**: verl v0.6.1 (FSDP distributed training)
- **Inference Frameworks**: vLLM 0.11.0, SGLang 0.5.2

### 2.1 Build Image

```bash
cd OpenJudge
docker build -f docker/Dockerfile.train -t openjudge-train:latest .
```

### 2.2 Start Container

```bash
docker run --gpus all -it \
    --shm-size=64g \
    -v $(pwd):/workspace/OpenJudge \
    -v /path/to/your/models:/models \
    -v /path/to/your/data:/data \
    --name openjudge-train \
    openjudge-train:latest
```

**Parameter Description:**

| Parameter | Description |
|-----------|-------------|
| `--gpus all` | Use all GPUs |
| `--shm-size=64g` | Set shared memory to 64GB (required for training) |
| `-v $(pwd):/workspace/OpenJudge` | Mount current directory to container |
| `-v /path/to/your/models:/models` | Mount model directory (modify path as needed) |
| `-v /path/to/your/data:/data` | Mount data directory (modify path as needed) |
| `--name` | Container name |

### 2.3 Run Training

After entering the container:

```bash
cd /workspace/OpenJudge/cookbooks/training_judge_model/sft
bash run_sft_rm.sh
```

---
