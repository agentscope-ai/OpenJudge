# OpenJudge Dockerfile
# Base image for running OpenJudge evaluation tasks
# For training Judge models with verl/sglang/vllm, use Dockerfile.train instead

FROM dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pytorch:2.8.0-gpu-py312-cu128-ubuntu24.04-3995b779-1764359181

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install OpenJudge with all dependencies
RUN pip install --no-cache-dir \
    "pandas>=2.2.3,<3.0.0" \
    "loguru>=0.7.3,<0.8.0" \
    "json_repair>=0.54.0,<1.0.0" \
    "pydantic>=2.11.5,<3.0.0" \
    "openai>=1.85.0,<2.0.0" \
    "tenacity>=9.1.0,<10.0.0" \
    "math-verify>=0.7.0,<0.8.0" \
    "tqdm>=4.66.0,<5.0.0" \
    "fire" \
    "numpy>=1.22.0,<2.0.0" \
    "dashscope>=1.19.0" \
    "tiktoken>=0.7.0" \
    "nltk>=3.8.1" \
    "jieba>=0.42.1" \
    "sacrebleu>=2.0.0" \
    "rouge-score>=0.1.2" \
    "python-Levenshtein>=0.20.0" \
    "scikit-learn>=1.0.0"

# Install OpenJudge from GitHub
RUN pip install --no-cache-dir git+https://github.com/agentscope-ai/OpenJudge.git

# Clean up
RUN pip cache purge

# Set default command
CMD ["/bin/bash"]
