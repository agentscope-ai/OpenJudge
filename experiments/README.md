
# OpenJudge Grader Evaluation Script Usage Guide

This repository provides a Bash script, run_grader_evals.sh, to run automated evaluation tasks for OpenJudge. The script supports customizing models, evaluation categories, graders, and the number of parallel worker processes.

## Quick Start

### Make the script executable

```bash
chmod +x run_grader_evals.sh
```

### Run evaluation with default settings (all categories, default models)

```bash
./run_grader_evals.sh
```

## Command-Line Options

| Option | Description                                               | Default |
|--------|-----------------------------------------------------------|---------|
| `--agent-model MODEL` | Language model used by the agent                          | `qwen3-32b` |
| `--text-model MODEL` | Model used for text-based evaluation                      | `qwen3-32b` |
| `--multimodal-model MODEL` | Model used for multimodal evaluation                      | `qwen-vl-max` |
| `--workers N` | Number of parallel worker processes                       | `5` |
| `--category CAT` | Evaluation category (e.g., `agent`, `text`, `multimodal`) | All categories |
| `--grader GRADER` | Evaluation grader name                                    | All graders |
| `--help` or `-h` | Show help message                                         | — |

Note: --category and --grader are mutually exclusive—you cannot use both at the same time.

## Usage Examples

### Example 1: Evaluate only the "agent" category

```bash
./run_grader_evals.sh --category agent
```

### Example 2: Evaluate the specific grader named tool_call_accuracy

```bash
./run_grader_evals.sh --grader tool_call_accuracy
```

### Example 3: Specify different models and increase parallelism

```bash
./run_grader_evals.sh \
  --agent-model qwen3-max \
  --text-model qwen3-32b \
  --multimodal-model qwen-vl-max \
  --workers 5
```

## API Key Configuration (Optional)
If you're using DashScope-compatible models (e.g., Qwen series), set your API credentials before running:

```bash
export OPENAI_API_KEY="your_dashscope_api_key"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```
Tip: The script includes commented-out export lines—uncomment and fill in your key to enable.

The script automatically:
- Installs Python dependencies: py-openjudge, datasets, huggingface_hub
```bash
  pip install py-openjudge datasets huggingface_hub
```
- Downloads the dataset to: agentscope-ai/OpenJudge
```bash
hf download agentscope-ai/OpenJudge --repo-type dataset --local-dir agentscope-ai/OpenJudge
```
- Copies run_grader_evaluations.py into the dataset directory and executes it
- Ensure that run_grader_evaluations.py exists in the same directory as the script.

## Help

Display full usage instructions:

```bash
./run_grader_evals.sh --help
```
