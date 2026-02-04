
# OpenJudge Grader Evaluation Script Usage Guide

This repository provides a Bash script, run_grader_evals.sh, to run automated evaluation tasks for OpenJudge. The script supports customizing models, evaluation categories, graders, and the number of parallel worker processes.

## Usage Context & Background

### When Should You Use This Script?

This script is designed primarily for evaluating and validating the quality of OpenJudge graders.

1. Validating Grader Reliability
Use this script when you need to verify that a grader produces accurate, consistent, and well-calibrated scores on a standardized test set. Common situations include:
New grader development: After implementing a new grading logic (e.g., tool_call_accuracy), run evaluations to confirm the grader assigns correct scores across diverse test cases.
Grader prompt iteration: When refining a grader's prompt template, run evaluations with a fixed inference model to compare scoring stability and accuracy across prompt versions.
Regression testing: After updates to the OpenJudge codebase, run full evaluations to ensure existing graders behave consistently and have not regressed.

2. Testing Grader Compatibility Across Inference Models
While the main goal is grader validation, the script also helps assess:
Whether the same grader produces consistent scores when powered by different inference models (e.g., qwen3-32b vs. qwen3-max)
Whether upgrades to the underlying inference model cause unintended shifts in grader behavior (e.g., due to changes in the model's reasoning or instruction-following capabilities)

## Workflow

   Develop new grader
           ↓
 Run run_grader_evals.sh for validation
           ↓
    ┌──────┴──────┐
    ↓             ↓
 Scores         Scores
accurate?      inaccurate?
   ↓              ↓
Merge to      Refine grader
main branch    prompt/logic
                  ↓
             Re-run validation

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
