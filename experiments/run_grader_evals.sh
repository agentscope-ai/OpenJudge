#!/bin/bash
set -e

# Display help information
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

This script runs the OpenJudge evaluation task.

Optional arguments:
  --agent-model MODEL       Specify the language model used by the agent (Optional - Default: qwen3-32b)
  --text-model MODEL        Specify the text evaluation model (Optional - Default: qwen3-32b)
  --multimodal-model MODEL  Specify the multimodal evaluation model (Optional - Default: qwen-vl-max)
  --workers N               Specify the number of parallel worker processes (Optional - Default: 5)
  --category CAT            Specify the evaluation category (Optional: mutually exclusive with --grader. Default: all categories)
  --grader GRADER           Specify a custom grader name (Optional: mutually exclusive with --category. Default: all graders)
  --help, -h                Show this help message and exit

Notes:
  - --category and --grader cannot be used together.
  - If no arguments are provided, the script runs with default settings.
EOF
    exit 0
}


agent_model=""
text_model=""
multimodal_model=""
workers=""
category=""
grader=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent-model)
            agent_model="$2"
            shift 2
            ;;
        --text-model)
            text_model="$2"
            shift 2
            ;;
        --multimodal-model)
            multimodal_model="$2"
            shift 2
            ;;
        --workers)
            workers="$2"
            shift 2
            ;;
        --category)
            category="$2"
            shift 2
            ;;
        --grader)
            grader="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# Check if both --category and --grader are specified (they are mutually exclusive)
if [[ -n "$category" && -n "$grader" ]]; then
    echo "Error: --category and --grader cannot be used at the same time." >&2
    exit 1
fi

# Install dependencies (only needed on first run; comment out for subsequent runs)
pip install huggingface_hub
hf download agentscope-ai/OpenJudge --repo-type dataset --local-dir agentscope-ai/OpenJudge
pip install py-openjudge datasets

# Set environment variables (replace 'your_dashscope_api_key' with your actual key if needed)
# export OPENAI_API_KEY=your_dashscope_api_key
# export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Copy the evaluation script into the dataset directory
cp run_grader_evaluations.py agentscope-ai/OpenJudge/
cd agentscope-ai/OpenJudge

# Build the command
cmd_array=("python" "run_grader_evaluations.py")

# Add either --category or --grader (mutually exclusive)
if [[ -n "$category" ]]; then
    cmd_array+=("--category" "$category")
elif [[ -n "$grader" ]]; then
    cmd_array+=("--grader" "$grader")
fi

# Append other optional arguments if they are non-empty
[[ -n "$agent_model" ]] && cmd_array+=("--agent-model" "$agent_model")
[[ -n "$text_model" ]] && cmd_array+=("--text-model" "$text_model")
[[ -n "$multimodal_model" ]] && cmd_array+=("--multimodal-model" "$multimodal_model")
[[ -n "$workers" ]] && cmd_array+=("--workers" "$workers")

echo "Executing command: ${cmd_array[*]}"

# Execute the command
"${cmd_array[@]}"
