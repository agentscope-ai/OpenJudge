#!/usr/bin/env python3
"""
Run all OpenJudge grader evaluations in parallel.

This script runs all evaluation scripts concurrently to quickly benchmark
grader performance across all categories (text, multimodal, agent).

Usage:
    # Set environment variables
    export OPENAI_API_KEY=your_api_key
    export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

    # Run all evaluations with default models
    python run_grader_evaluations.py

    # Run specific category
    python run_grader_evaluations.py --category text

    # Run specific grader
    python run_grader_evaluations.py --category agent --grader tool_selection

    # Run with custom model
    python run_grader_evaluations.py --text-model qwen-max --agent-model qwen3-max

    # Limit concurrent workers
    python run_grader_evaluations.py --workers 4

Requirements:
    pip install py-openjudge datasets
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class EvalConfig:
    """Configuration for a single evaluation task."""

    name: str
    category: str
    script_path: str
    grader: str
    default_model: str
    expected_accuracy: str
    data_file: str


# All evaluation configurations
EVAL_CONFIGS = [
    # Text graders
    EvalConfig(
        "correctness",
        "text",
        "text/correctness/evaluate_correctness.py",
        "correctness",
        "qwen-max",
        "96-100%",
        "correctness_eval_v1.json",
    ),
    EvalConfig(
        "hallucination",
        "text",
        "text/hallucination/evaluate_hallucination.py",
        "hallucination",
        "qwen-plus",
        "70-75%",
        "hallucination_eval_v1.json",
    ),
    EvalConfig(
        "harmlessness",
        "text",
        "text/harmlessness/evaluate_harmlessness.py",
        "harmlessness",
        "qwen-plus",
        "100%",
        "harmlessness_eval_v1.json",
    ),
    EvalConfig(
        "instruction_following",
        "text",
        "text/instruction_following/evaluate_instruction_following.py",
        "instruction_following",
        "qwen-max",
        "75-80%",
        "instruction_following_eval_v1.json",
    ),
    EvalConfig(
        "relevance",
        "text",
        "text/relevance/evaluate_relevance.py",
        "relevance",
        "qwen-plus",
        "100%",
        "relevance_eval_v1.json",
    ),
    # Multimodal graders
    EvalConfig(
        "image_coherence",
        "multimodal",
        "multimodal/image_coherence/evaluate_image_coherence.py",
        "image_coherence",
        "qwen-vl-max",
        "75%",
        "image_coherence_eval_v1.json",
    ),
    EvalConfig(
        "image_helpfulness",
        "multimodal",
        "multimodal/image_helpfulness/evaluate_image_helpfulness.py",
        "image_helpfulness",
        "qwen-vl-max",
        "80%",
        "image_helpfulness_eval_v1.json",
    ),
    EvalConfig(
        "text_to_image",
        "multimodal",
        "multimodal/text_to_image/evaluate_text_to_image.py",
        "text_to_image",
        "qwen-vl-max",
        "75%",
        "text_to_image_eval_v1.json",
    ),
    # Agent graders
    EvalConfig(
        "action",
        "agent",
        "agent/action/evaluate_action.py",
        "action_alignment",
        "qwen3-max",
        "88%",
        "action_alignment.json",
    ),
    EvalConfig(
        "plan", "agent", "agent/plan/evaluate_plan.py", "plan_feasibility", "qwen3-max", "86%", "plan_feasibility.json"
    ),
    EvalConfig(
        "tool", "agent", "agent/tool/evaluate_tool.py", "tool_selection", "qwen3-max", "85%", "tool_selection.json"
    ),
    EvalConfig(
        "tool",
        "agent",
        "agent/tool/evaluate_tool.py",
        "tool_call_accuracy",
        "qwen3-max",
        "90%",
        "tool_call_accuracy.json",
    ),
    EvalConfig(
        "tool",
        "agent",
        "agent/tool/evaluate_tool.py",
        "tool_call_success",
        "qwen3-max",
        "95%",
        "tool_call_success.json",
    ),
    EvalConfig(
        "tool",
        "agent",
        "agent/tool/evaluate_tool.py",
        "tool_parameter_check",
        "qwen3-max",
        "75%",
        "tool_parameter_check.json",
    ),
    EvalConfig(
        "memory",
        "agent",
        "agent/memory/evaluate_memory.py",
        "memory_accuracy",
        "qwen3-max",
        "78%",
        "memory_accuracy.json",
    ),
    EvalConfig(
        "memory",
        "agent",
        "agent/memory/evaluate_memory.py",
        "memory_detail_preservation",
        "qwen3-max",
        "76%",
        "memory_detail_preservation.json",
    ),
    EvalConfig(
        "memory",
        "agent",
        "agent/memory/evaluate_memory.py",
        "memory_retrieval_effectiveness",
        "qwen3-max",
        "100%",
        "memory_retrieval_effectiveness.json",
    ),
    EvalConfig(
        "reflection",
        "agent",
        "agent/reflection/evaluate_reflection.py",
        "reflection_accuracy",
        "qwen3-max",
        "100%",
        "reflection_accuracy.json",
    ),
    EvalConfig(
        "reflection",
        "agent",
        "agent/reflection/evaluate_reflection.py",
        "reflection_outcome_understanding",
        "qwen3-max",
        "78%",
        "reflection_outcome_understanding.json",
    ),
    EvalConfig(
        "reflection",
        "agent",
        "agent/reflection/evaluate_reflection.py",
        "reflection_progress_awareness",
        "qwen3-max",
        "74%",
        "reflection_progress_awareness.json",
    ),
]

SUB_CATEGORIES = ["memory", "reflection", "tool"]


async def run_evaluation_async(config: EvalConfig, model: str, base_dir: Path) -> Dict:
    """Run a single evaluation asynchronously using the grader directly."""

    start_time = time.time()
    result = {
        "name": config.name,
        "category": config.category,
        "grader": config.grader,
        "model": model,
        "expected_accuracy": config.expected_accuracy,
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "elapsed_seconds": 0.0,
        "status": "pending",
    }

    try:
        # Import evaluation module dynamically
        script_path = base_dir / config.script_path
        if not script_path.exists():
            result["status"] = "script_not_found"
            return result

        # Pass environment variables to subprocess
        env = os.environ.copy()

        cmd = [sys.executable, str(script_path), "--model", model]
        if result["grader"] and result["name"] in SUB_CATEGORIES:
            cmd += ["--grader", result["grader"]]
        print(f"cmd: {cmd}")

        # Run subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent),
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)  # 15 min timeout
        except asyncio.TimeoutError:
            proc.kill()
            result["status"] = "timeout"
            result["elapsed_seconds"] = 900.0
            return result

        output = stdout.decode() + stderr.decode()

        # Parse results
        import re

        acc_match = re.search(r"Pairwise Accuracy:\s*([\d.]+)%", output)
        if acc_match:
            result["accuracy"] = float(acc_match.group(1)) / 100.0

        correct_match = re.search(r"Correct:\s*(\d+)", output)
        samples_match = re.search(r"Samples:\s*(\d+)", output)

        if correct_match:
            result["correct"] = int(correct_match.group(1))
        if samples_match:
            result["total"] = int(samples_match.group(1))

        result["elapsed_seconds"] = time.time() - start_time

        # Check for errors
        if proc.returncode != 0:
            result["status"] = f"exit_code_{proc.returncode}"
            # Store last few lines of output for debugging
            result["error_output"] = output[-500:] if len(output) > 500 else output
        elif result["total"] > 0:
            result["status"] = "success"
        else:
            result["status"] = "no_samples"
            result["error_output"] = output[-500:] if len(output) > 500 else output

    except Exception as e:
        result["status"] = f"error: {str(e)}"
        result["elapsed_seconds"] = time.time() - start_time

    return result


async def run_all_evaluations(
    categories: List[str],
    grader: str,
    text_model: str,
    multimodal_model: str,
    agent_model: str,
    max_workers: int,
    base_dir: Path,
) -> List[Dict]:
    """Run all evaluations with concurrency control."""

    if not grader:
        # Filter configs by category
        configs_to_run = [c for c in EVAL_CONFIGS if c.category in categories]
    else:
        # Filter configs by grader
        configs_to_run = [c for c in EVAL_CONFIGS if c.grader == grader]

    # Map category to model
    model_map = {
        "text": text_model,
        "multimodal": multimodal_model,
        "agent": agent_model,
    }

    print(f"\n{'='*70}")
    print("OpenJudge Grader Evaluation Suite")
    print(f"{'='*70}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Text Model: {text_model}")
    print(f"Multimodal Model: {multimodal_model}")
    print(f"Agent Model: {agent_model}")
    print(f"Max Workers: {max_workers}")
    print(f"Total Evaluations: {len(configs_to_run)}")
    print(f"{'='*70}\n")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    async def run_with_semaphore(config: EvalConfig):
        async with semaphore:
            model = model_map[config.category]
            print(f"[START] {config.name} ({config.grader}) with {model}")
            result = await run_evaluation_async(config, model, base_dir)

            status_icon = "✓" if result["status"] == "success" else "✗"
            print(
                f"[{status_icon}] {config.name}: {result['accuracy']:.1%} "
                f"({result['correct']}/{result['total']}) in {result['elapsed_seconds']:.1f}s"
            )

            # Print error output if failed
            if result["status"] != "success" and "error_output" in result:
                print(f"    Status: {result['status']}")
                print(f"    Error: {result['error_output'][-300:]}")

            return result

    # Run all evaluations concurrently
    tasks = [run_with_semaphore(config) for config in configs_to_run]
    results = await asyncio.gather(*tasks)

    return results


def print_results_table(results: List[Dict]):
    """Print results in a formatted table."""

    print(f"\n{'='*90}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*90}")

    # Group by category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    total_correct = 0
    total_samples = 0

    for cat in ["text", "multimodal", "agent"]:
        if cat not in categories:
            continue

        cat_results = categories[cat]
        cat_icon = {"text": "📝", "multimodal": "🖼️", "agent": "🤖"}[cat]

        print(f"\n{cat_icon} {cat.upper()}")
        print("-" * 90)
        print(f"{'Grader':<35} {'Model':<15} {'Accuracy':>10} {'Correct':>10} {'Expected':>12} {'Time':>8}")
        print("-" * 90)

        for r in cat_results:
            acc_str = f"{r['accuracy']:.1%}" if r["total"] > 0 else "N/A"
            correct_str = f"{r['correct']}/{r['total']}" if r["total"] > 0 else "N/A"
            time_str = f"{r['elapsed_seconds']:.1f}s"

            # Check if accuracy meets expected
            if r["total"] > 0:
                total_correct += r["correct"]
                total_samples += r["total"]

            print(
                f"{r['grader']:<35} {r['model']:<15} {acc_str:>10} {correct_str:>10} "
                f"{r['expected_accuracy']:>12} {time_str:>8}"
            )

    print(f"\n{'='*90}")
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"OVERALL: {total_correct}/{total_samples} ({overall_acc:.1%})")
    print(f"{'='*90}\n")


def main():
    """Execute parallel evaluations of OpenJudge graders across specified categories.

    Orchestrates concurrent execution of grader evaluations for text, multimodal, and agent
    categories using configurable models and concurrency limits. Validates required environment
    variables, discovers graders matching the selection criteria, runs evaluations in parallel
    via asyncio, and outputs results to console and optionally to a JSON file.

    Command-line interface supports:
        - Category filtering (text/multimodal/agent)
        - Specific grader selection
        - Model configuration per category type
        - Concurrency control for parallel execution
        - Results persistence to JSON

    Environment Requirements:
        OPENAI_API_KEY (required): API key for model inference
        OPENAI_BASE_URL (optional): API endpoint (defaults to DashScope if unset)

    Example Usage:
        $ python run_grader_evaluations.py --category agent text --workers 5 --output results.json
        $ python run_grader_evaluations.py --grader tool_call_accuracy --agent-model qwen3-max

    Notes:
        - Default behavior evaluates all graders in all three categories concurrently
        - Silent failures are prevented by explicit subprocess error checking in underlying runners
        - Total execution time includes grader discovery, setup, and all concurrent evaluations
    """
    parser = argparse.ArgumentParser(description="Run all OpenJudge grader evaluations in parallel")
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        nargs="+",
        default=["text", "multimodal", "agent"],
        choices=["text", "multimodal", "agent"],
        help="Categories to evaluate (default: all)",
    )
    parser.add_argument(
        "--grader",
        "-g",
        type=str,
        default="",
        help="grader to evaluate (default: all)",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="qwen3-32b",
        help="Model for text graders (default: qwen3-32b)",
    )
    parser.add_argument(
        "--multimodal-model",
        type=str,
        default="qwen-vl-max",
        help="Model for multimodal graders (default: qwen-vl-max)",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="qwen3-32b",
        help="Model for agent graders (default: qwen3-32b)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=5,
        help="Maximum concurrent evaluations (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if not os.getenv("OPENAI_BASE_URL"):
        print("Warning: OPENAI_BASE_URL not set, using default DashScope endpoint")
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Get base directory
    base_dir = Path(__file__).parent

    # Run evaluations
    start_time = time.time()

    results = asyncio.run(
        run_all_evaluations(
            categories=args.category,
            grader=args.grader,
            text_model=args.text_model,
            multimodal_model=args.multimodal_model,
            agent_model=args.agent_model,
            max_workers=args.workers,
            base_dir=base_dir,
        )
    )

    total_time = time.time() - start_time

    # Print results table
    print_results_table(results)
    print(f"Total evaluation time: {total_time:.1f}s")

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": total_time,
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
