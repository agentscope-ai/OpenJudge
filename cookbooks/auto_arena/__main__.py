# -*- coding: utf-8 -*-
"""CLI entry point for Auto Arena evaluation.

Usage:
    python -m cookbooks.auto_arena --config config.yaml
    python -m cookbooks.auto_arena --config config.yaml --save
    python -m cookbooks.auto_arena --config config.yaml --queries_file queries.json --save
    python -m cookbooks.auto_arena --config config.yaml --rerun-judge --save
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import fire
from loguru import logger

from cookbooks.auto_arena.auto_arena_pipeline import AutoArenaPipeline
from cookbooks.auto_arena.schema import GeneratedQuery, load_config


def _load_queries_from_file(queries_file: str) -> List[GeneratedQuery]:
    """Load pre-generated queries from JSON file."""
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = [GeneratedQuery(**item) for item in data]
    logger.info(f"Loaded {len(queries)} queries from {queries_file}")
    return queries


def _clear_judge_results(output_dir: str) -> bool:
    """Clear only comparison results to re-run with new judge model.

    Keeps queries, responses, and rubrics intact.
    Only removes comparison_details.json and resets checkpoint to RUBRICS_GENERATED.

    Returns:
        True if successfully cleared, False if no checkpoint exists
    """
    output_path = Path(output_dir)
    checkpoint_file = output_path / "checkpoint.json"
    details_file = output_path / "comparison_details.json"

    if not checkpoint_file.exists():
        logger.warning("No checkpoint found, nothing to clear")
        return False

    # Remove comparison details
    if details_file.exists():
        details_file.unlink()
        logger.info(f"Removed {details_file}")

    # Update checkpoint to RUBRICS_GENERATED stage
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    checkpoint["stage"] = "rubrics_generated"
    checkpoint["evaluated_pairs"] = 0
    checkpoint["total_pairs"] = 0

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    logger.info("Reset checkpoint to RUBRICS_GENERATED stage (will re-run pairwise evaluation)")
    return True


async def _run_evaluation(
    config_path: str,
    output_dir: Optional[str] = None,
    queries_file: Optional[str] = None,
    save: bool = False,
    resume: bool = True,
) -> None:
    """Run evaluation pipeline.

    Args:
        config_path: Path to configuration file
        output_dir: Output directory (overrides config)
        queries_file: Path to pre-generated queries JSON file (skip generation)
        save: Whether to save results to file
        resume: Whether to resume from checkpoint
    """
    config = load_config(config_path)

    if output_dir:
        config.output.output_dir = output_dir

    # Load pre-generated queries if provided
    queries = None
    if queries_file:
        queries = _load_queries_from_file(queries_file)

    pipeline = AutoArenaPipeline(config=config, resume=resume)
    result = await pipeline.evaluate(queries=queries)

    if save:
        pipeline.save_results(result, output_dir)


def main(
    config: str,
    output_dir: Optional[str] = None,
    queries_file: Optional[str] = None,
    save: bool = False,
    fresh: bool = False,
    rerun_judge: bool = False,
) -> None:
    """Auto Arena evaluation CLI with checkpoint support.

    Args:
        config: Path to YAML configuration file
        output_dir: Output directory for results
        queries_file: Path to pre-generated queries JSON (skip query generation)
        save: Whether to save results to file
        fresh: Start fresh, ignore any existing checkpoint
        rerun_judge: Re-run only pairwise evaluation with new judge model
                     (keeps queries, responses, and rubrics)

    Examples:
        # Normal run (auto-resumes from checkpoint)
        python -m cookbooks.auto_arena --config config.yaml --save

        # Use pre-generated queries
        python -m cookbooks.auto_arena --config config.yaml --queries_file queries.json --save

        # Start fresh, ignore checkpoint
        python -m cookbooks.auto_arena --config config.yaml --fresh --save

        # Re-run with new judge model (keeps queries/responses/rubrics)
        python -m cookbooks.auto_arena --config config.yaml --rerun-judge --save
    """
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config}")
        return

    if queries_file:
        queries_path = Path(queries_file)
        if not queries_path.exists():
            logger.error(f"Queries file not found: {queries_file}")
            return

    # Load config to get output_dir
    loaded_config = load_config(str(config_path))
    effective_output_dir = output_dir or loaded_config.output.output_dir

    # Handle rerun_judge and fresh flags
    if rerun_judge:
        if fresh:
            logger.warning("--rerun-judge and --fresh are mutually exclusive, using --rerun-judge")
        logger.info("Re-running pairwise evaluation with new judge model...")
        if not _clear_judge_results(effective_output_dir):
            logger.info("No previous results found, will run full evaluation")
    elif fresh:
        logger.info("Starting fresh (ignoring checkpoint)")
    else:
        logger.info("Resume mode enabled (will continue from checkpoint if exists)")

    logger.info(f"Starting Auto Arena evaluation with config: {config}")
    if queries_file:
        logger.info(f"Using pre-generated queries from: {queries_file}")

    asyncio.run(_run_evaluation(str(config_path), output_dir, queries_file, save, resume=not fresh or rerun_judge))


if __name__ == "__main__":
    fire.Fire(main)
