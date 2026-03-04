# -*- coding: utf-8 -*-
"""CLI entry point for Reference Hallucination Arena.

Usage:
    python -m cookbooks.ref_hallucination_arena --config config.yaml
    python -m cookbooks.ref_hallucination_arena --config config.yaml --save
    python -m cookbooks.ref_hallucination_arena --config config.yaml --fresh
"""

import asyncio
from pathlib import Path
from typing import Optional

import fire
from loguru import logger

from cookbooks.ref_hallucination_arena.pipeline import RefArenaPipeline
from cookbooks.ref_hallucination_arena.schema import RefArenaConfig, load_config


async def _run_evaluation(
    config: RefArenaConfig,
    save: bool = False,
    resume: bool = True,
) -> None:
    """Run the evaluation pipeline."""
    pipeline = RefArenaPipeline(config=config, resume=resume)
    result = await pipeline.evaluate()

    if save:
        pipeline.save_results(result)


def main(
    config: str,
    output_dir: Optional[str] = None,
    save: bool = False,
    fresh: bool = False,
) -> None:
    """Reference Hallucination Arena CLI.

    Evaluate LLM reference recommendation capabilities by verifying
    recommended papers against Crossref, PubMed, arXiv, and DBLP.

    Args:
        config: Path to YAML configuration file.
        output_dir: Output directory for results (overrides config).
        save: Whether to save results to file.
        fresh: Start fresh, ignore any existing checkpoint.

    Examples:
        # Normal run (auto-resumes from checkpoint)
        python -m cookbooks.ref_hallucination_arena --config config.yaml --save

        # Start fresh
        python -m cookbooks.ref_hallucination_arena --config config.yaml --fresh --save
    """
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config}")
        return

    # Load config once and apply output_dir override
    loaded_config = load_config(str(config_path))
    if output_dir:
        loaded_config.output.output_dir = output_dir

    if fresh:
        logger.info("Starting fresh (ignoring checkpoint)")
        from cookbooks.ref_hallucination_arena.pipeline import CheckpointManager

        CheckpointManager(loaded_config.output.output_dir).clear()
    else:
        logger.info("Resume mode enabled")

    logger.info(f"Starting Reference Hallucination Arena with config: {config}")

    asyncio.run(
        _run_evaluation(
            loaded_config,
            save,
            resume=not fresh,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
