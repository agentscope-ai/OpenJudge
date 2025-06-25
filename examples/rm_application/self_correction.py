#!/usr/bin/env python3
"""
LLM Response Refinement Example

This script demonstrates how to use the LLMRefinement class for iterative
improvement of LLM responses using reward model feedback, with comprehensive
evaluation to prove refinement effectiveness.

Key Concepts:
- Iterative Refinement: Repeatedly improve responses through feedback loops
- Reward Model Feedback: Use reward model assessments to guide improvements
- Response Evolution: Maintain response history to enable refinement
- Dynamic Prompting: Construct prompts based on feedback and history
- Quality Evaluation: Compare pre and post refinement sample quality using reward models

Use Cases:
- Enhancing LLM response quality
- Automated response optimization
- Batch data processing and improvement
- Response quality assessment and analysis
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List

from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for example strategy registration
from rm_gallery.core.data.export import create_export_module
from rm_gallery.core.data.load.base import create_load_module

# Import core modules
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.application.refinement import LLMRefinement
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward


@dataclass
class RefinementConfig:
    """
    Configuration class for the refinement process.

    Attributes:
        model_name: Name of the LLM model to use, supporting OpenAI-format models
        max_iterations: Maximum number of iterations to prevent infinite loops
        enable_thinking: Whether to enable thinking mode for enhanced reasoning
        reward_name: Name of the reward model for evaluating response quality
        max_workers: Maximum number of worker threads for concurrent processing
    """

    model_name: str = "qwen3-8b"
    max_iterations: int = 3
    enable_thinking: bool = True
    reward_name: str = "base_helpfulness_listwise"
    max_workers: int = 50


@dataclass
class DataLoadConfig:
    """
    Configuration class for data loading.

    Attributes:
        path: Path to the data file, supporting local files
        limit: Limit the number of samples to load for testing and debugging
        source: Data source type corresponding to different loading strategies
    """

    path: str = "/Users/xielipeng/RM-Gallery/data/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set/Helpfulness/Chat/Casual Conversation.json"
    load_strategy_type: str = "local"
    limit: int = 100
    source: str = "rmbbenchmark_pairwise"


@dataclass
class DataExportConfig:
    """
    Configuration class for data export.

    Attributes:
        output_dir: Output directory path
        formats: List of supported export formats (e.g., jsonl, json)
        split_ratio: Dictionary of data split ratios (e.g., train/test split)
        preserve_structure: Whether to preserve original data structure
    """

    output_dir: str = "./exports/refinement_results"
    split_ratio: Dict[str, float] = None
    preserve_structure: bool = False
    formats: List[str] = field(default_factory=lambda: ["jsonl"])


class RefinementProcessor:
    """
    Main processing class for LLM response refinement and evaluation.

    This class encapsulates the complete refinement workflow, including:
    1. Data loading and preprocessing
    2. LLM response refinement
    3. Quality evaluation and comparison
    4. Result export and analysis

    Uses lazy loading pattern to initialize components for better performance
    and resource utilization.
    """

    def __init__(
        self,
        refinement_config: RefinementConfig = None,
        export_config: DataExportConfig = None,
    ):
        """
        Initialize the processor.

        Args:
            refinement_config: Refinement configuration, uses default if None
            export_config: Export configuration, uses default if None
        """
        self.refinement_config = refinement_config or RefinementConfig()
        self.export_config = export_config or DataExportConfig()
        # Use lazy loading pattern to initialize components only when needed
        self._llm = None
        self._reward = None
        self._refiner = None
        self._export_module = None
        self._evaluator = None

    @property
    def llm(self) -> OpenaiLLM:
        """
        Lazy initialization of LLM instance.

        Returns:
            OpenaiLLM: Configured LLM instance
        """
        if self._llm is None:
            self._llm = OpenaiLLM(
                model=self.refinement_config.model_name,
                enable_thinking=self.refinement_config.enable_thinking,
            )
        return self._llm

    @property
    def reward(self):
        """
        Lazy initialization of reward model instance.

        Returns:
            Reward model instance for evaluating response quality
        """
        if self._reward is None:
            self._reward = RewardRegistry.get(self.refinement_config.reward_name)(
                name="helpfulness", llm=self.llm
            )
        return self._reward

    @property
    def refiner(self) -> LLMRefinement:
        """
        Lazy initialization of refinement module instance.

        Returns:
            LLMRefinement: Configured refinement module
        """
        if self._refiner is None:
            self._refiner = LLMRefinement(
                llm=self.llm,
                reward_module=self.reward,
                max_iterations=self.refinement_config.max_iterations,
            )
        return self._refiner

    @property
    def evaluator(self) -> BaseHelpfulnessListwiseReward:
        """
        Lazy initialization of evaluation module instance.

        Returns:
            BaseHelpfulnessListwiseReward: Evaluator for comparing pre/post refinement quality
        """
        if self._evaluator is None:
            self._evaluator = BaseHelpfulnessListwiseReward(
                llm=self.llm, name="helpfulness_evaluation"
            )
        return self._evaluator

    @property
    def export_module(self):
        """
        Lazy initialization of export module instance.

        Returns:
            Export module instance for saving processing results
        """
        if self._export_module is None:
            self._export_module = create_export_module(
                name="refinement_exporter",
                config={
                    "output_dir": self.export_config.output_dir,
                    "split_ratio": self.export_config.split_ratio,
                    "preserve_structure": self.export_config.preserve_structure,
                    "formats": self.export_config.formats,
                },
            )
        return self._export_module

    def load_data(self, load_config: DataLoadConfig) -> List[DataSample]:
        """
        Load dataset according to the specified configuration.

        Args:
            config: Data loading configuration

        Returns:
            List[DataSample]: Loaded data sample list

        Raises:
            Exception: Raised when data loading fails
        """
        try:
            load_module = create_load_module(
                name=load_config.source,
                load_strategy_type=load_config.load_strategy_type,
                data_source=load_config.source,
                config={"path": load_config.path, "limit": load_config.limit},
            )
            dataset = load_module.run()
            logger.info(f"Successfully loaded {len(dataset.datas)} samples")
            return dataset.datas
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def evaluate_refinement_improvement(
        self, original_sample: DataSample
    ) -> DataSample:
        """
        Evaluate refinement improvement by comparing original and refined responses.

        This method performs the following steps:
        1. Apply refinement processing to the original sample
        2. Use evaluator to compare original and refined responses
        3. Return sample containing evaluation results

        Args:
            original_sample: Original data sample

        Returns:
            DataSample: Sample containing evaluation results showing ranking comparison

        Raises:
            Exception: Raised when evaluation process fails
        """
        try:
            # Execute refinement processing
            refined_sample = self.refiner.run(original_sample)
            # Run evaluation comparison
            evaluated_sample = self.evaluator.evaluate(sample=refined_sample)
            return evaluated_sample
        except Exception as e:
            logger.error(
                f"Evaluation failed for sample {original_sample.unique_id}: {e}"
            )
            raise

    def process_single_sample(self, sample: DataSample) -> DataSample:
        """
        Process a single sample through the complete workflow.

        Includes the complete processing pipeline with refinement and evaluation,
        providing detailed logging.

        Args:
            sample: Data sample to be processed

        Returns:
            DataSample: Processed sample (containing refinement and evaluation results)

        Raises:
            Exception: Raised when processing encounters errors
        """
        try:
            logger.info(f"Starting to process sample: {sample.unique_id}")
            result = self.evaluate_refinement_improvement(sample)
            logger.info(f"Completed processing sample: {sample.unique_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to process sample {sample.unique_id}: {e}")
            raise

    def batch_refinement_with_evaluation(self, samples: List[DataSample]) -> Dict:
        """
        Perform batch refinement and evaluation using ThreadPoolExecutor.

        This method implements efficient parallel processing:
        1. Use thread pool to process multiple samples in parallel
        2. Automatically handle exceptions and error recovery
        3. Export all processing results

        Args:
            samples: List of samples to be processed

        Returns:
            Dict: Processing result statistics

        Raises:
            Exception: Raised when batch processing fails
        """
        logger.info(
            f"Starting parallel refinement and evaluation for {len(samples)} samples"
        )

        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(
                max_workers=self.refinement_config.max_workers
            ) as executor:
                evaluation_results = list(
                    executor.map(self.process_single_sample, samples)
                )

            # Export all results (both successful and failed are DataSample objects)
            self.export_module.run(evaluation_results)

            logger.info(
                f"Batch processing completed, processed {len(evaluation_results)} samples"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise


def main():
    """
    Main function demonstrating complete batch refinement and evaluation workflow.

    Execution steps:
    1. Initialize configuration parameters
    2. Create processor instance
    3. Load dataset
    4. Run batch refinement and evaluation
    5. Export results
    """
    # Configuration initialization
    refinement_config = RefinementConfig()
    export_config = DataExportConfig()
    load_config = DataLoadConfig()

    # Initialize processor
    processor = RefinementProcessor(
        refinement_config=refinement_config, export_config=export_config
    )

    try:
        # Load dataset
        dataset = processor.load_data(load_config)

        # Run batch refinement and evaluation
        logger.info("Starting batch refinement and evaluation workflow")
        processor.batch_refinement_with_evaluation(dataset)
        logger.info("Batch refinement and evaluation workflow completed")

    except Exception as e:
        logger.error(f"Main workflow execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
