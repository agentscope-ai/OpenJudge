from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Dict, List, Type

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BaseListWisePrincipleReward
from rm_gallery.core.reward.principle.generator import BaseGeneratorTemplate
from rm_gallery.core.reward.principle.iter_generator import IterPrincipleGenerator


class PrincipleClusterTemplate(BaseGeneratorTemplate):
    """
    Template class for clustering and organizing evaluation principles.

    Methods:
        format: Formats a prompt for principle clustering and optimization.
    """

    @classmethod
    def format(
        cls, examples: str, scenario: str, number: int, principles, **kwargs
    ) -> str:
        """
        Generates a structured prompt for principle clustering analysis.

        Args:
            examples: Pre-generated example principles for reference
            scenario: Contextual description of the evaluation scenario
            number: Maximum number of clustered principles to generate
            principles: Raw principles to be clustered and optimized
            **kwargs: Additional formatting parameters

        Returns:
            Formatted prompt string for principle clustering
        """
        return f"""## Overview
As an principle aggregation and analysis expert, your task is to conduct cluster analysis on a large collection of pre-generated principles for improvements based on examples and provide the optimization principles for each category in the scenario, that are different from the original principles.
**Specific Steps:**
1. Organize the provided improvement principles into distinct categories, ensuring that each category is unique and succinct.
2. Summarize the principles within each category into a sample set for that category, while retaining detailed information.

Another assistant will evaluate the completions in the scenario based on these principles.
When consolidating the principles, be sure to maintain the integrity, clarity, and conciseness of each category.

## Requirements for Principles
(1) Principles are presented from most important to least important.
(2) Principles should be as critical as possible.
(3) Each principle should consist of a brief phrase accompanied by a single sentence description.
(4) The number of final principles should be LESS THAN OR EQUAL TO {number}.
(5) Focus on summarizing recurring candidate principles.

## Input
### Scenario
{scenario}

### Original Principles
{principles}

### Examples
{examples}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class IterCumPrincipleGenerator(IterPrincipleGenerator):
    """
    Iterative principle generator that combines evaluation, generation, and clustering.

    Attributes:
        reward: Reward module for principle-based evaluation
        max_epochs: Maximum number of iteration cycles
    """

    reward: BaseListWisePrincipleReward = Field(
        default=..., description="reward module"
    )
    max_epochs: int = Field(default=2, description="max epochs")
    cluster_template: Type[BaseGeneratorTemplate] = Field(
        default=PrincipleClusterTemplate,
        description="template for clustering principles",
    )

    def run_batch(
        self,
        samples: List[DataSample],
        thread_pool: ThreadPoolExecutor,
        principles: Dict[str, str] | None = None,
    ) -> Dict[str, str]:
        """
        Executes the iterative principle generation pipeline.

        Args:
            samples: List of initial data samples
            thread_pool: Executor for parallel processing

        Returns:
            Final optimized principles dictionary after iterations
        """
        if not principles:
            principles = super().run_batch(samples, thread_pool)

        bad_samples = samples

        for i in range(self.max_epochs):
            _samples = self.evaluate(deepcopy(samples), principles, thread_pool)
            bad_samples = self._split_samples(_samples)
            futures = [
                thread_pool.submit(self.generate_with_feedback, sample, principles)
                for sample in bad_samples
            ]
            wait(futures, return_when=ALL_COMPLETED)
            bad_samples = [future.result() for future in futures]
            principles.update(self.cluster_with_feedback(bad_samples, principles))

        return principles
