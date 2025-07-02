"""
Module for composing multiple reward evaluation modules with weighting and routing strategies.
Implements base classes and concrete compositions for handling complex reward calculations.
"""

from abc import abstractmethod
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Any, Dict

from pydantic import Field

from rm_gallery.core.data.schema import DataSample, Reward
from rm_gallery.core.reward.base import BaseReward
from rm_gallery.core.reward.registry import RewardRegistry


class BaseComposition(BaseReward):
    """
    Base class for reward compositions that provides shared configuration parameters.

    Attributes:
        params: General parameters dictionary containing shared configurations like LLM settings
    """

    params: Dict[str, Any] = Field(
        default={}, description="general parameters like llm"
    )


class SimpleComposition(BaseComposition):
    """
    Composite reward module that combines multiple reward modules with weighted averaging.
    Supports both sequential and parallel execution modes for reward evaluation.

    Attributes:
        weights: Dictionary mapping reward dimension names to their respective weights
        rewards: Dict of reward module configurations or instances
        is_parallel: Flag indicating whether to evaluate modules in parallel
    """

    weights: Dict[str, float] = Field(default={}, description="weight for each reward")
    rewards: Dict[str, Dict[str, Any] | BaseReward] = Field(
        default_factory=dict, description="reward modules"
    )
    is_parallel: bool = Field(default=False, description="parallel or not")

    def __init__(self, *args, **kwargs):
        """
        Initialize reward modules from configurations.
        Converts dictionary configurations to actual reward module instances using the registry.

        Args:
            *args: Variable length argument list passed to parent constructor
            **kwargs: Arbitrary keyword arguments passed to parent constructor
        """
        super().__init__(*args, **kwargs)
        for name, reward in self.rewards.items():
            if isinstance(reward, dict):
                params = {k: v for k, v in self.params.items()}
                params.update(reward.get("params", {}))
                params["name"] = name

                if isinstance(reward["cls"], str):
                    self.rewards[name] = RewardRegistry.get(reward["cls"])(**params)

                elif issubclass(reward["cls"], BaseReward):
                    self.rewards[name] = reward["cls"](
                        **params,
                    )
                else:
                    raise ValueError(f"Invalid dimension: {reward}")
            elif isinstance(reward, str):
                self.rewards[name] = RewardRegistry.get(reward)(**self.params)
            elif isinstance(reward, BaseReward):
                self.rewards[name] = reward
            elif issubclass(reward, BaseReward):
                self.rewards[name] = reward(**self.params)
            else:
                raise NotImplementedError(f"Invalid dimension: {reward}")

    def evaluate(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        """
        Evaluate rewards using configured modules with optional parallel execution.

        Args:
            sample: Input data sample to evaluate
            thread_pool: Optional thread pool executor for parallel execution

        Returns:
            DataSample with updated reward information
        """
        # Parallel evaluation using thread pool
        if self.is_parallel and thread_pool is not None:
            sample = deepcopy(sample)
            futures = []
            for name, reward in self.rewards.items():
                futures.append(
                    thread_pool.submit(
                        reward.evaluate, sample=sample, thread_pool=thread_pool
                    )
                )

            wait(futures, return_when=ALL_COMPLETED)
            samples = [future.result() for future in futures]

            # Merge results from parallel evaluations
            for s in samples:
                sample.update(s)

        # Sequential evaluation mode
        else:
            for name, reward in self.rewards.items():
                sample = reward.evaluate(sample, thread_pool)

        # Weighted reward calculation function (executed for both parallel and sequential modes)
        def weight(reward: Reward):
            """Calculate weighted average based on configured weights"""
            w_sum = 0
            d_sum = 0
            for d in reward.details:
                w = self.weights.get(d.name, 1.0)
                w_sum += w
                d_sum += w * d.score
            if w_sum != 0:
                reward.score = d_sum / w_sum

        # Apply weighting to all output rewards
        for output in sample.output:
            weight(output.answer.reward)
            if output.steps:
                for step in output.steps:
                    weight(step.reward)

        return sample


class RouterComposition(SimpleComposition):
    """
    Base class for conditional reward routing that selects different reward compositions
    based on input sample characteristics.

    Attributes:
        router: Dictionary mapping condition keys to reward composition instances
    """

    @abstractmethod
    def _condition(self, sample: DataSample) -> str:
        """
        Determine routing condition based on input sample.
        Must be implemented by subclasses to return a router key.

        Args:
            sample: Input data sample to evaluate

        Returns:
            str: Key identifying which reward composition to use
        """
        ...

    def evaluate(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        """
        Route sample to appropriate reward composition based on condition.

        Args:
            sample: Input data sample to evaluate
            thread_pool: Optional thread pool executor for parallel execution

        Returns:
            DataSample with updated reward information
        """
        condition = self._condition(sample)
        sample = self.rewards[condition].evaluate(sample, thread_pool)
        return sample
