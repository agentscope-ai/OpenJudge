from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Any, Dict, List

from pydantic import Field

from rm_gallery.core.data.schema import DataSample, Reward
from rm_gallery.core.rm.module import BaseReward
from rm_gallery.core.rm.registry import RewardRegistry


class BaseCompositionReward(BaseReward):
    params: Dict[str, Any] = Field(
        default={}, description="general parameters like llm"
    )


class SimpleCompositionReward(BaseCompositionReward):
    weights: Dict[str, float] = Field(default={}, description="weight for each reward")
    reward_modules: List[Dict[str, Any] | BaseReward] = Field(
        default_factory=list, description="reward modules"
    )
    is_parallel: bool = Field(default=False, description="parallel or not")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.reward_modules)):
            if isinstance(self.reward_modules[i], dict):
                params = deepcopy(self.params)
                params.update(self.reward_modules[i]["params"])

                if isinstance(self.reward_modules[i]["cls"], str):
                    self.reward_modules[i] = RewardRegistry.get(
                        self.reward_modules[i]["cls"]
                    )(**params)
                elif issubclass(self.reward_modules[i]["cls"], BaseReward):
                    self.reward_modules[i] = self.reward_modules[i]["cls"](
                        **params,
                    )
                else:
                    raise ValueError(f"Invalid dimension: {self.reward_modules[i]}")

    def evaluate(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        # parallel evaluation
        if self.is_parallel:
            sample = deepcopy(sample)
            futures = []
            for dimension in self.reward_modules:
                futures.append(
                    thread_pool.submit(dimension.evaluate, sample, thread_pool)
                )

            wait(futures, return_when=ALL_COMPLETED)
            samples = [future.result() for future in futures]

            for s in samples:
                sample.update(s)
            return sample
        else:
            for dimension in self.reward_modules:
                sample = dimension.evaluate(sample, thread_pool)

        # weight reward
        def weight(reward: Reward):
            w_sum = 0
            d_sum = 0
            for d in reward.details:
                w = self.weights.get(d.name, 1.0)
                w_sum += w
                d_sum += w * d.score
            if w_sum != 0:
                reward.score = d_sum / w_sum

        for output in sample.output:
            weight(output.answer.reward)
            if output.steps:
                for step in output.steps:
                    weight(step.reward)

        return sample
