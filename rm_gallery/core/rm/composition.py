from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Any, Dict, List

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.rm.module import BaseReward
from rm_gallery.core.rm.registry import RewardRegistry


class BaseComposition(BaseReward):
    params: Dict[str, Any] = Field(
        default={}, description="general parameters like llm"
    )


class SequenceComposition(BaseComposition):
    reward_modules: List[Dict[str, Any] | BaseReward] = Field(
        default_factory=list, description="reward modules"
    )

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
        for dimension in self.reward_modules:
            sample = dimension.evaluate(sample, thread_pool)

        return sample


class ParallelComposition(SequenceComposition):
    def evaluate(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        sample = deepcopy(sample)
        futures = []
        for dimension in self.reward_modules:
            futures.append(thread_pool.submit(dimension.run, sample, thread_pool))

        wait(futures, return_when=ALL_COMPLETED)
        samples = [future.result() for future in futures]

        for s in samples:
            sample.update(s)
        return sample
