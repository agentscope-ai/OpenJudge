from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.reward.base import BaseReward
from rm_gallery.core.reward.template import BasePromptTemplate


class LLMRefinement(BaseModule):
    """
    refer: https://selfrefine.info/
    """

    reward_module: BaseReward = Field(default=..., description="reward module")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: BasePromptTemplate = Field(default=..., description="prompt template")

    def run(self, sample: DataSample, **kwargs) -> DataSample:
        while not self._loop_can_stop():
            sample = self._execute_one_loop(sample, **kwargs)
        return sample

    def _loop_can_stop(self) -> bool:
        """
        Check if the loop can stop.
        """
        return False

    def _execute_one_loop(self, sample: DataSample, **kwargs) -> DataSample:
        """
        Execute one loop of the self-refining process.
        """
        sample = self._generate_response_with_feedback(sample, **kwargs)
        return self._generate_feedback(sample, **kwargs)

    def _generate_response_with_feedback(
        self, sample: DataSample, **kwargs
    ) -> DataSample:
        """
        Generate a response with feedback.
        feedback is None in the first loop
        """
        response = self.llm.simple_chat(
            query=self.template.format(**kwargs),
        )

        sample.output[0].answer.content = response
        return sample

    def _generate_feedback(self, sample: DataSample, **kwargs) -> DataSample:
        """
        Generate feedback for the response.
        """
        return self.reward_module.evaluate(sample, **kwargs)
