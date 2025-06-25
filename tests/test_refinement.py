# ... existing code ...

from loguru import logger

from rm_gallery.core.data.schema import DataSample, Reward, Step
from rm_gallery.core.model.message import MessageRole
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.application.refinement import LLMRefinement
from rm_gallery.core.reward.base import BaseReward
from rm_gallery.core.reward.schema import RewardDimensionWithScore


class MockReward(BaseReward):
    """Mock reward class for testing"""

    def evaluate(self, sample: DataSample, **kwargs):
        # Add minimal evaluation logic for testing
        sample.output[0].answer.reward = Reward(
            details=[
                RewardDimensionWithScore(name=self.name, score=1.0, reason="too short")
            ]
        )
        return sample


def test_initialization():
    """Test basic initialization of LLMRefinement"""
    llm = OpenaiLLM(model="qwen3-32b", enable_thinking=False)
    reward = MockReward(
        name="mock_reward",
    )

    refinement = LLMRefinement(llm=llm, reward_module=reward)
    assert refinement.llm == llm
    assert refinement.reward_module == reward
    assert refinement.max_iterations == 3


def test_run_with_sample():
    """Test running refinement with a sample input"""
    # Setup
    llm = OpenaiLLM(model="qwen3-32b", enable_thinking=True)
    reward = MockReward(
        name="mock_reward",
    )
    refinement = LLMRefinement(llm=llm, reward_module=reward, max_iterations=1)

    # Create test sample
    sample = DataSample(
        unique_id="mock_id",
        input=[Step(role=MessageRole.USER, content="Hello, how are you?")],
        output=[],
    )

    # Run refinement
    result = refinement.run(sample)

    # Basic assertions
    assert isinstance(result, DataSample)
    assert len(result.output) > 0
    logger.info(f"Result output: {result.output}")
