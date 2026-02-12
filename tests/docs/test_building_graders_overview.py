"""
Test code examples from docs/building_graders/overview.md
"""

import pytest

from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderScore
from openjudge.models import OpenAIChatModel
from openjudge.runner.grading_runner import GradingRunner


class TestOverviewLLMGrader:
    """Test LLM-based grader example from overview.md line 38-54"""

    def test_llm_grader_creation(self):
        """Test that the LLM grader can be created with the documented syntax"""
        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")

        grader = LLMGrader(
            name="domain_expert",
            model=model,
            template="""
            Evaluate the medical accuracy of this response:

            Query: {query}
            Response: {response}

            Return JSON: {{"score": <0.0-1.0>, "reason": "<explanation>"}}
            """,
        )

        assert grader.name == "domain_expert"
        assert grader.model == model


class TestOverviewFunctionGrader:
    """Test rule-based grader example from overview.md line 58-74"""

    @pytest.mark.asyncio
    async def test_compliance_checker(self):
        """Test the compliance checker function grader"""

        async def compliance_checker(response: str) -> GraderScore:
            """Check for required compliance statements."""
            required_terms = ["disclaimer", "terms", "conditions"]
            found = sum(term in response.lower() for term in required_terms)
            score = found / len(required_terms)

            return GraderScore(
                name="compliance_check", score=score, reason=f"Found {found}/{len(required_terms)} required terms"
            )

        grader = FunctionGrader(func=compliance_checker, name="compliance")

        # Test with all terms present
        result = await grader.aevaluate(response="Please read our disclaimer, terms, and conditions.")
        assert result.score == 1.0
        assert "3/3" in result.reason

        # Test with partial terms
        result = await grader.aevaluate(response="Please read our disclaimer.")
        assert result.score == pytest.approx(0.333, abs=0.01)


class TestOverviewCompleteExample:
    """Test complete example from overview.md line 293-342"""

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test the complete evaluation pipeline example"""

        # 1. Rule-based grader: Length check
        async def length_check(response: str) -> GraderScore:
            length = len(response)
            score = 1.0 if 50 <= length <= 500 else 0.5
            return GraderScore(name="length_check", score=score, reason=f"Length: {length} chars")

        length_grader = FunctionGrader(func=length_check, name="length")

        # Test length grader
        result = await length_grader.aevaluate(
            response="This is a test response that is long enough to pass the length check."
        )
        assert result.score == 1.0

        # 2. LLM-based grader: Domain accuracy
        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")
        accuracy_grader = LLMGrader(
            name="accuracy",
            model=model,
            template="""
            Rate technical accuracy (0.0-1.0):
            Query: {query}
            Response: {response}
            Return JSON: {{"score": <score>, "reason": "<reason>"}}
            """,
        )

        assert accuracy_grader.name == "accuracy"

        # 3. Combine in evaluation pipeline
        runner = GradingRunner(grader_configs={"length": length_grader, "accuracy": accuracy_grader})

        # Verify runner was created successfully
        assert len(runner.grader_configs) == 2


class TestOverviewIntegrationExamples:
    """Test integration examples from overview.md"""

    @pytest.mark.asyncio
    async def test_single_evaluation(self):
        """Test single evaluation example from line 234-239"""

        async def simple_grader(query: str, response: str) -> GraderScore:
            return GraderScore(name="test_grader", score=0.8, reason="Test evaluation")

        grader = FunctionGrader(func=simple_grader, name="test")

        result = await grader.aevaluate(query="What is machine learning?", response="ML is a subset of AI...")

        assert result.score == 0.8
        assert result.reason == "Test evaluation"

    @pytest.mark.asyncio
    async def test_batch_evaluation(self):
        """Test batch evaluation example from line 243-252"""

        async def simple_grader(response: str) -> GraderScore:
            return GraderScore(name="test_grader", score=0.8, reason="Test")

        custom_grader = FunctionGrader(func=simple_grader, name="test")

        runner = GradingRunner(grader_configs={"custom": custom_grader})
        results = await runner.arun([{"response": "A1"}, {"response": "A2"}])

        # results is a dict with grader names as keys
        assert "custom" in results
        assert len(results["custom"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
