"""
Test code examples from docs/building_graders/custom-graders.md
"""

import json
import re
from typing import List

import pytest
from pydantic import BaseModel

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderRank, GraderScore
from openjudge.models import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import PromptTemplate
from openjudge.runner.grading_runner import GradingRunner


class TestCustomGradersBasicLLM:
    """Test basic LLM grader examples from custom-graders.md"""

    def test_helpfulness_grader_creation(self):
        """Test helpfulness grader example from line 70-101"""
        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")

        helpfulness_grader = LLMGrader(
            name="helpfulness_evaluator",
            mode="pointwise",
            model=model,
            template="""
            You are an expert evaluator assessing the helpfulness of AI responses.

            Query: {query}
            Response: {response}

            Rate helpfulness on a scale of 0.0 to 1.0, where:
            - 0.0 = Not helpful at all
            - 1.0 = Extremely helpful

            Consider: accuracy, completeness, clarity, and relevance.

            Provide your response in JSON format:
            {{
                "score": <numerical_score>,
                "reason": "<explanation>"
            }}
            """,
            description="Evaluates response helpfulness",
        )

        assert helpfulness_grader.name == "helpfulness_evaluator"
        assert helpfulness_grader.mode == "pointwise"


class TestResponseParsingMethods:
    """Test response parsing methods from custom-graders.md"""

    def test_json_parsing_grader(self):
        """Test automatic JSON parsing example from line 120-143"""
        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")

        grader = LLMGrader(
            name="json_evaluator",
            mode="pointwise",
            model=model,
            template="""
            Evaluate the response to the query.

            Query: {query}
            Response: {response}

            Return JSON:
            {{
                "score": <0.0 to 1.0>,
                "reason": "<explanation>"
            }}
            """,
        )

        assert grader.name == "json_evaluator"

    def test_structured_output_grader(self):
        """Test structured output with Pydantic from line 148-183"""

        class DetailedEvaluation(BaseModel):
            score: float
            reason: str
            strengths: List[str]
            weaknesses: List[str]

        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")

        grader = LLMGrader(
            name="structured_grader",
            mode="pointwise",
            model=model,
            template="""
            Evaluate the response:

            Query: {query}
            Response: {response}

            Provide:
            - Score (0.0 to 1.0)
            - Reason
            - Strengths (list)
            - Weaknesses (list)
            """,
            structured_model=DetailedEvaluation,
        )

        assert grader.name == "structured_grader"

    def test_custom_callback_grader(self):
        """Test custom callback function from line 188-245"""

        def extract_metadata(chat_response) -> dict:
            """Extract confidence and other metadata from LLM response."""
            response_text = chat_response.content
            if isinstance(response_text, list) and len(response_text) > 0:
                response_text = response_text[0].text if hasattr(response_text[0], "text") else str(response_text[0])
            else:
                response_text = str(response_text)

            # Parse JSON response
            try:
                parsed = json.loads(response_text)
                score = parsed.get("score", 0.0)
                reason = parsed.get("reason", "No reason provided")
                confidence = parsed.get("confidence", 0.5)
            except json.JSONDecodeError:
                # Fallback: regex extraction
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', response_text)
                score = float(score_match.group(1)) if score_match else 0.0
                reason = "Extracted from unstructured response"
                confidence = 0.5

            return {"confidence": confidence, "extracted_score": score, "extracted_reason": reason}

        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")

        grader = LLMGrader(
            name="metadata_grader",
            mode="pointwise",
            model=model,
            template="""
            Evaluate the response:

            Query: {query}
            Response: {response}

            Return JSON with score, reason, and confidence:
            {{
                "score": <0.0 to 1.0>,
                "reason": "<explanation>",
                "confidence": <0.0 to 1.0>
            }}
            """,
            callback=extract_metadata,
        )

        assert grader.name == "metadata_grader"


class TestMultilingualGrader:
    """Test multilingual grader from line 249-280"""

    def test_multilingual_grader_creation(self):
        """Test multilingual grader with localized prompts"""
        model = OpenAIChatModel(model="qwen3-32b", api_key="your_api_key")

        template = PromptTemplate(
            messages={
                "en": [
                    ChatMessage(role="system", content="You are an English evaluator."),
                    ChatMessage(role="user", content="Query: {query}\nResponse: {response}\nRate helpfulness:"),
                ],
                "zh": [
                    ChatMessage(role="system", content="你是一个中文评估者。"),
                    ChatMessage(role="user", content="问题: {query}\n回答: {response}\n评估有用性:"),
                ],
            }
        )

        grader = LLMGrader(name="multilingual_grader", mode="pointwise", model=model, template=template)

        assert grader.name == "multilingual_grader"


class TestRuleBasedGraders:
    """Test rule-based grader examples"""

    @pytest.mark.asyncio
    async def test_length_evaluator(self):
        """Test pointwise length check from line 294-313"""

        async def length_evaluator(query: str, response: str) -> GraderScore:
            """Evaluate response length."""
            length = len(response)
            score = min(length / 100.0, 1.0)  # Normalize to 0-1

            return GraderScore(name="length_grader", score=score, reason=f"Length: {length} chars (target: 100+)")

        grader = FunctionGrader(func=length_evaluator, name="length_check", mode="pointwise")

        result = await grader.aevaluate(query="Test query", response="Short")
        assert 0.0 <= result.score <= 1.0
        assert "Length:" in result.reason

    @pytest.mark.asyncio
    async def test_length_ranker(self):
        """Test listwise length ranking from line 317-337"""

        async def length_ranker(query: str, answer_1: str, answer_2: str) -> GraderRank:
            """Rank responses by length (shorter is better)."""
            lengths = [len(answer_1), len(answer_2)]
            rank = [1, 2] if lengths[0] <= lengths[1] else [2, 1]

            return GraderRank(name="length_ranker", rank=rank, reason=f"Lengths: {lengths[0]} vs {lengths[1]} chars")

        grader = FunctionGrader(func=length_ranker, name="length_ranking", mode="listwise")

        result = await grader.aevaluate(query="Test query", answer_1="Short", answer_2="Much longer response")
        assert result.rank == [1, 2]
        assert "Lengths:" in result.reason


class TestComplexLogicGraders:
    """Test complex logic with BaseGrader"""

    @pytest.mark.asyncio
    async def test_regex_pattern_grader(self):
        """Test regex pattern grader from line 348-375"""

        class RegexPatternGrader(BaseGrader):
            """Validate responses against regex patterns."""

            def __init__(self, pattern: str, flags: int = 0, **kwargs):
                super().__init__(**kwargs)
                self.pattern = re.compile(pattern, flags)

            async def _aevaluate(self, response: str, **kwargs) -> GraderScore:
                """Check if response matches pattern."""
                match = self.pattern.search(response)

                return GraderScore(
                    name=self.name,
                    score=1.0 if match else 0.0,
                    reason="Pattern matched" if match else "Pattern not found",
                    metadata={"pattern": self.pattern.pattern},
                )

        email_checker = RegexPatternGrader(
            name="email_validator", pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        # Test with valid email
        result = await email_checker.aevaluate(response="Contact us at test@example.com")
        assert result.score == 1.0

        # Test without email
        result = await email_checker.aevaluate(response="No email here")
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_content_quality_grader(self):
        """Test multi-criteria quality check from line 378-423"""

        class ContentQualityGrader(BaseGrader):
            """Evaluate content using multiple criteria."""

            def __init__(self, min_length: int = 10, required_keywords: list = None, **kwargs):
                super().__init__(**kwargs)
                self.min_length = min_length
                self.required_keywords = required_keywords or []

            async def _aevaluate(self, response: str, **kwargs) -> GraderScore:
                """Check length and keyword requirements."""
                # Check length
                length_ok = len(response) >= self.min_length

                # Check keywords
                keywords_found = [kw.lower() in response.lower() for kw in self.required_keywords]

                # Calculate score
                checks_passed = sum([length_ok] + keywords_found)
                total_checks = 1 + len(self.required_keywords)
                score = checks_passed / total_checks

                # Generate reason
                reasons = [f"Length: {len(response)} chars ({'✓' if length_ok else '✗'})"]
                for kw, found in zip(self.required_keywords, keywords_found):
                    reasons.append(f"Keyword '{kw}': {'✓' if found else '✗'}")

                return GraderScore(name=self.name, score=score, reason=" | ".join(reasons))

        quality_grader = ContentQualityGrader(
            name="content_quality", min_length=50, required_keywords=["introduction", "conclusion", "evidence"]
        )

        # Test with all criteria met
        result = await quality_grader.aevaluate(
            response="This is an introduction with evidence and a conclusion for our discussion."
        )
        assert result.score == 1.0

        # Test with partial criteria
        result = await quality_grader.aevaluate(response="Short")
        assert result.score < 1.0


class TestBestPractices:
    """Test best practices examples"""

    @pytest.mark.asyncio
    async def test_robust_grader(self):
        """Test error handling from line 434-452"""

        class RobustGrader(BaseGrader):
            async def _aevaluate(self, **kwargs) -> GraderScore:
                try:
                    # Evaluation logic that might fail
                    if "error" in kwargs.get("response", ""):
                        raise ValueError("Simulated error")

                    return GraderScore(name=self.name, score=1.0, reason="Success")
                except Exception as e:
                    # Graceful failure
                    return GraderScore(
                        name=self.name, score=0.0, reason=f"Error: {str(e)}", metadata={"error_type": type(e).__name__}
                    )

        grader = RobustGrader(name="robust_test")

        # Test normal case
        result = await grader.aevaluate(response="Normal response")
        assert result.score == 1.0

        # Test error case
        result = await grader.aevaluate(response="error case")
        assert result.score == 0.0
        assert "Error:" in result.reason


class TestGraderTesting:
    """Test grader testing examples from line 481-523"""

    @pytest.mark.asyncio
    async def test_custom_grader_validation(self):
        """Test grader validation workflow"""

        async def simple_math_checker(query: str, response: str) -> GraderScore:
            """Check if response contains correct answer."""
            if "2+2" in query:
                score = 1.0 if "4" in response or "four" in response.lower() else 0.0
            else:
                score = 0.5

            return GraderScore(
                name="math_checker", score=score, reason="Correct answer" if score == 1.0 else "Incorrect or unknown"
            )

        grader = FunctionGrader(func=simple_math_checker, name="math_test")

        test_cases = [
            {"query": "What is 2+2?", "response": "4", "expected": 1.0},
            {"query": "What is 2+2?", "response": "The answer is four.", "expected": 1.0},
            {"query": "What is 2+2?", "response": "I don't know.", "expected": 0.0},
        ]

        for case in test_cases:
            result = await grader.aevaluate(query=case["query"], response=case["response"])
            assert result.score == case["expected"]

    @pytest.mark.asyncio
    async def test_integration_with_runner(self):
        """Test integration testing with GradingRunner from line 510-523"""

        async def simple_grader(response: str) -> GraderScore:
            return GraderScore(name="simple", score=0.8, reason="Test")

        grader = FunctionGrader(func=simple_grader, name="test_grader")
        runner = GradingRunner(grader_configs={"test": grader})

        results = await runner.arun([{"response": "A1"}, {"response": "A2"}])

        # results is a dict with grader names as keys
        assert "test" in results
        assert len(results["test"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
