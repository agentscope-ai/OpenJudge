# -*- coding: utf-8 -*-
"""
Query-Specific Rubric Generator with Full Iterative Capability

This module contains the complete logic for query-specific rubric generation:
- Generate rubrics
- Evaluate samples
- Validate results
- Revise rubrics
- Iterative improvement loop

Prompts are imported from prompts.py (Chat-based)
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from rm_gallery.core.grader.auto.rubric.prompts import RubricPromptTemplates
from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.data import EvalCase, EvalCaseParser
from rm_gallery.core.schema.template import LanguageEnum


class RubricGenerationOutput(BaseModel):
    rubrics: List[str] = Field(description="List of generated rubrics")
    reason: str = Field(description="Reasoning for the generated rubrics")


class QuerySpecificRubricGenerator:
    """
    Complete query-specific rubric generator with iterative improvement

    Workflow for each sample:
    1. Generate initial rubrics based on sample + annotations
    2. Evaluate sample using rubrics
    3. Validate evaluation against ground truth
    4. If incorrect, revise rubrics and repeat
    5. Stop when valid or max_epochs reached
    """

    def __init__(
        self,
        model: ChatModelBase,
        grader_mode: GraderMode | str = GraderMode.POINTWISE,
        generate_number: int = 3,
        max_retries: int = 5,
        max_epochs: int = 3,
        min_score: int = 0,
        max_score: int = 4,
        language: LanguageEnum | str = LanguageEnum.ZH,
        parser: Optional[EvalCaseParser] = None,
    ):
        """
        Initialize generator

        Args:
            model: Language model for generation and evaluation
            grader_mode: GraderMode enum or string ("pointwise" or "listwise")
            generate_number: Number of rubrics to generate
            max_retries: Maximum retry attempts for LLM calls
            max_epochs: Maximum iteration epochs for improvement
            min_score: Minimum score for pointwise
            max_score: Maximum score for pointwise
            language: LanguageEnum or string ("zh" or "en")
            parser: Optional EvalCaseParser for field transformation
        """
        self.model = model

        if isinstance(grader_mode, str):
            self.grader_mode = GraderMode(grader_mode)
        else:
            self.grader_mode = grader_mode

        if isinstance(language, str):
            self.language = (
                LanguageEnum(language)
                if language in [item.value for item in LanguageEnum]
                else LanguageEnum.ZH
            )
        else:
            self.language = language

        self.generate_number = generate_number
        self.max_retries = max_retries
        self.max_epochs = max_epochs
        self.min_score = min_score
        self.max_score = max_score
        self.parser = parser

        # Initialize templates
        self._init_templates()

        logger.info(
            f"QuerySpecificRubricGenerator initialized: mode={grader_mode}, language={language}",
        )

    def _init_templates(self):
        """Initialize ChatTemplate objects"""
        # Generation templates
        if self.grader_mode == "pointwise":
            self.generation_template = RubricPromptTemplates.pointwise_generation(
                self.model,
            )
            self.evaluation_template = RubricPromptTemplates.pointwise_evaluation(
                self.model,
            )
            self.revision_template = RubricPromptTemplates.pointwise_revision(
                self.model,
            )
        else:  # listwise (includes former pairwise)
            self.generation_template = RubricPromptTemplates.listwise_generation(
                self.model,
            )
            self.evaluation_template = RubricPromptTemplates.listwise_evaluation(
                self.model,
            )
            self.revision_template = RubricPromptTemplates.listwise_revision(
                self.model,
            )

    async def generate_iterative(self, sample: EvalCase) -> Dict[str, Any]:
        """
        Complete iterative generation and improvement for a single sample

        Returns:
            Dict with:
            - rubrics: List[str]
            - rubric_valid: "True" or "False"
            - rubric_epoch: str (convergence epoch)
            - evaluation_result: Dict
        """
        # Apply parser transformation
        mapped_sample = self._apply_parser(sample)

        # Initial generation
        rubrics = await self.generate(mapped_sample)
        if not rubrics:
            return {
                "rubrics": [],
                "rubric_valid": "False",
                "rubric_epoch": "0",
                "evaluation_result": {},
            }

        # Iterative improvement
        for epoch in range(self.max_epochs):
            # Evaluate current rubrics
            evaluation_result = await self.aevaluate(mapped_sample, rubrics)

            # Validate
            is_correct = self.validate(mapped_sample, evaluation_result)
            logger.debug(f"Epoch {epoch}: correctness = {is_correct}")

            if is_correct:
                return {
                    "rubrics": rubrics,
                    "rubric_valid": "True",
                    "rubric_epoch": str(epoch),
                    "evaluation_result": evaluation_result,
                }

            # Failed, try to revise
            feedback = self.generate_feedback(mapped_sample, evaluation_result)
            revised_rubrics = await self.revise(
                mapped_sample,
                rubrics,
                feedback,
            )

            if not revised_rubrics:
                break

            rubrics = revised_rubrics

        # Max epochs reached, still not converged
        return {
            "rubrics": rubrics,
            "rubric_valid": "False",
            "rubric_epoch": str(self.max_epochs),
            "evaluation_result": evaluation_result,
        }

    async def generate(self, sample: EvalCase) -> List[str]:
        """Generate rubrics for a single sample using ChatTemplate"""
        sample_content = self._format_sample_context(sample)

        # @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def generate_rubrics():
            # Prepare parameters for generation
            if self.grader_mode == "pointwise":
                params = {
                    "language": self.language,
                    "sample_content": sample_content,
                    "generate_number": self.generate_number,
                    "min_score": self.min_score,
                    "max_score": self.max_score,
                }
            else:  # listwise
                params = {
                    "language": self.language,
                    "sample_content": sample_content,
                    "generate_number": self.generate_number,
                }

            # Use ChatTemplate with structured output
            response = await self.generation_template(
                callback=RubricGenerationOutput,
                **params,
            )

            # Get structured data from metadata
            if not response.metadata or "rubrics" not in response.metadata:
                raise ValueError("No rubrics in structured response")

            rubrics = response.metadata["rubrics"]

            if not rubrics or len(rubrics) == 0:
                raise ValueError("No rubrics generated")

            return rubrics

        try:
            rubrics = await generate_rubrics()
            logger.debug(f"Generated {len(rubrics)} rubrics")
            return rubrics
        except Exception as e:
            logger.error(
                f"Failed to generate rubrics after {self.max_retries} attempts: {str(e)}",
            )
        return []

    async def aevaluate(
        self,
        sample: EvalCase,
        rubrics: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate sample using rubrics

        Returns:
            - pointwise: {"scores": [int, ...]}
            - listwise: {"rank_values": [int, ...]}
        """
        if self.grader_mode == "pointwise":
            return await self._evaluate_pointwise(sample, rubrics)
        else:  # listwise
            return await self._evaluate_listwise(sample, rubrics)

    def validate(
        self,
        sample: EvalCase,
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """Validate evaluation result against ground truth"""
        if self.grader_mode == "pointwise":
            return self._validate_pointwise(sample, evaluation_result)
        else:  # listwise
            return self._validate_listwise(sample, evaluation_result)

    async def revise(
        self,
        sample: EvalCase,
        rubrics: List[str],
        feedback: str,
    ) -> List[str]:
        """Revise rubrics based on feedback using ChatTemplate"""
        sample_content = self._format_sample_context(sample)
        rubrics_text = self._format_rubrics_text(rubrics)

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def revise_rubrics():
            # Prepare parameters for revision
            params = {
                "language": self.language,
                "sample_content": sample_content,
                "rubrics": rubrics_text,
                "feedback": feedback,
                "generate_number": self.generate_number,
            }

            response = await self.revision_template(
                callback=RubricGenerationOutput,
                **params,
            )

            # Get structured data from metadata
            if not response.metadata or "rubrics" not in response.metadata:
                raise ValueError("No rubrics in structured response")

            revised_rubrics = response.metadata["rubrics"]

            if not revised_rubrics or len(revised_rubrics) == 0:
                raise ValueError("No revised rubrics generated")

            return revised_rubrics

        try:
            revised_rubrics = await revise_rubrics()
            logger.debug(f"Revised {len(revised_rubrics)} rubrics")
            return revised_rubrics
        except Exception as e:
            logger.error(
                f"Failed to revise rubrics after {self.max_retries} attempts: {str(e)}",
            )
        return []

    def generate_feedback(
        self,
        sample: EvalCase,
        evaluation_result: Dict[str, Any],
    ) -> str:
        """Generate simple feedback for revision"""
        if self.grader_mode == "pointwise":
            return self._generate_pointwise_feedback(sample, evaluation_result)
        else:  # listwise
            return self._generate_listwise_feedback(sample, evaluation_result)

    # ========== Evaluation Methods ==========

    async def _evaluate_pointwise(
        self,
        sample: EvalCase,
        rubrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate in pointwise mode using ChatTemplate"""
        scores = []
        rubrics_text = self._format_rubrics_text(rubrics)
        query = self._get_query_from_sample(sample)

        for item in sample.outputs:
            item_query = query or item.get("query", "")
            answer = item.get("answer", "")

            try:
                # Prepare parameters for pointwise evaluation
                params = {
                    "language": self.language,
                    "rubrics": rubrics_text,
                    "query": item_query,
                    "answer": answer,
                    "min_score": self.min_score,
                    "max_score": self.max_score,
                }

                # Use ChatTemplate with structured output
                response_obj = await self.evaluation_template(
                    callback=GraderScore,
                    **params,
                )
                logger.debug(f"Pointwise evaluation response: {response_obj}")
                # Get structured data from metadata
                if response_obj.metadata and "score" in response_obj.metadata:
                    score = response_obj.metadata["score"]
                    score = max(self.min_score, min(self.max_score, score))
                else:
                    score = self.min_score
                scores.append(score)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                scores.append(self.min_score)

        return {"scores": scores}

    async def _evaluate_listwise(
        self,
        sample: EvalCase,
        rubrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate in listwise mode - model gives complete ranking at once"""
        rubrics_text = self._format_rubrics_text(rubrics)
        query = self._get_query_from_sample(sample)
        responses_text = self._format_responses_for_listwise(sample)

        try:
            # Prepare parameters for listwise evaluation
            params = {
                "language": self.language,
                "rubrics": rubrics_text,
                "query": query,
                "answer": responses_text,
                "num_responses": len(sample.outputs),
            }

            # Use ChatTemplate with structured output
            response_obj = await self.evaluation_template(
                callback=GraderRank,
                **params,
            )
            logger.debug(f"Listwise evaluation response: {response_obj}")
            # Get structured data from metadata
            if response_obj.metadata and "rank" in response_obj.metadata:
                rank_values = response_obj.metadata["rank"]

                # Validate rank values
                if len(rank_values) == len(sample.outputs):
                    if len(set(rank_values)) != len(rank_values):
                        logger.warning(
                            f"Duplicate rank values detected (ties not allowed): {rank_values}",
                        )
                    return {"rank_values": rank_values}
                else:
                    logger.warning(
                        f"Invalid rank values from structured output: {rank_values}",
                    )

        except Exception as e:
            logger.error(f"Listwise evaluation failed: {e}")

    # ========== Validation Methods ==========

    def _validate_pointwise(
        self,
        sample: EvalCase,
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """strict score match for each sample"""
        scores = evaluation_result.get("scores", [])
        if not scores or len(scores) != len(sample.outputs):
            return False

        # Check against expected scores
        has_score_annotations = any(
            item.get("score") is not None for item in sample.outputs
        )
        if has_score_annotations:
            for i, item in enumerate(sample.outputs):
                expected_score = item.get("score")
                if expected_score is not None:
                    actual_score = scores[i]
                    if actual_score != expected_score:
                        return False
            return True

        return False

    def _validate_listwise(
        self,
        sample: EvalCase,
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """Validate listwise results - supports rank value comparison by relative order"""
        rank_values = evaluation_result.get("rank_values", [])

        # Check if we have rank annotations
        has_rank_annotations = any(
            item.get("rank") is not None for item in sample.outputs
        )
        if has_rank_annotations:
            if not rank_values:
                return False

            # Extract expected rank values from ground truth
            expected_ranks = []
            for item in sample.outputs:
                rank_value = item.get("rank")
                if rank_value is not None:
                    expected_ranks.append(rank_value)
                else:
                    return False  # All samples must have rank annotations

            if len(rank_values) == len(expected_ranks):
                expected_order = self._get_relative_order(expected_ranks)
                logger.debug(f"Expected order: {expected_order}")
                predicted_order = self._get_relative_order(rank_values)
                logger.debug(f"Predicted order: {predicted_order}")
                # Must match relative order exactly
                return expected_order == predicted_order

        return False

    def _get_relative_order(self, values: List[float]) -> List[int]:
        """Convert values to relative order rankings (0-based indices sorted by value desc)"""
        # Create (index, value) pairs
        indexed_values = [(i, val) for i, val in enumerate(values)]
        # Sort by value in descending order (higher value = better)
        indexed_values.sort(key=lambda x: x[1], reverse=True)
        # Return the indices in order from best to worst
        return [idx for idx, _ in indexed_values]

    # ========== Feedback Generation ==========

    def _generate_pointwise_feedback(
        self,
        sample: EvalCase,
        evaluation_result: Dict[str, Any],
    ) -> str:
        """Generate simple pointwise feedback"""
        # Extract expected scores
        expected_scores = []
        for item in sample.outputs:
            score = item.get("score")
            if score is not None:
                expected_scores.append(score)

        # Get actual scores
        actual_scores = evaluation_result.get("scores", [])

        # Format as simple comparison
        expected_scores_str = str(expected_scores)
        actual_scores_str = str(actual_scores)

        return f"Expected scores: {expected_scores_str}\nActual scores: {actual_scores_str}"

    def _generate_listwise_feedback(
        self,
        sample: EvalCase,
        evaluation_result: Dict[str, Any],
    ) -> str:
        """Generate simple listwise feedback"""
        # Extract expected rank values
        expected_ranks = []
        for item in sample.outputs:
            rank = item.get("rank")
            if rank is not None:
                expected_ranks.append(rank)

        # Get actual rank values
        actual_ranks = evaluation_result.get("rank_values", [])

        # Format as simple comparison
        expected_ranks_str = str(expected_ranks)
        actual_ranks_str = str(actual_ranks)

        return f"Expected ranks: {expected_ranks_str}\nActual ranks: {actual_ranks_str}"

    def _format_sample_context(self, sample: EvalCase) -> str:
        """Format sample context for reference - language-neutral data formatting"""
        lines = []

        # Extract query
        query = ""
        if hasattr(sample, "data") and isinstance(sample.input, dict):
            query = sample.input.get("query", "")

        if hasattr(sample, "samples") and sample.outputs:
            for i, item in enumerate(sample.outputs):
                item_query = query or item.get("query", "")
                response = item.get("answer", "")

                lines.append(f"Sample {i+1}:")
                if item_query:
                    lines.append(f"Query: {item_query}")
                lines.append(f"Response: {response}")

                # Add ground truth annotation
                score = item.get("score")
                rank = item.get("rank")

                if score is not None:
                    lines.append(f"Expected score: {score}")
                elif rank is not None:
                    lines.append(f"Expected rank: {rank}")

                lines.append("")  # Add blank line between samples

        return "\n".join(lines)

    # ========== Data Processing Methods ==========

    def _apply_parser(self, sample: EvalCase) -> EvalCase:
        """Apply parser transformation if provided, otherwise return original sample"""
        if self.parser is not None:
            return self.parser(sample)
        return sample

    # ========== Utility Methods ==========

    def _get_query_from_sample(self, sample: EvalCase) -> str:
        """Extract query from sample data or first sample item"""
        data_query = ""
        if isinstance(sample.input, dict):
            data_query = sample.input.get("query", "")

        # Fallback to first sample's query if no data-level query
        if not data_query and sample.outputs:
            data_query = sample.outputs[0].get("query", "")

        return data_query

    def _format_rubrics_text(self, rubrics: List[str]) -> str:
        """Format rubrics list into numbered text"""
        return "\n".join(
            [f"{i+1}. {rubric}" for i, rubric in enumerate(rubrics)],
        )

    def _format_responses_for_listwise(self, sample: EvalCase) -> str:
        """Format all responses for listwise evaluation"""
        responses = []
        for i, item in enumerate(sample.outputs):
            responses.append(f"Response {i+1}: {item.get('answer', '')}")
        return "\n\n".join(responses)
