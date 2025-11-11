"""
Multimodal G-Eval Metric

Based on the G-Eval framework (https://arxiv.org/pdf/2303.16634.pdf).
Provides flexible evaluation with custom criteria and automatic evaluation step generation.
"""

import asyncio
from typing import Any, List, Optional, Tuple, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.metrics.multimodal.base import BaseMultimodalMetric
from rm_gallery.core.metrics.multimodal.schema import (
    MLLMTestCase,
    MLLMTestCaseParams,
    ReasonScore,
)
from rm_gallery.core.metrics.multimodal.utils import construct_verbose_logs
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.schema import EvaluationSteps, Rubric
from rm_gallery.gallery.rm.multimodal.template import MultimodalGEvalTemplate
from rm_gallery.gallery.rm.multimodal.utils import (
    construct_g_eval_params_string,
    construct_test_case_list,
    format_rubrics,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
)


class MultimodalGEval(BaseMultimodalMetric):
    """
    Multimodal G-Eval metric for flexible evaluation with custom criteria

    This metric implements the G-Eval framework for multimodal evaluation,
    allowing custom evaluation criteria and automatic generation of evaluation steps.

    Key Features:
    - Chain-of-Thought evaluation with step-by-step reasoning
    - Automatic evaluation step generation from criteria
    - Custom rubrics for detailed scoring standards
    - Support for multiple test case parameters
    - Flexible scoring (0-10 scale, normalized to 0-1)

    Attributes:
        evaluation_name: Name for this evaluation
        evaluation_params: List of test case parameters to evaluate
        criteria: Evaluation criteria description
        evaluation_steps: Explicit evaluation steps (optional, auto-generated if not provided)
        rubric: Detailed scoring rubric (optional)
        vlm_api: Qwen VLM API instance for evaluation

    Example:
        >>> from rm_gallery.gallery.rm.multimodal import MultimodalGEval
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMTestCase, MLLMTestCaseParams
        >>>
        >>> # Initialize VLM API
        >>> vlm_api = QwenVLAPI(model_name="qwen-vl-plus")
        >>>
        >>> # Create metric
        >>> metric = MultimodalGEval(
        ...     name="image_caption_quality",
        ...     evaluation_name="Image Caption Quality",
        ...     evaluation_params=[
        ...         MLLMTestCaseParams.INPUT,
        ...         MLLMTestCaseParams.ACTUAL_OUTPUT
        ...     ],
        ...     criteria="Evaluate the quality of image captions based on accuracy and detail",
        ...     vlm_api=vlm_api,
        ...     threshold=0.7
        ... )
        >>>
        >>> # Evaluate
        >>> test_case = MLLMTestCase(
        ...     input=[MLLMImage(url="..."), "Describe this image"],
        ...     actual_output=["A cat sitting on a mat"]
        ... )
        >>> score = metric.measure(test_case)
    """

    name: str = Field(default="multimodal_geval", description="Metric identifier")
    evaluation_name: str = Field(..., description="Name for this evaluation")
    evaluation_params: List[MLLMTestCaseParams] = Field(
        ..., description="Test case parameters to evaluate"
    )
    criteria: Optional[str] = Field(None, description="Evaluation criteria description")
    evaluation_steps: Optional[List[str]] = Field(
        None, description="Explicit evaluation steps (auto-generated if not provided)"
    )
    rubric: Optional[List[Rubric]] = Field(None, description="Detailed scoring rubric")
    vlm_api: QwenVLAPI = Field(..., description="Qwen VLM API instance")
    score_range: Tuple[int, int] = Field(
        default=(0, 10), description="Score range (min, max)"
    )

    # Internal state
    _generated_steps: Optional[List[str]] = None

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization"""
        super().model_post_init(__context)

        validate_criteria_and_evaluation_steps(self.criteria, self.evaluation_steps)

        if self.rubric:
            self.rubric = validate_and_sort_rubrics(self.rubric)

        self.evaluation_model = self.vlm_api.get_model_name()

        if self.strict_mode and self.rubric:
            score_values = [r.score for r in self.rubric]
            if not (len(score_values) == 2 and 0 in score_values and 1 in score_values):
                raise ValueError(
                    "In strict_mode, rubric must have exactly 2 entries with scores 0 and 1"
                )

    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _additional_context: Optional[str] = None,
    ) -> float:
        """
        Measure the metric score for a test case

        Args:
            test_case: Multimodal test case to evaluate
            _show_indicator: Whether to show progress indicator
            _in_component: Whether this is called within another component
            _additional_context: Additional context for evaluation

        Returns:
            float: Normalized score [0, 1]
        """
        self.check_test_case_params(test_case, self.evaluation_params)

        self.evaluation_cost = 0.0

        if self.async_mode:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.a_measure(
                    test_case,
                    _show_indicator=_show_indicator,
                    _in_component=_in_component,
                    _additional_context=_additional_context,
                )
            )

        # Generate evaluation steps if not provided
        if self._generated_steps is None:
            self._generated_steps = self._generate_evaluation_steps()

        # Evaluate
        g_score, reason = self._evaluate(test_case, _additional_context)

        # Store results
        self.reason = reason
        self.score = self._normalize_score(g_score)

        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.success = self.is_successful()

        # Generate verbose logs
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Criteria: {self.criteria}",
                f"Evaluation Steps: {self._generated_steps}",
                f"Rubric: {format_rubrics(self.rubric) if self.rubric else 'None'}",
                f"Score: {self.score:.4f} (raw: {g_score}/{self.score_range[1]})",
                f"Reason: {self.reason}",
            ],
        )

        return self.score

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _additional_context: Optional[str] = None,
    ) -> float:
        """
        Async version of measure

        Args:
            test_case: Multimodal test case to evaluate
            _show_indicator: Whether to show progress indicator
            _in_component: Whether this is called within another component
            _additional_context: Additional context for evaluation

        Returns:
            float: Normalized score [0, 1]
        """
        self.check_test_case_params(test_case, self.evaluation_params)

        self.evaluation_cost = 0.0

        # Generate evaluation steps if not provided
        if self._generated_steps is None:
            self._generated_steps = await self._a_generate_evaluation_steps()

        # Evaluate
        g_score, reason = await self._a_evaluate(test_case, _additional_context)

        # Store results
        self.reason = reason
        self.score = self._normalize_score(g_score)

        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.success = self.is_successful()

        # Generate verbose logs
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Criteria: {self.criteria}",
                f"Evaluation Steps: {self._generated_steps}",
                f"Rubric: {format_rubrics(self.rubric) if self.rubric else 'None'}",
                f"Score: {self.score:.4f} (raw: {g_score}/{self.score_range[1]})",
                f"Reason: {self.reason}",
            ],
        )

        return self.score

    def _normalize_score(self, raw_score: Union[int, float]) -> float:
        """
        Normalize raw score to [0, 1] range

        Args:
            raw_score: Raw score from evaluation

        Returns:
            float: Normalized score [0, 1]
        """
        if self.strict_mode:
            return float(raw_score)

        min_score, max_score = self.score_range
        normalized = (raw_score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))

    def _generate_evaluation_steps(self) -> List[str]:
        """
        Generate evaluation steps from criteria (synchronous)

        Returns:
            List[str]: Generated evaluation steps
        """
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(self.evaluation_params)
        prompt = MultimodalGEvalTemplate.generate_evaluation_steps(
            parameters=g_eval_params_str, criteria=self.criteria
        )

        try:
            response = self.vlm_api.generate(
                text=prompt, images=[], response_format=EvaluationSteps
            )

            if isinstance(response, dict) and "steps" in response:
                return response["steps"]
            elif hasattr(response, "steps"):
                return response.steps
            else:
                logger.warning(
                    f"Unexpected response format from VLM: {response}, "
                    f"using default evaluation steps"
                )
                return [
                    f"Evaluate the {construct_g_eval_params_string(self.evaluation_params)}"
                ]

        except Exception as e:
            logger.error(f"Error generating evaluation steps: {e}")
            return [
                f"Evaluate the {construct_g_eval_params_string(self.evaluation_params)}"
            ]

    async def _a_generate_evaluation_steps(self) -> List[str]:
        """
        Generate evaluation steps from criteria (asynchronous)

        Returns:
            List[str]: Generated evaluation steps
        """
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(self.evaluation_params)
        prompt = MultimodalGEvalTemplate.generate_evaluation_steps(
            parameters=g_eval_params_str, criteria=self.criteria
        )

        try:
            response = await self.vlm_api.a_generate(
                text=prompt, images=[], response_format=EvaluationSteps
            )

            if isinstance(response, dict) and "steps" in response:
                return response["steps"]
            elif hasattr(response, "steps"):
                return response.steps
            else:
                logger.warning(
                    f"Unexpected response format from VLM: {response}, "
                    f"using default evaluation steps"
                )
                return [
                    f"Evaluate the {construct_g_eval_params_string(self.evaluation_params)}"
                ]

        except Exception as e:
            logger.error(f"Error generating evaluation steps: {e}")
            return [
                f"Evaluate the {construct_g_eval_params_string(self.evaluation_params)}"
            ]

    def _evaluate(
        self,
        test_case: MLLMTestCase,
        _additional_context: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Evaluate test case (synchronous)

        Args:
            test_case: Test case to evaluate
            _additional_context: Additional context

        Returns:
            Tuple of (score, reason)
        """
        g_eval_params_str = construct_g_eval_params_string(self.evaluation_params)
        test_case_list = construct_test_case_list(test_case, self.evaluation_params)

        evaluation_steps_str = "\n".join(
            f"{i+1}. {step}"
            for i, step in enumerate(self._generated_steps or self.evaluation_steps)
        )

        if self.strict_mode:
            prompt_parts = MultimodalGEvalTemplate.generate_strict_evaluation_results(
                evaluation_steps=evaluation_steps_str,
                test_case_list=test_case_list,
                parameters=g_eval_params_str,
                _additional_context=_additional_context,
            )
        else:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt_parts = MultimodalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=evaluation_steps_str,
                test_case_list=test_case_list,
                parameters=g_eval_params_str,
                rubric=rubric_str,
                score_range=self.score_range,
                _additional_context=_additional_context,
            )

        try:
            response = self.vlm_api.generate_from_parts(
                parts=prompt_parts, response_format=ReasonScore
            )

            if isinstance(response, dict):
                score = response.get("score", 0)
                reason = response.get(
                    "reasoning", response.get("reason", "No reason provided")
                )
            elif hasattr(response, "score"):
                score = response.score
                reason = (
                    response.reasoning
                    if hasattr(response, "reasoning")
                    else "No reason provided"
                )
            else:
                logger.warning(f"Unexpected response format: {response}")
                score = 0
                reason = "Error: Invalid response format"

            return float(score), str(reason)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return 0.0, f"Error: {str(e)}"

    async def _a_evaluate(
        self,
        test_case: MLLMTestCase,
        _additional_context: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Evaluate test case (asynchronous)

        Args:
            test_case: Test case to evaluate
            _additional_context: Additional context

        Returns:
            Tuple of (score, reason)
        """
        g_eval_params_str = construct_g_eval_params_string(self.evaluation_params)
        test_case_list = construct_test_case_list(test_case, self.evaluation_params)

        evaluation_steps_str = "\n".join(
            f"{i+1}. {step}"
            for i, step in enumerate(self._generated_steps or self.evaluation_steps)
        )

        if self.strict_mode:
            prompt_parts = MultimodalGEvalTemplate.generate_strict_evaluation_results(
                evaluation_steps=evaluation_steps_str,
                test_case_list=test_case_list,
                parameters=g_eval_params_str,
                _additional_context=_additional_context,
            )
        else:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt_parts = MultimodalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=evaluation_steps_str,
                test_case_list=test_case_list,
                parameters=g_eval_params_str,
                rubric=rubric_str,
                score_range=self.score_range,
                _additional_context=_additional_context,
            )

        try:
            response = await self.vlm_api.a_generate_from_parts(
                parts=prompt_parts, response_format=ReasonScore
            )

            if isinstance(response, dict):
                score = response.get("score", 0)
                reason = response.get(
                    "reasoning", response.get("reason", "No reason provided")
                )
            elif hasattr(response, "score"):
                score = response.score
                reason = (
                    response.reasoning
                    if hasattr(response, "reasoning")
                    else "No reason provided"
                )
            else:
                logger.warning(f"Unexpected response format: {response}")
                score = 0
                reason = "Error: Invalid response format"

            return float(score), str(reason)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return 0.0, f"Error: {str(e)}"
