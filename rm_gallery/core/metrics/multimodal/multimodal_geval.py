"""
Multimodal G-Eval Grader

Based on the G-Eval framework for flexible evaluation with custom criteria.
Restructured to work with Grader framework.
"""

from typing import List, Optional, Tuple

from loguru import logger

from rm_gallery.core.grader import Grader, GraderMode, GraderScore
from rm_gallery.core.metrics.multimodal.schema import (
    MLLMImage,
    MLLMTestCaseParams,
    ReasonScore,
)
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.schema import EvaluationSteps, Rubric
from rm_gallery.gallery.rm.multimodal.template import MultimodalGEvalTemplate
from rm_gallery.gallery.rm.multimodal.utils import (
    construct_g_eval_params_string,
    format_rubrics,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
)


class MultimodalGEvalGrader(Grader):
    """
    Multimodal G-Eval Grader

    Flexible evaluation with custom criteria using the G-Eval framework.
    Supports:
    - Chain-of-Thought evaluation with step-by-step reasoning
    - Automatic evaluation step generation from criteria
    - Custom rubrics for detailed scoring standards
    - Flexible scoring (0-10 scale, normalized to 0-1)

    Attributes:
        name: Grader name
        model: QwenVLAPI instance for evaluation
        evaluation_name: Name for this evaluation
        evaluation_params: List of parameters to evaluate (e.g., input, actual_output)
        criteria: Evaluation criteria description
        evaluation_steps: Explicit evaluation steps (optional, auto-generated if not provided)
        rubric: Detailed scoring rubric (optional)
        threshold: Success threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMTestCaseParams, MLLMImage
        >>>
        >>> vlm_api = QwenVLAPI(model_name="qwen-vl-plus")
        >>> grader = MultimodalGEvalGrader(
        ...     model=vlm_api,
        ...     evaluation_name="Image Caption Quality",
        ...     evaluation_params=[
        ...         MLLMTestCaseParams.INPUT,
        ...         MLLMTestCaseParams.ACTUAL_OUTPUT
        ...     ],
        ...     criteria="Evaluate the quality of image captions based on accuracy and detail",
        ...     threshold=0.7
        ... )
        >>>
        >>> result = await grader.evaluate(
        ...     input=[MLLMImage(url="..."), "Describe this image"],
        ...     actual_output=["A cat sitting on a mat"]
        ... )
    """

    def __init__(
        self,
        model: QwenVLAPI,
        evaluation_name: str,
        evaluation_params: List[MLLMTestCaseParams],
        name: str = "multimodal_geval",
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        rubric: Optional[List[Rubric]] = None,
        threshold: float = 0.7,
        score_range: Tuple[int, int] = (0, 10),
        description: str = "Multimodal G-Eval flexible evaluation",
    ):
        super().__init__(
            name=name,
            grader_mode=GraderMode.POINTWISE,
            description=description,
        )
        self.model = model
        self.evaluation_name = evaluation_name
        self.evaluation_params = evaluation_params
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.rubric = rubric
        self.threshold = threshold
        self.score_range = score_range
        self.evaluation_cost = 0.0
        self._generated_steps: Optional[List[str]] = None

        # Validate criteria and evaluation steps
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)

        # Validate and sort rubrics
        if rubric:
            self.rubric = validate_and_sort_rubrics(rubric, score_range)

    def _generate_evaluation_steps(self, params_dict: dict) -> List[str]:
        """Generate evaluation steps from criteria"""
        if self._generated_steps is not None:
            return self._generated_steps

        if self.criteria is None:
            raise ValueError("Cannot generate evaluation steps without criteria")

        # Build parameters string for context
        params_str = construct_g_eval_params_string(self.evaluation_params)

        # Generate steps using VLM
        prompt = MultimodalGEvalTemplate.generate_evaluation_steps(
            parameters=params_str, criteria=self.criteria
        )

        try:
            result = self.model.generate(
                text=prompt,
                images=None,
                response_format=EvaluationSteps,
            )

            self._generated_steps = result.steps
            return self._generated_steps

        except Exception as e:
            logger.error(f"Error generating evaluation steps: {e}")
            # Fallback to default steps
            return [
                f"Analyze the {param.value}" for param in self.evaluation_params
            ] + ["Evaluate based on the given criteria"]

    def _evaluate_with_geval(
        self,
        params_dict: dict,
    ) -> Tuple[float, str]:
        """Evaluate using G-Eval framework (synchronous)"""
        # Get or generate evaluation steps
        steps = (
            self.evaluation_steps
            if self.evaluation_steps
            else self._generate_evaluation_steps(params_dict)
        )

        # Format evaluation steps as string
        steps_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

        # Build parameters string
        params_str = construct_g_eval_params_string(self.evaluation_params)

        # Format rubric if provided
        rubric_str = format_rubrics(self.rubric) if self.rubric else None

        # Build test case list (text and images)
        test_case_list = []
        for param in self.evaluation_params:
            param_value = params_dict.get(param.value, [])
            if isinstance(param_value, list):
                test_case_list.extend(param_value)
            else:
                test_case_list.append(param_value)

        # Generate evaluation prompt using template
        prompt_parts = MultimodalGEvalTemplate.generate_evaluation_results(
            evaluation_steps=steps_str,
            test_case_list=test_case_list,
            parameters=params_str,
            rubric=rubric_str,
            score_range=self.score_range,
        )

        try:
            # Extract text and images from prompt parts
            prompt_text = "".join([p for p in prompt_parts if isinstance(p, str)])
            prompt_images = [p for p in prompt_parts if isinstance(p, MLLMImage)]

            result = self.model.generate(
                text=prompt_text,
                images=prompt_images if prompt_images else None,
                response_format=ReasonScore,
            )

            score = (
                float(result.score)
                if not isinstance(result.score, list)
                else float(result.score[0])
            )
            return score, result.reasoning

        except Exception as e:
            logger.error(f"Error in G-Eval evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _a_evaluate_with_geval(
        self,
        params_dict: dict,
    ) -> Tuple[float, str]:
        """Evaluate using G-Eval framework (asynchronous)"""
        # Get or generate evaluation steps
        steps = (
            self.evaluation_steps
            if self.evaluation_steps
            else self._generate_evaluation_steps(params_dict)
        )

        # Format evaluation steps as string
        steps_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

        # Build parameters string
        params_str = construct_g_eval_params_string(self.evaluation_params)

        # Format rubric if provided
        rubric_str = format_rubrics(self.rubric) if self.rubric else None

        # Build test case list (text and images)
        test_case_list = []
        for param in self.evaluation_params:
            param_value = params_dict.get(param.value, [])
            if isinstance(param_value, list):
                test_case_list.extend(param_value)
            else:
                test_case_list.append(param_value)

        # Generate evaluation prompt using template
        prompt_parts = MultimodalGEvalTemplate.generate_evaluation_results(
            evaluation_steps=steps_str,
            test_case_list=test_case_list,
            parameters=params_str,
            rubric=rubric_str,
            score_range=self.score_range,
        )

        try:
            # Extract text and images from prompt parts
            prompt_text = "".join([p for p in prompt_parts if isinstance(p, str)])
            prompt_images = [p for p in prompt_parts if isinstance(p, MLLMImage)]

            result = await self.model.a_generate(
                text=prompt_text,
                images=prompt_images if prompt_images else None,
                response_format=ReasonScore,
            )

            score = (
                float(result.score)
                if not isinstance(result.score, list)
                else float(result.score[0])
            )
            return score, result.reasoning

        except Exception as e:
            logger.error(f"Error in G-Eval evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    def _compute(
        self,
        **params_dict,
    ) -> Tuple[float, dict]:
        """
        Compute G-Eval score (synchronous)

        Args:
            **params_dict: Dictionary containing evaluation parameters
                e.g., input=..., actual_output=..., expected_output=...

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Validate required parameters
        for param in self.evaluation_params:
            if param.value not in params_dict:
                return 0.0, {
                    "error": f"Missing required parameter: {param.value}",
                }

        # Evaluate
        raw_score, reasoning = self._evaluate_with_geval(params_dict)

        # Normalize score to [0, 1]
        score_min, score_max = self.score_range
        normalized_score = (raw_score - score_min) / (score_max - score_min)
        normalized_score = max(0.0, min(1.0, normalized_score))

        details = {
            "raw_score": raw_score,
            "score_range": self.score_range,
            "reasoning": reasoning,
            "evaluation_name": self.evaluation_name,
            "evaluation_params": [p.value for p in self.evaluation_params],
            "evaluation_steps": (self.evaluation_steps or self._generated_steps),
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return normalized_score, details

    async def _a_compute(
        self,
        **params_dict,
    ) -> Tuple[float, dict]:
        """
        Compute G-Eval score (asynchronous)

        Args:
            **params_dict: Dictionary containing evaluation parameters

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Validate required parameters
        for param in self.evaluation_params:
            if param.value not in params_dict:
                return 0.0, {
                    "error": f"Missing required parameter: {param.value}",
                }

        # Evaluate
        raw_score, reasoning = await self._a_evaluate_with_geval(params_dict)

        # Normalize score to [0, 1]
        score_min, score_max = self.score_range
        normalized_score = (raw_score - score_min) / (score_max - score_min)
        normalized_score = max(0.0, min(1.0, normalized_score))

        details = {
            "raw_score": raw_score,
            "score_range": self.score_range,
            "reasoning": reasoning,
            "evaluation_name": self.evaluation_name,
            "evaluation_params": [p.value for p in self.evaluation_params],
            "evaluation_steps": (self.evaluation_steps or self._generated_steps),
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return normalized_score, details

    async def evaluate(
        self,
        async_mode: bool = True,
        **params_dict,
    ) -> GraderScore:
        """
        Evaluate using Multimodal G-Eval framework

        Args:
            async_mode: Whether to use async evaluation
            **params_dict: Dictionary containing evaluation parameters
                Expected keys depend on evaluation_params, e.g.:
                - input: List[Union[str, MLLMImage]]
                - actual_output: List[Union[str, MLLMImage]]
                - expected_output: List[Union[str, MLLMImage]]
                - context: List[Union[str, MLLMImage]]
                etc.

        Returns:
            GraderScore: Score with normalized evaluation value [0, 1]

        Example:
            >>> result = await grader.evaluate(
            ...     input=[MLLMImage(url="..."), "Describe this"],
            ...     actual_output=["A cat sitting"]
            ... )
        """
        if async_mode:
            score, details = await self._a_compute(**params_dict)
        else:
            score, details = self._compute(**params_dict)

        if "error" in details:
            return GraderScore(
                score=0.0,
                reason=details["error"],
                metadata=details,
            )

        reason = f"""{self.evaluation_name}: {score:.4f} (raw: {details['raw_score']:.2f}/{self.score_range[1]})

{details['reasoning']}

Evaluation Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(details['evaluation_steps'] or []))}
"""

        return GraderScore(
            score=score,
            reason=reason.strip(),
            metadata=details,
        )


__all__ = ["MultimodalGEvalGrader"]
