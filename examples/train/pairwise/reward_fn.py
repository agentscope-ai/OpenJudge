import math
import re

from examples.train.pairwise.template import (
    PairwiseComparisonTemplate,
    PointwiseEvaluationTemplate,
)


def calculate_evaluation_reward(predicted_score, true_score):
    """
    Calculate reward based on relative error between predicted and true scores.
    Uses exponential decay function - more tolerant for small errors, more severe for large errors.
    """
    if true_score is None:
        return 0.0

    # calculate relative error (consider the case of denominator is 0)
    if true_score == 0:
        abs_error = abs(predicted_score)
        max_possible_error = 4  # max score
    else:
        abs_error = abs(predicted_score - true_score)
        max_possible_error = 4  # score range 0-4

    # use exponential decay function
    # reward = exp(-k * error_ratio), where k is the decay parameter
    k = 2.0  # decay parameter, can be adjusted
    error_ratio = abs_error / max_possible_error
    reward = math.exp(-k * error_ratio)

    return float(reward)


def extract_preference_from_response(response_text):
    """
    Extract preference from model response for pairwise comparison
    """
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # First try to parse using the template
    try:
        parsed_result = PairwiseComparisonTemplate.parse(response_text)
        return parsed_result.preference
    except:
        pass

    # Fallback: extract from <preference> tags
    preference_pattern = r"<preference>(.*?)</preference>"
    match = re.search(preference_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if match:
        preference_content = match.group(1).strip().upper()

        # Normalize preference values
        if preference_content == "A" or "RESPONSE A" in preference_content:
            return "A"
        elif preference_content == "B" or "RESPONSE B" in preference_content:
            return "B"
        elif preference_content == "TIE" or "EQUAL" in preference_content:
            return "tie"

    # Final fallback: check text content
    lines = response_text.strip().split("\n")
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip().upper()
        if line == "A" or "RESPONSE A" in line:
            return "A"
        elif line == "B" or "RESPONSE B" in line:
            return "B"
        elif "TIE" in line or "EQUAL" in line:
            return "tie"

    return "unknown"


def calculate_pairwise_reward(predicted_preference, true_preference):
    """
    Calculate reward for pairwise comparison
    """
    if true_preference is None or predicted_preference == "unknown":
        return 0.0

    # Simple reward: 1.0 for correct prediction, 0.0 for incorrect
    if predicted_preference == true_preference:
        return 1.0
    else:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Compute score function compatible with naive.py
    Supports both pointwise and pairwise evaluation for HelpSteer3 dataset
    """
    try:
        # Determine task type based on ground_truth structure
        is_pairwise = False
        if (
            isinstance(ground_truth, dict)
            and ground_truth.get("task_type") == "pairwise"
        ):
            is_pairwise = True
        elif extra_info and isinstance(extra_info, dict):
            # Check if extra_info indicates pairwise task
            output_data = extra_info.get("output", [])
            if len(output_data) >= 2:
                is_pairwise = True

        if is_pairwise:
            # Pairwise evaluation
            predicted_preference = extract_preference_from_response(solution_str)

            # Extract true preference from ground_truth
            if isinstance(ground_truth, dict):
                true_preference = ground_truth.get("preference", "tie")
                preference_strength = ground_truth.get("preference_strength", 0)
                response_id = ground_truth.get("response_id", "A")
            else:
                # Fallback: try to infer from extra_info
                if extra_info and isinstance(extra_info, dict):
                    output_data = extra_info.get("output", [])
                    if len(output_data) >= 2:
                        label_a = output_data[0].get("answer", {}).get("label", {})
                        label_b = output_data[1].get("answer", {}).get("label", {})

                        # Try multiple score fields for HelpSteer3 compatibility
                        score_a = (
                            label_a.get("helpfulness", 0)
                            or label_a.get("overall_quality", 0)
                            or label_a.get("score", 0)
                        )
                        score_b = (
                            label_b.get("helpfulness", 0)
                            or label_b.get("overall_quality", 0)
                            or label_b.get("score", 0)
                        )

                        if score_a > score_b:
                            true_preference = "A"
                            preference_strength = score_a - score_b
                        elif score_b > score_a:
                            true_preference = "B"
                            preference_strength = score_b - score_a
                        else:
                            true_preference = "tie"
                            preference_strength = 0

                        response_id = "A"
                    else:
                        true_preference = "tie"
                        preference_strength = 0
                        response_id = "A"
                else:
                    true_preference = "tie"
                    preference_strength = 0
                    response_id = "A"

            # Calculate pairwise reward
            reward = calculate_pairwise_reward(predicted_preference, true_preference)
            accuracy = (
                1.0
                if (
                    predicted_preference == true_preference
                    and predicted_preference != "unknown"
                )
                else 0.0
            )

            return {
                "score": reward,
                "predicted_preference": predicted_preference,
                "true_preference": true_preference,
                "preference_strength": preference_strength,
                "response_id": response_id,
                "accuracy": accuracy,
                "task_type": "pairwise",
                "data_source": data_source,
            }

        else:
            # Pointwise evaluation (original logic)
            parsed_result = PointwiseEvaluationTemplate.parse(solution_str)
            predicted_score = parsed_result.score

            # Validate range
            if not (0 <= predicted_score <= 4):
                predicted_score = 0

            # Process ground_truth
            if isinstance(ground_truth, dict):
                true_score = (
                    ground_truth.get("helpfulness", 0)
                    or ground_truth.get("overall_quality", 0)
                    or ground_truth.get("score", 0)
                )
            elif isinstance(ground_truth, (int, float)):
                true_score = int(ground_truth)
            elif isinstance(ground_truth, str) and ground_truth.isdigit():
                true_score = int(ground_truth)
            else:
                # Try to get from extra_info
                if extra_info and isinstance(extra_info, dict):
                    output_data = extra_info.get("output", [])
                    if output_data and len(output_data) > 0:
                        label_data = output_data[0].get("answer", {}).get("label", {})
                        true_score = (
                            label_data.get("helpfulness", 0)
                            or label_data.get("overall_quality", 0)
                            or label_data.get("score", 0)
                        )
                    else:
                        true_score = 0
                else:
                    true_score = 0

            # Calculate reward
            reward = calculate_evaluation_reward(predicted_score, true_score)
            accuracy = 1 if predicted_score == true_score else 0

            return {
                "score": reward,
                "predicted_score": predicted_score,
                "true_score": true_score,
                "accuracy": accuracy,
                "task_type": "pointwise",
                "data_source": data_source,
            }

    except Exception as e:
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "error": str(e),
            "data_source": data_source,
        }
