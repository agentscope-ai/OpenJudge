import re

from examples.train.pairwise.template import PairwiseComparisonTemplate


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
    Compute score function compatible with naive.py for pairwise evaluation
    """
    try:
        # Extract predicted preference from model response
        predicted_preference = extract_preference_from_response(solution_str)

        # Extract true preference from extra_info metadata
        true_preference = "tie"
        preference_strength = 0

        if extra_info and isinstance(extra_info, dict):
            metadata = extra_info.get("metadata", {})

            # Handle case where metadata might be a JSON string (from parquet files)
            if isinstance(metadata, str):
                try:
                    import json

                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, ValueError):
                    metadata = {}

            if isinstance(metadata, dict):
                true_preference = metadata.get("preferred", "tie")
                preference_strength = metadata.get("preference_strength", 0)

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
            "accuracy": accuracy,
            "task_type": "pairwise",
            "data_source": data_source,
        }

    except Exception as e:
        return {
            "score": 0.0,
            "predicted_preference": "unknown",
            "true_preference": "tie",
            "accuracy": 0.0,
            "error": str(e),
            "task_type": "pairwise",
            "data_source": data_source,
        }
