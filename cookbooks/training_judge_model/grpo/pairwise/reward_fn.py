import re


def filter_thinking_parts(text):
    """
    Filter thinking parts from text (for models like Qwen3 that support thinking mode).

    Supported thinking tag formats:
    - <think>...</think>
    """
    if not isinstance(text, str):
        return text

    # Define regex patterns for thinking parts
    thinking_patterns = [r"<think>.*?</think>"]

    # Apply all patterns sequentially for filtering
    filtered_text = text
    for pattern in thinking_patterns:
        filtered_text = re.sub(pattern, "", filtered_text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace
    filtered_text = re.sub(r"\n\s*\n", "\n\n", filtered_text)  # Merge multiple newlines
    filtered_text = filtered_text.strip()

    return filtered_text


def extract_preference_response(response_text):
    """
    Extract preference from model response.
    Extract preference choice from <better> tag.
    """
    # Handle case where response_text might not be a string
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # Extract preference from <better> tag
    preference_pattern = r"<better>(.*?)</better>"
    match = re.search(preference_pattern, response_text, re.DOTALL)

    if match:
        preference_content = match.group(1).strip().upper()

        # First check if it's directly A or B
        if preference_content == "A":
            return "A"
        elif preference_content == "B":
            return "B"
        elif preference_content == "TIE":
            return "tie"

        # Then check if it contains specific words but not both
        if "A" in preference_content and "B" not in preference_content:
            return "A"
        elif "B" in preference_content and "A" not in preference_content:
            return "B"
        elif "TIE" in preference_content or ("A" in preference_content and "B" in preference_content):
            return "tie"

    # If no tag found, try to extract from the last part of text
    lines = response_text.strip().split("\n")
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip().upper()
        if line == "A" or "RESPONSE A" in line or "ANSWER A" in line:
            return "A"
        elif line == "B" or "RESPONSE B" in line or "ANSWER B" in line:
            return "B"
        elif "TIE" in line or "EQUAL" in line:
            return "tie"

    return "unknown"  # Return unknown if extraction fails


def calculate_pairwise_reward(predicted_preference, true_preference, response_id):
    """
    Calculate reward based on how well the predicted preference matches the true preference.

    Args:
        predicted_preference: Model's predicted preference ('A', 'B', 'tie', 'unknown')
        true_preference: Ground truth preference ('A', 'B', 'tie')
        response_id: Current response ID ('A' or 'B')

    Returns:
        float: Reward score (1.0 if prediction is correct, 0.0 if incorrect)
    """
    if true_preference is None or predicted_preference == "unknown":
        return 0.0

    # Simplified reward logic: 1 point for correct prediction, 0 for incorrect
    if predicted_preference == true_preference:
        return 1.0
    else:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    compute_score function compatible with naive.py, handles pairwise comparison tasks.

    Args:
        data_source: Data source type
        solution_str: Model generated response
        ground_truth: Ground truth label (contains preference information)
        extra_info: Additional information
    """
    try:
        # First filter out thinking parts (support thinking mode for models like Qwen3)
        filtered_solution = filter_thinking_parts(solution_str)

        # Extract preference from filtered solution_str
        predicted_preference = extract_preference_response(filtered_solution)

        # Handle ground_truth - should contain preference information
        if isinstance(ground_truth, dict):
            true_preference = ground_truth.get("preference", "tie")
            response_id = ground_truth.get("response_id", "A")
            preference_strength = ground_truth.get("preference_strength", 0)
            task_type = ground_truth.get("task_type", "pairwise")
        else:
            # Fallback handling
            if extra_info and isinstance(extra_info, dict):
                # Try to get preference info from extra_info
                data_mode = extra_info.get("data_mode", "pointwise")
                if data_mode == "pairwise":
                    # Analyze original data
                    output_data = extra_info.get("output", [])
                    if output_data and len(output_data) >= 2:
                        # Infer preference from original labels
                        label_a = output_data[0].get("answer", {}).get("label", {})
                        label_b = output_data[1].get("answer", {}).get("label", {})

                        pref_a = label_a.get("overall_preference", 0)
                        pref_b = label_b.get("overall_preference", 0)

                        if pref_a > pref_b:
                            true_preference = "A"
                        elif pref_b > pref_a:
                            true_preference = "B"
                        else:
                            true_preference = "tie"

                        # Assume we're evaluating the first response (A)
                        response_id = "A"
                        preference_strength = abs(pref_a - pref_b)
                        task_type = "pairwise"
                    else:
                        true_preference = "tie"
                        response_id = "A"
                        preference_strength = 0
                        task_type = "pairwise"
                else:
                    # Not a pairwise task, return default values
                    return {"score": 0.0, "error": "Not a pairwise task", "data_source": data_source}
            else:
                true_preference = "tie"
                response_id = "A"
                preference_strength = 0
                task_type = "pairwise"

        # Calculate reward
        reward = calculate_pairwise_reward(predicted_preference, true_preference, response_id)

        # Calculate accuracy
        accuracy = 1.0 if (predicted_preference == true_preference and predicted_preference != "unknown") else 0.0

        # Return detailed information
        return {
            "score": reward,
            "predicted_preference": predicted_preference,
            "accuracy": accuracy,
            "true_preference": true_preference,
            "response_id": response_id,
            "preference_strength": preference_strength,
            "task_type": task_type,
            "data_source": data_source,
        }

    except Exception as e:
        print(f"Error in compute_score: {e}")
        # Return default values
        return {"score": 0.0, "accuracy": 0.0, "error": str(e), "data_source": data_source}


if __name__ == "__main__":
    # Test cases - simulate model's actual output
    model_response = """<think>Let me analyze both responses based on the given principles:

1. Helpfulness: Response A provides detailed step-by-step instructions including washing, peeling, cutting, soaking, and drying. Response B only mentions cutting and frying, missing crucial preparation steps.

2. Accuracy: Response A is factually correct about the soaking process to remove starch. Response B, while not incorrect, lacks important details.

3. Clarity: Response A is clear and well-structured. Response B is clear but overly brief.

4. Completeness: Response A covers all necessary preparation steps. Response B is incomplete, missing several important steps.

5. Relevance: Both responses are relevant, but Response A is more comprehensive in addressing the question.

Response A is significantly better as it provides complete, accurate, and helpful instructions for preparing potatoes for frying.
</think>
<better>A</better>"""

    # Test better tag extraction
    extracted_pref = extract_preference_response(model_response)
    print(f"Extracted preference: {extracted_pref}")

    # Simulate ground_truth data
    ground_truth = {"preference": "A", "preference_strength": 2, "response_id": "A", "task_type": "pairwise"}

    # Test reward calculation
    result = compute_score("helpsteer3", model_response, ground_truth)
    print(f"Reward result: {result}")

    # Test different prediction results
    test_cases = [
        ("A", "A", "A"),  # Correct prediction A is better, current is A
        ("A", "A", "B"),  # Correct prediction A is better, current is B
        ("B", "A", "A"),  # Wrong prediction B is better, current is A
        ("tie", "A", "A"),  # Predict tie, true is A better, current is A
    ]

    print("\n=== Testing different prediction results ===")
    for pred, true, resp_id in test_cases:
        test_gt = {"preference": true, "preference_strength": 1, "response_id": resp_id, "task_type": "pairwise"}
        reward = calculate_pairwise_reward(pred, true, resp_id)
        print(f"Predicted: {pred}, True: {true}, Response ID: {resp_id} -> Reward: {reward:.1f}")
