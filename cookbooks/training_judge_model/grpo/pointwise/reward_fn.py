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


def extract_helpfulness_score(response_text):
    """
    Extract helpfulness score from model response.
    Extract score from <score> tag.
    """
    # Handle case where response_text might not be a string
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # Extract score from <score> tag
    score_pattern = r"<score>(.*?)</score>"
    match = re.search(score_pattern, response_text, re.DOTALL)

    if match:
        score_content = match.group(1).strip()
        # Extract numbers from content
        numbers = re.findall(r"\d+", score_content)
        if numbers:
            try:
                score = int(numbers[0])  # Take the first number as score
                if 0 <= score <= 4:  # Assume score range is 0-4
                    return score
            except:
                pass

    return 0  # Default to 0 if extraction fails


def calculate_helpfulness_reward(predicted_score, true_score):
    """
    Calculate reward based on the difference between predicted and true helpfulness scores.
    Smaller difference results in higher reward.

    For binary classification scenarios (true_score is 0 or 1):
    - Correct prediction (exact match) -> Reward 1.0
    - Wrong prediction -> Reward 0.0
    """
    if true_score is None:
        return 0.0

    # Calculate difference
    diff = abs(predicted_score - true_score)

    # For binary classification (0 or 1), use exact match
    if true_score in [0, 1]:
        return 1.0 if diff == 0 else 0.0

    # For multi-class scenarios (0-4), use difference calculation
    # Convert difference to reward score (smaller difference = higher reward)
    max_possible_diff = 4
    normalized_diff = min(diff / max_possible_diff, 1.0)

    # Reward = 1 - normalized difference
    reward = 1.0 - normalized_diff

    return reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    compute_score function compatible with naive.py.

    Args:
        data_source: Data source type
        solution_str: Model generated response
        ground_truth: Ground truth label (obtained from reward_model field)
        extra_info: Additional information
    """
    try:
        # First filter out thinking parts (support thinking mode for models like Qwen3)
        filtered_solution = filter_thinking_parts(solution_str)

        # Extract helpfulness score from filtered solution_str
        predicted_helpfulness = extract_helpfulness_score(filtered_solution)

        # Handle ground_truth - could be a number or dict
        if isinstance(ground_truth, dict):
            true_helpfulness = ground_truth.get("helpfulness", 0)
        elif isinstance(ground_truth, (int, float)):
            true_helpfulness = int(ground_truth)
        elif isinstance(ground_truth, str) and ground_truth.isdigit():
            true_helpfulness = int(ground_truth)
        else:
            # If ground_truth is unavailable, try to get from extra_info
            if extra_info and isinstance(extra_info, dict):
                output_data = extra_info.get("output", [])
                if output_data and len(output_data) > 0:
                    label_data = output_data[0].get("label", {})
                    true_helpfulness = label_data.get("helpfulness", 0)
                else:
                    true_helpfulness = 0
            else:
                true_helpfulness = 0

        # Calculate reward
        reward = calculate_helpfulness_reward(predicted_helpfulness, true_helpfulness)

        # Return detailed information
        return {
            "score": reward,
            "predicted_helpfulness": predicted_helpfulness,
            "true_helpfulness": true_helpfulness,
            "data_source": data_source,
        }

    except Exception as e:
        print(f"Error in compute_score: {e}")
        # Return default values
        return {"score": 0.0, "error": str(e), "data_source": data_source}


if __name__ == "__main__":
    # Test cases
    test_response = """<think>Let me analyze this answer step by step:
1. First, I'll check if the answer is well-structured...
4. Finally, I'll look at the overall helpfulness...
</think>
<score>2</score>"""

    ground_truth = {"helpfulness": 3, "task_type": "pointwise"}

    # Test compute_score function
    result = compute_score(data_source="test", solution_str=test_response, ground_truth=ground_truth)

    print("Test Result:")
    print(f"  Predicted Score: {result.get('predicted_helpfulness')}")
    print(f"  True Score: {result.get('true_helpfulness')}")
    print(f"  Reward: {result.get('score')}")
