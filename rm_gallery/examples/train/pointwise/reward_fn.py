import re


def extract_helpfulness_score(response_text):
    """
    extract helpfulness score from model response
    directly extract score from <violation> tag
    """
    # Handle case where response_text might not be a string
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # 从<violation>标签中提取分数
    violation_pattern = r"<violation>(.*?)</violation>"
    match = re.search(violation_pattern, response_text, re.DOTALL)

    if match:
        violation_content = match.group(1).strip()
        # 提取其中的数字
        numbers = re.findall(r"\d+", violation_content)
        if numbers:
            try:
                score = int(numbers[0])  # 取第一个数字作为分数
                if 0 <= score <= 4:  # 假设分数范围是0-4
                    return score
            except:
                pass

    return 0  # 如果无法提取，默认为0


def calculate_helpfulness_reward(predicted_score, true_score):
    """
    calculate reward based on the difference between predicted helpfulness score and true helpfulness score
    the smaller the difference, the higher the reward
    """
    if true_score is None or true_score == 0:
        return 0.0

    # calculate difference
    diff = abs(predicted_score - true_score)

    # convert difference to reward score (the smaller the difference, the higher the reward)
    # the maximum possible difference is 4 (0-4 range), so normalize to 0-1
    max_possible_diff = 4
    normalized_diff = min(diff / max_possible_diff, 1.0)

    # reward = 1 - normalized difference
    reward = 1.0 - normalized_diff

    return reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    compute_score function compatible with naive.py
    parameters:
    - data_source: data source type
    - solution_str: model generated response
    - ground_truth: true label (from reward_model field)
    - extra_info: extra information
    """
    try:
        # extract helpfulness score from solution_str
        predicted_helpfulness = extract_helpfulness_score(solution_str)

        # process ground_truth - maybe a number or a dictionary
        if isinstance(ground_truth, dict):
            true_helpfulness = ground_truth.get("helpfulness", 0)
        elif isinstance(ground_truth, (int, float)):
            true_helpfulness = int(ground_truth)
        elif isinstance(ground_truth, str) and ground_truth.isdigit():
            true_helpfulness = int(ground_truth)
        else:
            # if ground_truth is not available, try to get it from extra_info
            if extra_info and isinstance(extra_info, dict):
                output_data = extra_info.get("output", [])
                if output_data and len(output_data) > 0:
                    label_data = output_data[0].get("label", {})
                    true_helpfulness = label_data.get("helpfulness", 0)
                else:
                    true_helpfulness = 0
            else:
                true_helpfulness = 0

        # calculate reward
        reward = calculate_helpfulness_reward(predicted_helpfulness, true_helpfulness)

        # return detailed information
        return {
            "score": reward,
            "predicted_helpfulness": predicted_helpfulness,
            "true_helpfulness": true_helpfulness,
            "data_source": data_source,
        }

    except Exception as e:
        print(f"Error in compute_score: {e}")
        # return default value
        return {"score": 0.0, "error": str(e), "data_source": data_source}
