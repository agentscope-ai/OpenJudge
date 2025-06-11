import math

from rm_gallery.examples.train.pointwise.helpfulness_template import (
    HelperfulnessTrainTemplate,
)


def calculate_helpfulness_reward(predicted_score, true_score):
    """
    base on relative error
    use exponential decay, more tolerant for small errors, more severe for large errors
    """
    if true_score is None:
        return 0.0

    # calculate relative error (consider the case of denominator is 0)
    if true_score == 0:
        abs_error = abs(predicted_score)
        max_possible_error = 4  # max score
    else:
        abs_error = abs(predicted_score - true_score)
        max_possible_error = 4  # helpfulness score range 0-4

    # use exponential decay function
    # reward = exp(-k * error_ratio), where k is the decay parameter
    k = 2.0  # decay parameter, can be adjusted
    error_ratio = abs_error / max_possible_error
    reward = math.exp(-k * error_ratio)

    return float(reward)


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
        # extract helpfulness score from solution_str using BasePromptTemplate logic
        parsed_result = HelperfulnessTrainTemplate.parse(solution_str)

        # Extract integer score from violation (now List[int])
        predicted_helpfulness = parsed_result.score

        # Validate range
        if not (0 <= predicted_helpfulness <= 4):
            predicted_helpfulness = 0

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

        accuracy = 1 if predicted_helpfulness == true_helpfulness else 0

        # return detailed information
        return {
            "score": reward,
            "predicted_helpfulness": predicted_helpfulness,
            "true_helpfulness": true_helpfulness,
            "data_source": data_source,
            "accuracy": accuracy,
        }

    except Exception as e:
        return {
            "score": 0.0,
            "predicted_helpfulness": 0,
            "true_helpfulness": 0,
            "data_source": data_source,
            "accuracy": 0,
        }
