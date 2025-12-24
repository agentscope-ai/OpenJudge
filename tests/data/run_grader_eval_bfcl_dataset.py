"""
Grader Test Runner for RM Gallery

This script evaluates the performance of a specified grader (e.g., ToolCallAccuracyGrader)
on a set of test cases from the BFCL dataset. It supports random sampling, skipping initial
cases, and validating scores against expected min/max thresholds.

Usage:
    python run_grader_tests.py --grader_name <GraderClassName> --sample_size <N> --skip <K>

Environment Variables:
    - BASE_URL: The base URL for the LLM API endpoint.
    - API_KEY: The API key for authenticating with the LLM service.

Example:
    python run_grader_eval_bfcl_dataset.py --grader_name ToolCallAccuracyGrader --sample_size 50 --skip 10

"""

import argparse
import asyncio
import json
import os
import random
from typing import List

import nest_asyncio

from rm_gallery.core.graders.agent import *
from rm_gallery.core.graders.common import *
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

nest_asyncio.apply()


def read_jsonl(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def random_sample_jsonl(data, sample_size, seed=42):
    if not sample_size:
        print(f"All data will be selected because sample_size is not set")
        return data
    if sample_size > len(data):
        raise ValueError(f"Random select {sample_size} samples, but only {len(data)} samples exists")
    random.seed(seed)
    sample_data = random.sample(data, sample_size)
    return sample_data


def get_required_env(var_name):
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Required environment variable '{var_name}' is not set.")
    return value


def run_cases(case_file: str, grader_name: str, sample_size: int, skip: int):
    model_config = {
        "model": "qwen-long",
        "base_url": get_required_env("BASE_URL"),
        "api_key": get_required_env("API_KEY"),
        "temperature": 0,
        "stream": False,
    }
    cnt = 0
    cases = read_jsonl(case_file)
    select_cases = cases[skip:]
    sample_data = random_sample_jsonl(select_cases, sample_size)
    num_data = len(sample_data)
    for data in sample_data:
        cls_name = grader_name
        kwargs = data.get("parameters")
        index = data["index"]

        try:
            grader = eval(cls_name)(model=model_config, language=LanguageEnum.ZH)
            result = asyncio.run(grader.aevaluate(**kwargs))

            has_min = "min_expect_score" in data
            has_max = "max_expect_score" in data

            if not (has_min or has_max):
                print(f"\033[91mFAILED \033[0m: index: {index}, missing min_expect_score or max_expect_score")
                continue

            if has_min and result.score < data["min_expect_score"]:
                print(
                    f"\033[91mFAILED \033[0m, index: {index}, result score {result.score} is less than min_expect_score {data['min_expect_score']}, result: {result}"
                )
                continue

            if has_max and result.score > data["max_expect_score"]:
                print(
                    f"\033[91mFAILED\033[0m, index: {index}, result score {result.score} is greater than max_expect_score {data['max_expect_score']}, result: {result}"
                )
                continue
            cnt += 1
            print(f"PASSED: index: {index}, score: {result.score}")

        except Exception as e:
            print(f"failed case index: {index}")
            print(e)

    pass_rate = cnt / num_data
    print(f"{grader_name} pass rate: {pass_rate} ({cnt}/{num_data})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grader test cases from BFCL dataset.")
    parser.add_argument(
        "--case_file",
        type=str,
        default="bfcl_v3/tool_call/tool_call_bfcl_v3_multiple_eval_data.jsonl",
        help="Specify the case file as input data",
    )
    parser.add_argument(
        "--grader_name",
        type=str,
        default="ToolCallAccuracyGrader",
        help="Specify the name of grader to run",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of random sampling",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of test cases to skip from the beginning (default: 0)",
    )
    args = parser.parse_args()

    run_cases(
        case_file=args.case_file,
        grader_name=args.grader_name,
        sample_size=args.sample_size,
        skip=args.skip,
    )
