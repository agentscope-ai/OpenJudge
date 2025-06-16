from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.principle.iter import IterPrincipleGenerator
from rm_gallery.core.utils.file import read_jsonl, write_json
from rm_gallery.gallery.rm.alignment.base import (
    DEFAULT_HELPFULNESS_PRINCIPLES,
    BaseHelpfulnessListwiseReward,
)
from rm_gallery.gallery.rm.alignment.rmb.helpfulness import TASKS


def calc_acc(samples: List[DataSample]):
    labels = []
    for sample in samples:
        labels.append(0)
        for output in sample.output:
            if output.answer.label["preference"] == "chosen":
                score = sum(r.score for r in output.answer.reward.details)
                if score > 0:
                    labels[-1] = 1
    return sum(labels) / len(labels)


def generate(
    samples: List[DataSample], scenario, generator_number, cluster_number, reward
):
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)

    generator = IterPrincipleGenerator(
        llm=llm,
        scenario=scenario,
        generate_number=generator_number,
        cluster_number=cluster_number,
        reward=reward,
    )
    principles = generator.run_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=256)
    )
    return principles


def get_reward(scenario, principles=[]):
    llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)
    reward = BaseHelpfulnessListwiseReward(
        llm=llm,
        name="rmb_helpfulness_listwise",
        principles=principles,
        scenario=scenario,
    )
    return reward


def test_generate(
    file: str = "data/RMBbench/pairwise/Helpfulness/Role Playing/Specific Character.jsonl",
    scenario: str = "",
    generate_number: int = 10,
    cluster_number: int = 3,
):
    reward = get_reward(scenario)
    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]
    principles = generate(samples, scenario, generate_number, cluster_number, reward)
    principles = [f"{k}: {v}" for k, v in principles.items()]
    logger.info(f"------------principles--------------\n{principles}")
    return principles


def test_evaluate(
    file: str = "data/RMBbench/pairwise/Helpfulness/Role Playing/Specific Character.jsonl",
    principles: List[str] = [
        "Intent Understanding: Understand user intentions and response."
    ],
    scenario: str = "",
):
    # qwen3-235b-a22b

    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]

    reward_module = get_reward(scenario, principles)

    samples = reward_module.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=256)
    )
    acc = calc_acc(samples)
    logger.info(f"{file}: {acc}")
    return acc


def parse(path: str):
    path.replace("RMBbench", "RMBbench_Train")


def test_single(task: str):
    scenario = TASKS[task]
    logger.info(f"------------{task}--------------")
    logger.info(f"Scenario: {scenario}")

    path = f"data/RMBbench/bestofn/Helpfulness/{task}.jsonl"
    train, test = path.replace("RMBbench", "RMBbench_Train"), path.replace(
        "RMBbench", "RMBbench_Test"
    )

    principles = DEFAULT_HELPFULNESS_PRINCIPLES

    principles = test_generate(
        train,
        scenario=scenario,
    )

    acc = test_evaluate(
        test,
        principles=principles,
        scenario=scenario,
    )

    return acc, principles


def test_all(tasks):
    results = {}
    for task, _ in tasks.items():
        results[task] = test_single(task)

        logger.info(f"Results: {results}")
        write_json(results, "data/RMBbench/results_rmb_iter_10_3_0.json")


test_all(tasks=TASKS)
