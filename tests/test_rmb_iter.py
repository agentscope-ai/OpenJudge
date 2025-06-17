import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.principle.iter import IterPrincipleGenerator
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.utils.file import read_jsonl, write_json
from rm_gallery.gallery.rm.alignment.rmb.helpfulness import TASKS

os.environ["OPENAI_API_KEY"] = "REDACTED_API_KEY"
os.environ["BASE_URL"] = "http://8.130.177.212:3000/v1"
os.environ["PYTHONPATH"] = "${workspaceFolder}"


def calc_acc(samples: List[DataSample]):
    labels = []
    for sample in samples:
        for output in sample.output:
            if (
                output.answer.label["preference"] == "chosen"
                and output.answer.reward.details
            ):
                score = sum(r.score for r in output.answer.reward.details)
                if score > 0:
                    labels.append(1)
                else:
                    labels.append(0)

    return sum(labels) / len(labels), len(labels), len(samples)


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


def get_reward(task: str, principles=None):
    kwargs = {}
    kwargs["llm"] = OpenaiLLM(model="qwen3-8b", enable_thinking=True)
    if principles is not None:
        kwargs["principles"] = principles

    task = task.lower().replace(" ", "_")
    kwargs["name"] = task
    return RewardRegistry.get(f"{task}_listwise_reward")(**kwargs)


def test_generate(
    task: str,
    file: str = "data/RMBbench/pairwise/Helpfulness/Role Playing/Specific Character.jsonl",
    generate_number: int = 10,
    cluster_number: int = 3,
):
    reward = get_reward(task)
    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]
    principles = generate(
        samples, reward.scenario, generate_number, cluster_number, reward
    )
    principles = [f"{k}: {v}" for k, v in principles.items()]
    logger.info(f"------------principles--------------\n{principles}")
    return principles


def test_evaluate(
    task: str,
    file: str = "data/RMBbench/pairwise/Helpfulness/Role Playing/Specific Character.jsonl",
    principles: List[str] | None = None,
):
    # qwen3-235b-a22b

    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]
    total = len(samples)

    reward_module = get_reward(task, principles)

    samples = reward_module.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=256)
    )
    acc, success, total = calc_acc(samples)
    logger.info(f"{file}: {acc}")
    return acc, success, total


def parse(path: str):
    path.replace("RMBbench", "RMBbench_Train")


def test_single(task: str):
    logger.info(f"------------{task}--------------")

    path = f"data/RMBbench/bestofn/Helpfulness/{task}.jsonl"
    train, test = path.replace("RMBbench", "RMBbench_Train"), path.replace(
        "RMBbench", "RMBbench_Test"
    )

    # principles = DEFAULT_HELPFULNESS_PRINCIPLES
    # principles = []
    principles = None

    # principles = test_generate(
    #     task,
    #     train,
    # )

    acc, success, total = test_evaluate(
        task,
        test,
        principles=principles,
    )

    return acc, success, total, principles


def test_all(tasks, i):
    results = {}
    for task in tasks:
        results[task] = test_single(task)

        logger.info(f"Results: {results}")
        # iter_10_3
        write_json(results, f"data/RMBbench/results_rmb_opt_{i}.json")


for i in range(1, 2):
    test_all(TASKS, i)
