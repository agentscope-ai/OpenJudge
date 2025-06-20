from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.principle.generator import PrincipleGenerator
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.utils.file import read_jsonl, write_json

TASKS = [
    "Chat",
    "Brainstorming",
    "Classification",
    "Closed QA",
    "Open QA",
    "Generation",
    "Summarization",
    "Translation",
    "Rewrite",
    "Reasoning",
    "Role Playing",
    "Code",
]


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
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=False)

    generator = PrincipleGenerator(
        llm=llm,
        scenario=scenario,
        generate_number=generator_number,
        cluster_number=cluster_number,
        # reward=reward,
    )
    principles = generator.run_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=256)
    )
    return principles


def get_reward(task: str, principles=None):
    kwargs = {}
    kwargs["llm"] = OpenaiLLM(model="qwen3-32b", enable_thinking=False)
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
    thread_pool=ThreadPoolExecutor(max_workers=256),
):
    # qwen3-235b-a22b

    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]
    total = len(samples)

    reward_module = get_reward(task, principles)

    samples = reward_module.evaluate_batch(samples, thread_pool=thread_pool)
    acc, success, total = calc_acc(samples)
    logger.info(f"{file}: {acc}")
    return acc, success, total


def parse(path: str):
    path.replace("RMBbench", "RMBbench_Train")


def test_single(task: str, thread_pool=ThreadPoolExecutor(max_workers=256)):
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
        task, test, principles=principles, thread_pool=thread_pool
    )

    return acc, success, total, principles


def test_all(tasks, i):
    thread_pool = ThreadPoolExecutor(max_workers=384)
    results = {}
    futures = [(task, thread_pool.submit(test_single, task)) for task in tasks]
    wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
    results = {task: future.result() for task, future in futures}
    write_json(results, f"data/RMBbench/results_rmb_generate_{i}.json")


for i in range(5):
    test_all(TASKS, i)
