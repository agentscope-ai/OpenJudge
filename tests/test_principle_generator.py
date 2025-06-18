import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.principle.generator import PrincipleGenerator
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.utils.file import read_json, read_jsonl, write_json
from rm_gallery.gallery import *  # ignore [E402]


def generate(samples: List[DataSample], scenario, generator_number, cluster_number):
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)

    generator = PrincipleGenerator(
        llm=llm,
        scenario=scenario,
        generate_number=generator_number,
        cluster_number=cluster_number,
    )
    principles = generator.run_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=256)
    )
    return principles


def test_principle_generator():
    samples = read_jsonl("data/loaded_dataset_rewardbench_2_output.jsonl")
    samples = [DataSample(**sample) for sample in samples]

    principles = generate(samples)
    print(principles)


def test_3h_generate():
    HHH = {
        "Honesty": ["Factuality"],
        # "Harmfulness": ["Safety"],
        # "Helpfulness": ["Precise IF", "Math", "Focus", "Ties"],
    }

    principles = {}

    for h_name, h_set in HHH.items():
        samples = []
        for file in h_set:
            samples += read_jsonl(f"data/rewardbench2/{file} Train.jsonl")

        samples = [DataSample(**sample) for sample in samples]
        desc = f"You are a professional expert in {h_name.lower()} evaluation"
        principles[h_name] = generate(samples, desc)

    write_json(principles, "data/3h_principles.json")


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


def test_3h_rm():
    HHH = {
        "Honesty": ["Factuality"],
        # "Harmfulness": ["Safety"],
        # "Helpfulness": ["Precise IF"],  # , "Math", "Focus", "Ties"
    }

    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)

    results = {}
    for h_name, h_set in HHH.items():
        for file in h_set:
            samples = read_jsonl(f"data/rewardbench2/{file} Test.jsonl")
            samples = [DataSample(**sample) for sample in samples]

            reward_name = f"{h_name.lower()}_listwise"
            reward_module = RewardRegistry.get(reward_name)(
                llm=llm,
                name=f"{h_name.lower()}_listwise",
                # principles=["Judge according to your own standard."],
            )

            samples = reward_module.evaluate_batch(
                samples, thread_pool=ThreadPoolExecutor(max_workers=128)
            )
            acc = calc_acc(samples)
            logger.info(f"{file}: {acc}")
            results[file] = acc

    logger.info(results)
    write_json(results, "data/3h_rm_no_result.json")


def test_rmb_generate():
    principles = {}
    for file in os.listdir("data/rmb/pair/train"):
        if not file.endswith(".jsonl"):
            continue

        samples = read_jsonl(f"data/rmb/pair/train/{file}")
        samples = [DataSample(**sample) for sample in samples]
        desc = "You are a professional expert in helpfuness evaluation"
        principles[file] = generate(samples, desc)

    write_json(principles, "data/rmb_principles.json")


def test_3h_rmb():
    results = {}
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)
    principles = read_json("data/rmb_principles.json")
    for file in os.listdir("data/rmb/bon"):
        if not file.endswith(".jsonl"):
            continue

        samples = read_jsonl(f"data/rmb/bon/{file}")
        samples = [DataSample(**sample) for sample in samples]
        file = file.replace("bestofn", "pairwise")

        reward_name = "helpfulness_listwise"
        _principles = []
        for k, v in principles[file].items():
            _principles.append(f"{k}: {v}")
        reward_module = RewardRegistry.get(reward_name)(
            llm=llm,
            name="helpfulness_listwise",
            principles=["Judge according to your own standard."],
            # principles=_principles,
        )

        samples = reward_module.evaluate_batch(
            samples, thread_pool=ThreadPoolExecutor(max_workers=1)
        )
        acc = calc_acc(samples)
        logger.info(f"{file}: {acc}")
        results[file] = acc

    logger.info(results)
    write_json(results, "data/rmb.json")


# test_3h_rm()
# test_3h_generate()
# test_rmb_generate()
# test_3h_rmb()
