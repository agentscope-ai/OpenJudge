from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.file import read_jsonl, write_json
from rm_gallery.gallery.classification.classification import (
    ClassificationListWiseReward,
)
from tests.test_principle_generator import calc_acc, generate


def test_generate(
    file: str = "data/rmb/pair/rmbbenchmark_pairwise_classification_content_categorization_output.jsonl",
):
    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]
    desc = """Classification: Entails assigning predefined categories or labels to text based on its content."""
    principles = generate(samples, desc)
    principles = [f"{k}: {v}" for k, v in principles.items()]
    logger.info(f"------------principles--------------\n{principles}")
    return principles


def test_evaluate(
    file: str = "data/rmb/pair/rmbbenchmark_pairwise_classification_content_categorization_output.jsonl",
    principles: List[str] = ["Judge according to your own standard."],
):
    llm = OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True)

    samples = read_jsonl(file)
    samples = [DataSample(**sample) for sample in samples]

    reward_module = ClassificationListWiseReward(
        llm=llm,
        name="classification_listwise",
        principles=principles,
    )

    samples = reward_module.evaluate_batch(
        samples, thread_pool=ThreadPoolExecutor(max_workers=128)
    )
    acc = calc_acc(samples)
    logger.info(f"{file}: {acc}")


principles = test_generate(
    "/mnt3/huangsen.huang/codes/RM-Gallery/data/rmb/pair/train/rmbbenchmark_pairwise_classification_quality_and_compliance_assessment_output.jsonl"
)

write_json(principles, "data/principle.json")
test_evaluate(
    "/mnt3/huangsen.huang/codes/RM-Gallery/data/rmb/pair/test/rmbbenchmark_pairwise_classification_quality_and_compliance_assessment_output.jsonl",
    principles=principles,
)
