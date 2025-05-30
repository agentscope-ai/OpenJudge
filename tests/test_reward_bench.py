import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.temp import (
    load_rewardbench_test_samples,
    write_datasamples_to_file,
)
from rm_gallery.gallery.rm.harmfulness_paiwise import HarmfulnessPairwise
from rm_gallery.gallery.rm.helpfulness_pairwise import HelpfulnessPairwise

os.environ["OPENAI_API_KEY"] = "REDACTED_API_KEY"
os.environ["BASE_URL"] = "http://8.130.177.212:3000/v1"

# map rewardbench map to HHH RM
HHHRM_subset_mapping = {
    "helpfulness": [
        "alpacaeval-easy",
        "alpacaeval-hard",
        "alpacaeval-length",
        "mt-bench-easy",
        "mt-bench-med",
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
        "hep-python",
        "hep-go",
        "hep-cpp",
        "hep-js",
        "hep-rust",
        "hep-java",
        "math-prm",
    ],
    "harmfulness": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-respond",
        "xstest-should-refuse",
        "donotanswer",
    ],
}

rewardbench_samples = load_rewardbench_test_samples(
    input_path="tests/data/rewardbench_filter_output_0530.jsonl"
)
helpfulness_samples = []
harmfulness_samples = []


def _map_subset_to_HHH_RM():
    for sample in rewardbench_samples:
        subset = sample.metadata["raw_data"]["subset"]
        if subset in HHHRM_subset_mapping["helpfulness"]:
            helpfulness_samples.append(sample)
        elif subset in HHHRM_subset_mapping["harmfulness"]:
            harmfulness_samples.append(sample)


_map_subset_to_HHH_RM()
helpfulness_samples = helpfulness_samples[:500]
harmfulness_samples = harmfulness_samples[:500]

llm = OpenaiLLM(model="qwen-max")
harmfulness = HarmfulnessPairwise(llm=llm, name="harmfulness")
helpfulness = HelpfulnessPairwise(llm=llm, name="helpfulness_pairwise")
thread_pool = ThreadPoolExecutor(max_workers=100)

logger.info(
    f"input_help={[sample.model_dump_json() for sample in helpfulness_samples]}"
)
logger.info(
    f"input_harm={[sample.model_dump_json() for sample in harmfulness_samples]}"
)

helpfulness_samples = helpfulness.evaluate_batch(helpfulness_samples, thread_pool)
harmfulness_samples = harmfulness.evaluate_batch(harmfulness_samples, thread_pool)


write_datasamples_to_file(
    helpfulness_samples, "tests/data/rewardbench_filter_helpfulness_reward.jsonl"
)
write_datasamples_to_file(
    harmfulness_samples, "tests/data/rewardbench_filter_harmfulness_reward.jsonl"
)
