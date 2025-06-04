from concurrent.futures import ThreadPoolExecutor

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.rm.generator import PrincipleGenerator
from rm_gallery.core.utils.file import read_jsonl


def test_principle_generator():
    llm = OpenaiLLM(model="deepseek-chat")
    samples = read_jsonl("data/alpacaeval-hard-0603.jsonl")
    samples = [DataSample(**sample) for sample in samples]

    generator = PrincipleGenerator(llm=llm)
    principles = generator.run_batch(
        samples[:10], thread_pool=ThreadPoolExecutor(max_workers=10)
    )
    print(principles)


test_principle_generator()
