from loguru import logger

from rm_gallery.core.model.openai_llm import OpenaiLLM


def test_llm():
    llm = OpenaiLLM(model="qwen3-32b", reasoning=True)
    response = llm.simple_chat(query="hello")
    logger.info(f"response: {response}")


test_llm()
