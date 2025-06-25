from loguru import logger

from rm_gallery.core.model.openai_llm import OpenaiLLM


def test_llm():
    llm = OpenaiLLM(model="qwen3-32b", enable_thinking=False)
    response = llm.simple_chat(query="hello")
    logger.info(f"response: {response}")
