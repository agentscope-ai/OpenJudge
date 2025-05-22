from src.data.base import BaseData
from src.evaluation.prompt import BasePrompt
from src.pipeline.base import PipelineBase


def demo():
    data: BaseData = None
    """
    init pipeline from gallery directly
    """
    reward_pipeline = PipelineBase.from_gallery(path="pipeline.mxc.demo")
    reward_pipeline.run(data)

    """
    init pipeline with node in gallery
    """

    parser_prompt = BasePrompt.from_gallery(path="block.prompt.raw.writing")
    judge_prompt = BasePrompt.from_string(data="XXXXXXXXX")

    #custom_prompt = BasePrompt()

    nodes = [("parse",parser_prompt),("judge",judge_prompt)]
    reward_pipeline = PipelineBase(nodes)
    reward_pipeline.run(data)

