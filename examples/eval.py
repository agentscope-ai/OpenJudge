from src.data.base import BaseData
from src.evaluation.prompt import BasePrompt
from src.executor.base import MultiStrateragyExecutor


def demo():
    data: BaseData = None
    """
    init executor from gallery directly
    """
    reward_pipeline = MultiStrateragyExecutor.from_gallery(path="executor.mxc.demo")
    reward_pipeline.run(data)

    """
    init executor with node in gallery
    """

    parser_prompt = BasePrompt.from_gallery(path="block.prompt.raw.writing")
    judge_prompt = BasePrompt.from_string(data="XXXXXXXXX")

    #custom_prompt = BasePrompt()

    nodes = [("parse",parser_prompt),("judge",judge_prompt)]
    reward_pipeline = MultiStrateragyExecutor(nodes)
    reward_pipeline.run(data)

