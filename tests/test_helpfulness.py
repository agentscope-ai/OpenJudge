from loguru import logger

from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step
from rm_gallery.core.model.message import MessageRole
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward


def test_single() -> None:
    sample = DataSample(
        unique_id="test_helpfulness",
        input=[
            ChatMessage(
                role=MessageRole.USER,
                content="Explain sarcoidosis to me like I’m a five year old",
                additional_kwargs={"context": ""},
            )
        ],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="""
            Okay, you know how sometimes when you get a little cut, it gets all red and puffy? That's because your body is sending tiny soldiers called \"white blood cells\" to fix the problem. They're like the superheroes inside you that help you heal.\n\nSarcoidosis is when these tiny superheroes get a little confused. Instead of fighting only the bad guys like germs or fixing boo-boos, they start to gather together in places inside your body where they don't need to be, like your lungs or your skin. They form little groups called \"granulomas.\"\n\nImagine you have a bunch of playdough balls sticking together in your toy box when they're not supposed to. It’s kind of like that, but inside your body. These playdough balls can make it hard for the parts of your body to work right, like making it harder to breathe if they're in your lungs, or making your skin feel funny.\n\nDoctors don't really know why the superheroes get confused, but they're working super hard to figure it out. And they have medicines that can help the superheroes calm down and stop making those playdough balls, so the person with sarcoidosis can feel better.
            """,
                )
            ),
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="""
           Sarcoidosis is a condition where your body's immune system starts to attack different parts of your body, like your lungs or your skin. It's like your body is trying to protect itself from something it thinks is harmful, but it ends up damaging your own body instead. It can cause lumps and bumps on your skin, and it can also affect your lungs and other organs in your body. It's important to stay healthy and take care of yourself so that the condition doesn't get any worse.
            """,
                )
            ),
        ],
    )

    logger.info(f"input={sample.model_dump_json()}")
    llm = OpenaiLLM(model="qwen-max")

    helpfulness = BaseHelpfulnessListwiseReward(llm=llm, name="helpfulness_pairwise")
    # helpfulness = RewardRegistry.get("helpfulness_pairwise")(llm = llm,name="helpfulness_pairwise")

    sample = helpfulness.evaluate(sample=sample)
    logger.info(f"output={sample.model_dump_json()}")


test_single()
