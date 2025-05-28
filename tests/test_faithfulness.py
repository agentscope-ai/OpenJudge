from gallery.rm.faithfulness import ExtractClaims, ExtractTruths, FaithfulnessReward
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, MessageRole, Step
from rm_gallery.core.model.base import LLMClient


def test_faithfulness() -> None:
    sample = DataSample(
        unique_id="test_faithfulness",
        input=[
            ChatMessage(
                role=MessageRole.USER, content="今天是几号", additional_kwargs={
                    "context": "当前日期：2025年5月20号，周二"
                }
            )
        ],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT, content="今天是周四"
                )
            ),
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT, content="今天是周二"
                )
            ),
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT, content="今天是2025年5月20号，周二"
                )
            )
        ]
    )

    llm = LLMClient(model="qwen-max")

    extract_turths = ExtractTruths(
        llm=llm,
        name="extract_truths"
    )

    extract_claims = ExtractClaims(
        llm=llm,
        name="extract_claims"
    )
        
    faithfullness = FaithfulnessReward(
        llm=llm,
        name="faithfulness"
    )

    extract_turths.run(sample)
    extract_claims.run(sample)
    faithfullness.run(sample)