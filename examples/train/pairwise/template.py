from typing import Optional

from pydantic import Field

from rm_gallery.core.reward.template import BasePromptTemplate


class PairwiseComparisonTemplate(BasePromptTemplate):
    """
    Template for pairwise comparison tasks between two responses.
    Used for training models to compare and rank responses.
    """

    think: str = Field(default=..., description="your reasoning trace", alias="think")
    preference: Optional[str] = Field(
        default=None, description="which response is better: A, B, or tie"
    )

    @classmethod
    def parse(cls, text: str):
        """Parse text and create instance."""
        try:
            contents = cls._parse(text)
            # 当未检测到 <preference> 标签时，给出默认值以避免未定义错误
            preference = contents.get("preference", "unknown").strip().upper()

            # Normalize preference values
            if preference in ["A", "RESPONSE A", "ANSWER A"]:
                contents["preference"] = "A"
            elif preference in ["B", "RESPONSE B", "ANSWER B"]:
                contents["preference"] = "B"
            elif preference in ["TIE", "EQUAL", "SAME"]:
                contents["preference"] = "tie"
            else:
                contents["preference"] = preference
            return cls(**contents)
        except Exception as e:
            raise ValueError(f"Failed to parse: {e}")

    @classmethod
    def format(
        cls,
        desc: str,
        principles: str,
        examples: str,
        query: str,
        response_a: str,
        response_b: str,
        **kwargs,
    ) -> str:
        if examples:
            examples = f"# Examples\n{examples}\n"

        return f"""# Task Description
{desc}

# Principles
{principles}

{examples}

# Query
{query}

# Response A
{response_a}

# Response B
{response_b}

# Output Requirement
{cls.schema(**kwargs)}
"""
