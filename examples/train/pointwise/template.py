from pydantic import Field

from rm_gallery.core.reward.template import BasePromptTemplate


class HelpfulnessTrainTemplate(BasePromptTemplate):
    """
    The PrincipleTemplate class inherits from BasePromptTemplate and is used to define the template for principles reasoning.
    """

    score: int = Field(default=..., description="score of helpfulness from 0 to 4")

    @classmethod
    def parse(cls, text: str):
        """Parse text and create instance."""
        try:
            contents = cls._parse(text)
            if "score" in contents:
                contents["score"] = int(contents["score"])
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
        context: str,
        answer: str,
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

        # Context
        {context}

        # Answer
        {answer}

        # Output Requirement
        {cls.schema(**kwargs)}
        """
