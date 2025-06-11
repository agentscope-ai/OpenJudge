from pydantic import Field

from rm_gallery.core.rm.template import BasePromptTemplate


class HelperfulnessTrainTemplate(BasePromptTemplate):
    """
    The PrincipleTemplate class inherits from BasePromptTemplate and is used to define the template for principles reasoning.
    """

    score: str = Field(default=..., description="score of helpfulness from 0 to 4")

    @classmethod
    def parse(cls, text: str):
        """
        Parse text and create instance with error handling.

        Args:
            text: The text to parse

        Returns:
            cls: An instance of the class

        Raises:
            ValueError: When text is invalid or required fields are missing
            TypeError: When score cannot be converted to integer
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            contents = cls._parse(text)
        except Exception as e:
            raise ValueError(f"Failed to parse text: {e}")

        if not contents:
            raise ValueError("No content found in the parsed text")

        # Check if required fields exist
        if "score" not in contents:
            raise ValueError("Missing required field 'score' in parsed content")

        # Convert score to integer with error handling
        try:
            contents["score"] = int(contents["score"])
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Score must be convertible to integer, got '{contents['score']}': {e}"
            )

        # Validate score range if needed (optional)
        if not (0 <= contents["score"] <= 4):  # Assuming score should be 0-4
            raise ValueError(f"Score must be between 0 and 4, got {contents['score']}")

        # Create instance with error handling
        try:
            return cls(**contents)
        except Exception as e:
            raise ValueError(f"Failed to create instance with contents {contents}: {e}")

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
