import re
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BasePromptTemplate(BaseModel):
    """
    BasePromptTemplate serves as the abstract base class for all prompt template implementations.

    This class provides core functionality for parsing structured templates, formatting output schemas,
    and validating content against defined field requirements. It implements the fundamental patterns
    for bidirectional conversion between string representations and structured data models.

    Attributes:
        reason (str): A field capturing the reasoning trace for decision-making processes
    """

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)
    reason: Optional[str] = Field(
        default=None, description="your reasoning trace", alias="think"
    )

    @classmethod
    def _parse(cls, text: str) -> Dict[str, str]:
        """
        Extracts key-value pairs from XML-style tagged text using regex pattern matching.

        This internal method identifies structured patterns in the format <key>value</key>
        and converts them into a dictionary mapping for further processing.

        Args:
            text (str): Input string containing XML-style tagged content

        Returns:
            Dict[str, str]: Dictionary mapping of tag names to corresponding values
        """
        pattern = r"<([^>]+)>(.*)</\1>"
        matches = re.findall(pattern, text, re.DOTALL)
        contents = {match[0]: match[1].strip() for match in matches}
        return contents

    @classmethod
    def parse(cls, text: str) -> "BasePromptTemplate":
        """
        Converts a structured text string into a validated template instance.

        Processes input text through internal parsing mechanism and constructs
        a model instance with validated field values.

        Args:
            text (str): XML-style formatted string containing template data

        Returns:
            BasePromptTemplate: Constructed instance with parsed field values
        """
        contents = cls._parse(text)
        contents.setdefault("think", "")
        return cls(**contents)

    @classmethod
    def schema(cls, enable_thinking: bool = False, **kwargs) -> str:
        """
        Generates a descriptive schema documentation string for the template structure.

        Creates a human-readable documentation showing required fields, their descriptions,
        and proper output formatting requirements.

        Args:
            enable_thinking (bool): Flag to include/exclude thinking field in schema
            **kwargs: Additional parameters passed to schema generation

        Returns:
            str: Formatted schema documentation string with field descriptions
        """
        schema_str = "Note: Ensure all outputs are placed within the tags like <tag> </tag> as required!!!\n"
        for key, property in cls.model_json_schema(by_alias=True)["properties"].items():
            if key == "model_config":
                continue

            if key == "think" and enable_thinking:
                continue

            if key == "think":
                schema_str += f"<reason>\n{property['description']}\n</reason>\n"
            else:
                schema_str += f"<{key}>\n{property['description']}\n</{key}>\n"
        return schema_str

    @classmethod
    def format(cls, enable_thinking: bool = False, **kwargs) -> str:
        """
        Formats provided content into the template's required output structure.

        Takes arbitrary keyword arguments and formats them into the appropriate
        template structure for response generation.

        Args:
            enable_thinking (bool): Flag to control inclusion of reasoning field
            **kwargs: Content to be formatted into template structure

        Returns:
            str: Formatted string ready for model processing
        """
        ...


class PrinciplePointWiseTemplate(BasePromptTemplate):
    """
    Template implementation for principle-based point-wise evaluation tasks.

    This template structure is designed for scenarios requiring analysis of principle
    violations in specific contexts, with support for detailed scenario descriptions
    and example-based guidance.

    Attributes:
        violation (List[str]): List of identified principle violations
    """

    violation: List[str] = Field(
        default=..., description="a list of violated principles"
    )

    @classmethod
    def parse(cls, text: str):
        """
        Parses text input containing principle violation information.

        Processes standard template format and converts violation field
        from string representation to Python list.

        Args:
            text (str): Input string containing XML-style tagged content

        Returns:
            PrinciplePointWiseTemplate: Constructed instance with parsed values
        """
        contents = cls._parse(text)
        try:
            contents["violation"] = eval(contents["violation"])
        except Exception:
            contents["violation"] = []
        return cls(**contents)

    @classmethod
    def format(
        cls,
        desc: str,
        scenario: str,
        principles: str,
        examples: str,
        query: str,
        context: str,
        answer: str,
        **kwargs,
    ) -> str:
        """
        Formats evaluation components into structured prompt template.

        Combines task description, scenario context, principles, and response
        requirements into standardized prompt format.

        Args:
            desc (str): Task description text
            scenario (str): Scenario context description
            principles (str): List of relevant principles
            examples (str): Example-based guidance
            query (str): Evaluation query text
            context (str): Additional contextual information
            answer (str): Reference answer text
            **kwargs: Additional formatting parameters

        Returns:
            str: Formatted prompt string following template requirements
        """
        if examples:
            examples = f"\n# Examples\n{examples}\n"

        if scenario:
            scenario = f"\n# Scenario\n{scenario}\n"

        if context:
            context = f"\n# Context\n{context}\n"

        return f"""# Task Description
{desc}
{scenario}

# Principles
{principles}
{examples}

# Query
{query}
{context}

# Answer
{answer}

# Output Requirement
{cls.schema(**kwargs)}
"""


class PrincipleListWiseTemplate(BasePromptTemplate):
    """
    Template implementation for principle-based list-wise evaluation tasks.

    Designed for comparative evaluation scenarios where multiple answers need
    to be assessed against defined principles to determine the optimal choice.

    Attributes:
        best (int): Index of the best-performing answer according to principles
    """

    best: int = Field(
        default=...,
        description="which answer is the best? just give the number here!!!",
    )

    @classmethod
    def parse(cls, text: str):
        """
        Parses text input containing list-wise evaluation results.

        Converts best answer index from string to integer format
        during template instantiation.

        Args:
            text (str): Input string containing XML-style tagged content

        Returns:
            PrincipleListWiseTemplate: Constructed instance with parsed values
        """
        contents = cls._parse(text)
        contents["best"] = int(contents["best"])
        return cls(**contents)

    @classmethod
    def format(
        cls,
        desc: str,
        scenario: str,
        principles: str,
        examples: str,
        query: str,
        context: str,
        answers: List[str],
        **kwargs,
    ) -> str:
        """
        Formats comparative evaluation components into structured prompt template.

        Combines task description, scenario context, principles, and multiple
        candidate answers into standardized prompt format for list-wise evaluation.

        Args:
            desc (str): Task description text
            scenario (str): Scenario context description
            principles (str): List of relevant principles
            examples (str): Example-based guidance
            query (str): Evaluation query text
            context (str): Additional contextual information
            answers (List[str]): List of candidate answers for comparison
            **kwargs: Additional formatting parameters

        Returns:
            str: Formatted prompt string following template requirements
        """
        answer_str = ""
        for i, answer in enumerate(answers):
            answer_str += f"## Answer {i + 1}\n{answer}\n\n"

        if examples:
            examples = f"# Examples\n{examples}\n"

        if scenario:
            scenario = f"\n# Scenario\n{scenario}\n"

        if context:
            context = f"\n# Context\n{context}\n"

        if principles:
            principles = f"# Principles\n{principles}\n"

        return f"""# Task Description
{desc}
{scenario}

{principles}
{examples}

# Query
{query}
{context}

# Answers
{answer_str}

# Output Requirement
{cls.schema(**kwargs)}
"""
