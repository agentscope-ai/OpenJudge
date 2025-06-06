import re
from typing import Dict, List, Self

from pydantic import BaseModel, Field


class BasePromptTemplate(BaseModel):
    """
    BasePromptTemplate serves as the base class for all template classes, providing methods to parse, format, and generate schema based on template structures.
    """

    reason: str = Field(default=..., description="your reasoning trace", alias="think")

    @classmethod
    def _parse(cls, text: str) -> Dict[str, str]:
        # Define a regular expression pattern to match the template format
        pattern = r"<([^>]+)>(.*)</\1>"
        # Use the findall method of the re module to get all matches
        matches = re.findall(pattern, text, re.DOTALL)
        # Convert the matches into a dictionary
        contents = {match[0]: match[1].strip() for match in matches}
        return contents

    @classmethod
    def parse(cls, text: str) -> Self:
        """
        Parses a string according to a specified template format and returns an instance of this class.

        The template format is: <key>value</key>, where the key is the property name and the value is the property content.

        Parameters:
        - text (str): The string to parse, which should follow the template format.

        Returns:
        - Self: An instance of the class, initialized with the parsed key-value pairs.
        """
        contents = cls._parse(text)
        # Use the dictionary to initialize an instance of the class
        return cls(**contents)

    @classmethod
    def schema(cls, enable_thinking: bool = False, **kwargs) -> str:
        """
        Generates a schema string based on the class's JSON schema, describing the structure and purpose of the template.

        Returns:
        - str: A string describing the schema, containing the description of each property.
        """
        # Initialize an empty string to store the schema
        schema_str = "Note: Ensure all outputs are placed within the tags like <tag> </tag> as required!!!\n"
        # Iterate through the properties in the JSON schema
        for key, property in cls.model_json_schema(by_alias=True)["properties"].items():
            # Add the property description to the schema string in the specified format
            if key != "think" or not enable_thinking:
                schema_str += f"<{key}>\n{property['description']}\n</{key}>\n"
        # Return the schema string
        return schema_str

    @classmethod
    def format(cls, enable_thinking: bool = False, **kwargs) -> str:
        """
        Formats the input parameters according to the template format into a string.

        Parameters:
        - **kwargs: Arbitrary keyword arguments, representing the properties to be formatted.

        Returns:
        - str: A string formatted according to the template, containing the given properties.
        """
        ...


class PrinciplePointWiseTemplate(BasePromptTemplate):
    """
    The PrincipleTemplate class inherits from BasePromptTemplate and is used to define the template for principles reasoning.
    """

    violation: List[str] = Field(
        default=..., description="a list of voilated principles"
    )

    @classmethod
    def parse(cls, text: str):
        contents = cls._parse(text)
        contents["violation"] = eval(contents["violation"])
        return cls(**contents)

    @classmethod
    def format(
        cls,
        desc: str,
        principles: List[str],
        examples: str,
        query: str,
        context: str,
        answer: str,
        **kwargs,
    ) -> str:
        principles_str = ""
        for i, principle in enumerate(principles):
            principles_str += f"{i + 1}. {principle}\n"
        return f"""# Task Description
{desc}
# Principles
{principles_str}
# Examples
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


class PrincipleListWiseTemplate(BasePromptTemplate):
    best: int = Field(
        default=...,
        description="which answer is the best? just give the number here!!!",
    )

    @classmethod
    def parse(cls, text: str):
        contents = cls._parse(text)
        contents["best"] = int(contents["best"])
        return cls(**contents)

    @classmethod
    def format(
        cls,
        desc: str,
        principles: str,
        examples: str,
        query: str,
        answers: List[str],
        **kwargs,
    ) -> str:
        answer_str = ""
        for i, answer in enumerate(answers):
            answer_str += f"# Answer {i + 1}\n{answer}\n"

        return f"""# Task Description
{desc}
# Principles
{principles}
# Examples
{examples}
# Query
{query}

{answer_str}
# Output Requirement
{cls.schema(**kwargs)}
"""
