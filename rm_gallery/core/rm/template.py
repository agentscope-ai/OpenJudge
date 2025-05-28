import re
from typing import Self
from pydantic import BaseModel, Field

from rm_gallery.core.model.base import BaseLLM


class BaseTemplate(BaseModel):
    """
    BaseTemplate serves as the base class for all template classes, providing methods to parse, format, and generate schema based on template structures.
    """
    
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
        # Define a regular expression pattern to match the template format
        pattern = r'<([^>]+)>(.*)</\1>'
        # Use the findall method of the re module to get all matches
        matches = re.findall(pattern, text, re.DOTALL)
        # Convert the matches into a dictionary
        contents = {match[0]: match[1] for match in matches}
        # Use the dictionary to initialize an instance of the class
        return cls(**contents)

    @classmethod
    def schema(cls) -> str:
        """
        Generates a schema string based on the class's JSON schema, describing the structure and purpose of the template.
        
        Returns:
        - str: A string describing the schema, containing the description of each property.
        """
        # Initialize an empty string to store the schema
        schema_str = ""
        # Iterate through the properties in the JSON schema
        for key, property in cls.model_json_schema(by_alias=True)["properties"].items():
            # Add the property description to the schema string in the specified format
            schema_str += f"<{key}>{property['description']}</{key}>"
        # Return the schema string
        return schema_str
    
    @classmethod
    def format(cls, **kwargs) -> str:
        """
        Formats the input parameters according to the template format into a string.
        
        Parameters:
        - **kwargs: Arbitrary keyword arguments, representing the properties to be formatted.
        
        Returns:
        - str: A string formatted according to the template, containing the given properties.
        """
        ...

    @classmethod
    def call(cls, llm: BaseLLM, **kwargs) -> Self:
        """
        Calls the format method to generate a string based on the input parameters.
        
        Parameters:
        - **kwargs: Arbitrary keyword arguments, representing the properties to be formatted.
        
        Returns:
        - str: A string formatted according to the template, containing the given properties.
        """
        query = cls.format(**kwargs)
        response = llm.simple_chat(query=query)
        output = cls.parse(response)
        return output


class ReasoningTemplate(BaseTemplate):
    """
    The ReasoningTemplate class inherits from BaseTemplate and is used to define the template for reasoning analysis.
    It mainly includes a field for reasoning analysis process, which is used to record the analysis and reasoning process during the reasoning.
    
    Attributes:
        reason (str): Represents the analysis process, defaults to an empty string. It is identified by the alias "think".
    """
    reason: str = Field(default=..., description="analysis process", alias="think")


class PrincipleTemplate(ReasoningTemplate):
    """
    The PrincipleTemplate class inherits from ReasoningTemplate and is used to define the template for principles reasoning.
    """
    violation: str = Field(default=..., description="indices of voilated principles")

    @classmethod
    def format(cls, desc: str, principles: str, examples: str, query: str, context: str, answer: str) -> str:

        return f"""# Task Description
{desc}
# Principles
{principles}
# Examples
{examples}
# Query
{query}
# Context
{context}
# Answer
{answer}
# Output Format
{cls.schema()}
"""
