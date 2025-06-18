from typing import Dict, List

from loguru import logger
from pydantic import BaseModel, Field
from retry import retry

from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.reward.principle.generator import BaseGeneratorTemplate


class PrincipleFilterTemplate(BaseGeneratorTemplate):
    """Template for filtering principles."""

    @classmethod
    def format(cls, scenario: str, principles: List[str], number: int, **kwargs) -> str:
        """Format prompt for principle clustering task.

        Args:
            scenario: Task context description
            number: Maximum number of clustered principles
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string
        """
        return f"""## Overview
Please filter evaluation principles based on the scenario to meet the requirements.

## Requirements for Principles
(1) Principles are presented from most important to least important.
(2) Principles should be as critical as possible.
(3) Each principle should consist of a brief phrase accompanied by a single sentence description.
(4) The number of principles should be LESS THAN OR EQUAL TO {number}.
(5) Duplicate principles should be eliminated.

## Scenario
{scenario}

## Principles
{principles}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class BasePrincipleFilter(BaseModel):
    """Main class for filtering evaluation principles.

    Attributes:
        llm: Language model client for generating responses
        scenario: Task context description
        filter_number: Number of principles to cluster in final output
    """

    llm: BaseLLM = Field(default=..., description="llm client")
    filter_number: int = Field(default=10, description="number of filtered principles")
    max_retries: int = Field(default=3, description="max retries")

    def run(self, principles: Dict[str, str], scenario: str) -> Dict[str, str]:
        """Filter principles across scenario.

        Args:
            principles: Dictionary of principles to filter
            scenario: Task context description

        Returns:
            Dictionary of clustered principles
        """

        # Get filtered principles from LLM
        @retry(tries=self.max_retries, delay=1.0)
        def call():
            response = self.llm.simple_chat(
                PrincipleFilterTemplate.format(
                    scenario=scenario,
                    enable_thinking=self.llm.enable_thinking,
                    number=self.filter_number,
                ),
                sys_prompt="You are a skilled professional assistant focusing on filtering.",
            )
            result = PrincipleFilterTemplate.parse(response)
            logger.info("===FILTER RESULT===\n" + result.model_dump_json())
            return result.principles

        try:
            principles = call()
        except Exception as e:
            principles = {}
            logger.error(f"API call failed: {str(e)}")
        return principles
