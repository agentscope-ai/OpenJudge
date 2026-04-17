# -*- coding: utf-8 -*-
"""Agent grader for evaluating agent performance using OpenJudge evaluation framework."""

from typing import Any, Dict, List

from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
from openjudge.models.base_chat_model import BaseChatModel


class AgentPerformanceGrader(BaseGrader):
    """Grader for evaluating agent performance on tasks.

    This grader evaluates agent outputs against expected outputs or
    evaluates the quality of agent responses using various metrics.
    """

    def __init__(
        self,
        model: BaseChatModel,
        name: str = "agent-performance",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "Evaluates agent performance on tasks",
        **kwargs: Any
    ):
        """Initialize the agent performance grader.

        Args:
            model: The language model to use for evaluation
            name: Name of the grader
            mode: Evaluation mode (pointwise or listwise)
            description: Description of the grader
            **kwargs: Additional configuration
        """
        super().__init__(name=name, mode=mode, description=description, **kwargs)
        self.model = model

    async def _aevaluate(
        self,
        task_instruction: str,
        agent_output: str,
        expected_output: str = "",
        agent_name: str = "",
        **kwargs: Any
    ) -> GraderScore:
        """Evaluate agent performance.

        Args:
            task_instruction: The original task instruction
            agent_output: The output from the agent
            expected_output: The expected/correct output (optional)
            agent_name: Name of the agent being evaluated
            **kwargs: Additional parameters

        Returns:
            GraderScore with the performance score
        """
        # Evaluate the agent's performance using the model
        evaluation_messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Assess the quality of the agent's response to the given task. Score from 1-5 where 5 is excellent."
            },
            {
                "role": "user",
                "content": f"""Task: {task_instruction}

Agent Response: {agent_output}

Expected Response: {expected_output if expected_output else 'No expected output provided'}

Evaluate the agent's response based on:
1. Accuracy/completeness relative to the task
2. Quality of the response
3. If expected output is provided, how closely it matches

Provide your evaluation in JSON format:
{{
    "score": <number>,
    "reason": "<brief explanation>"
}}"""
            }
        ]

        try:
            response = await self.model.achat(messages=evaluation_messages)

            # Try to parse the JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', str(response), re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
                score = eval_data.get('score', 3.0)
                reason = eval_data.get('reason', 'No reason provided')
            else:
                score = 3.0
                reason = f"Could not parse evaluation response: {response}"

        except Exception as e:
            score = 2.0
            reason = f"Evaluation error: {str(e)}. Defaulting to score of 2.0"

        # Create a composite score based on various factors
        metadata = {
            "accuracy_score": score,
            "agent_name": agent_name,
            "task_instruction_length": len(task_instruction),
            "agent_output_length": len(agent_output),
            "expected_output_provided": bool(expected_output),
        }

        return GraderScore(
            name=self.name,
            score=float(score),
            reason=reason,
            metadata=metadata
        )


class AgentComparisonGrader(BaseGrader):
    """Grader for comparing multiple agents on the same task."""

    def __init__(
        self,
        model: BaseChatModel,
        name: str = "agent-comparison",
        mode: GraderMode = GraderMode.LISTWISE,
        description: str = "Compares multiple agents on the same task",
        **kwargs: Any
    ):
        """Initialize the agent comparison grader.

        Args:
            model: The language model to use for evaluation
            name: Name of the grader
            mode: Evaluation mode (should be LISTWISE for comparison)
            description: Description of the grader
            **kwargs: Additional configuration
        """
        super().__init__(name=name, mode=GraderMode.LISTWISE, description=description, **kwargs)
        self.model = model

    async def _aevaluate(
        self,
        task_instruction: str,
        agent_outputs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> "GraderRank":  # type: ignore
        """Compare multiple agent outputs for the same task.

        Args:
            task_instruction: The task instruction that all agents attempted
            agent_outputs: List of dictionaries containing agent outputs and metadata
            **kwargs: Additional parameters

        Returns:
            GraderRank with the ranking of agents
        """
        from openjudge.graders.schema import GraderRank

        # Prepare a comparison request to the model
        comparison_input = f"Task: {task_instruction}\n\n"

        for i, output_data in enumerate(agent_outputs):
            agent_output = output_data.get('output', '')
            agent_name = output_data.get('agent_name', f'Agent_{i}')
            comparison_input += f"Agent {i+1} ({agent_name}): {agent_output}\n\n"

        comparison_input += "Rank the agents from best (1) to worst based on how well they addressed the task. Respond with a JSON array of rankings like [1, 3, 2] where position corresponds to agent order."

        evaluation_messages = [
            {
                "role": "system",
                "content": "You are comparing multiple AI agents' responses to the same task. Rank them from best to worst."
            },
            {
                "role": "user",
                "content": comparison_input
            }
        ]

        try:
            response = await self.model.achat(messages=evaluation_messages)

            # Try to parse the ranking from response
            import json
            import re

            # Extract JSON array from response
            json_match = re.search(r'\[(.*?)\]', str(response))
            if json_match:
                ranking_str = json_match.group(1)
                ranking = [int(x.strip()) for x in ranking_str.split(',') if x.strip().isdigit()]
            else:
                # If no ranking found, default to [1, 2, 3, ...]
                ranking = list(range(1, len(agent_outputs) + 1))

            reason = f"Ranked agents based on quality of response to: {task_instruction[:100]}..."
        except Exception as e:
            ranking = list(range(1, len(agent_outputs) + 1))  # Default ranking
            reason = f"Comparison error: {str(e)}. Default ranking applied."

        return GraderRank(
            name=self.name,
            rank=ranking,
            reason=reason,
            metadata={
                "agent_names": [output.get('agent_name', f'Agent_{i}') for i, output in enumerate(agent_outputs)],
                "total_agents": len(agent_outputs)
            }
        )