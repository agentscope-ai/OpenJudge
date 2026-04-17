# -*- coding: utf-8 -*-
"""Task executor for running agent evaluations."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from openjudge.agents.base_agent import BaseAgent
from openjudge.environments.base_env import BaseEnvironment


class TaskExecutor:
    """Executor for running agent evaluations on specific tasks."""

    def __init__(self, environment: BaseEnvironment):
        """Initialize the task executor.

        Args:
            environment: The execution environment for running tasks
        """
        self.environment = environment

    async def execute_task(self, agent: BaseAgent, instruction: str) -> Dict[str, Any]:
        """Execute a single task with the given agent.

        Args:
            agent: The agent to run the task
            instruction: The task instruction

        Returns:
            Dictionary containing the execution result
        """
        try:
            # Setup the agent in the environment
            await agent.setup(self.environment)

            # Run the agent with the instruction
            result = await agent.run(instruction, self.environment)

            # Teardown the agent
            await agent.teardown(self.environment)

            return {
                "success": result.get("success", False),
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "metrics": result.get("metrics", {}),
                "agent_name": agent.name
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "metrics": {},
                "agent_name": agent.name
            }

    async def execute_batch_tasks(
        self,
        agents: List[BaseAgent],
        instructions: List[str]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tasks with multiple agents.

        Args:
            agents: List of agents to run
            instructions: List of task instructions

        Returns:
            List of execution results for each (agent, instruction) pair
        """
        results = []

        for agent in agents:
            for instruction in instructions:
                result = await self.execute_task(agent, instruction)
                results.append(result)
                # Add a small delay between executions to avoid overwhelming
                await asyncio.sleep(1)

        return results


class AgentEvaluator:
    """Evaluator for comparing agent performance."""

    def __init__(self, environment: BaseEnvironment):
        """Initialize the agent evaluator.

        Args:
            environment: The execution environment for running evaluations
        """
        self.environment = environment
        self.task_executor = TaskExecutor(environment)

    async def evaluate_agents(
        self,
        agents: List[BaseAgent],
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple agents on multiple tasks.

        Args:
            agents: List of agents to evaluate
            tasks: List of tasks, each with 'instruction' and optional 'expected_output'

        Returns:
            List of evaluation results
        """
        results = []

        for task in tasks:
            instruction = task.get("instruction", "")
            expected_output = task.get("expected_output", "")

            for agent in agents:
                result = await self.task_executor.execute_task(agent, instruction)

                # Add expected output and calculate basic metrics
                result["expected_output"] = expected_output
                result["task_instruction"] = instruction

                # Calculate basic success metric based on presence of output
                result["basic_success"] = bool(result["output"].strip())

                results.append(result)

        return results