# -*- coding: utf-8 -*-
"""Agent evaluation runner for OpenJudge agent evaluation framework."""

import asyncio
from typing import Any, Dict, List

from openjudge.evaluation.agent_grader import AgentPerformanceGrader
from openjudge.evaluation.task_executor import AgentEvaluator
from openjudge.runner.grading_runner import GradingRunner, GraderConfig
from openjudge.agents.name import AgentName


class AgentEvaluationRunner:
    """Runner for evaluating agents using the OpenJudge framework."""

    def __init__(self, environment, model):
        """Initialize the agent evaluation runner.

        Args:
            environment: The execution environment for running agents
            model: The language model for evaluation
        """
        self.environment = environment
        self.model = model
        self.agent_evaluator = AgentEvaluator(environment)
        self.performance_grader = AgentPerformanceGrader(model)

    async def run_single_agent_evaluation(
        self,
        agent_name: AgentName,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run evaluation for a single agent on multiple tasks.

        Args:
            agent_name: Name of the agent to evaluate
            tasks: List of tasks to run

        Returns:
            List of evaluation results
        """
        from openjudge.agents.factory import AgentFactory

        # Create the agent
        agent = AgentFactory.create_agent(agent_name)

        # Run the agent on all tasks
        results = await self.agent_evaluator.evaluate_agents([agent], tasks)

        # Grade each result
        graded_results = []
        for result in results:
            grade = await self.performance_grader.aevaluate(
                task_instruction=result["task_instruction"],
                agent_output=result["output"],
                expected_output=result.get("expected_output", ""),
                agent_name=result["agent_name"]
            )
            result["grade"] = grade
            graded_results.append(result)

        return graded_results

    async def run_multi_agent_evaluation(
        self,
        agent_names: List[AgentName],
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run evaluation for multiple agents on multiple tasks.

        Args:
            agent_names: List of agent names to evaluate
            tasks: List of tasks to run

        Returns:
            Dictionary mapping agent names to their evaluation results
        """
        from openjudge.agents.factory import AgentFactory

        # Create all agents
        agents = [AgentFactory.create_agent(name) for name in agent_names]

        # Run all agents on all tasks
        results = await self.agent_evaluator.evaluate_agents(agents, tasks)

        # Group results by agent
        agent_results = {}
        for result in results:
            agent_name = result["agent_name"]
            if agent_name not in agent_results:
                agent_results[agent_name] = []
            agent_results[agent_name].append(result)

        # Grade each result
        for agent_name, agent_results_list in agent_results.items():
            for result in agent_results_list:
                grade = await self.performance_grader.aevaluate(
                    task_instruction=result["task_instruction"],
                    agent_output=result["output"],
                    expected_output=result.get("expected_output", ""),
                    agent_name=result["agent_name"]
                )
                result["grade"] = grade

        return agent_results

    async def run_comparative_evaluation(
        self,
        agent_names: List[AgentName],
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run comparative evaluation of multiple agents on the same tasks.

        Args:
            agent_names: List of agent names to compare
            tasks: List of tasks to run

        Returns:
            List of comparative evaluation results
        """
        from openjudge.agents.factory import AgentFactory
        from openjudge.evaluation.agent_grader import AgentComparisonGrader
        from openjudge.runner.grading_runner import GradingRunner, GraderConfig

        comparison_grader = AgentComparisonGrader(self.model)

        results = []

        for task in tasks:
            task_instruction = task.get("instruction", "")

            # Run each agent on the task
            task_results = []
            for agent_name in agent_names:
                agent = AgentFactory.create_agent(agent_name)
                result = await self.agent_evaluator.task_executor.execute_task(agent, task_instruction)
                result["agent_name"] = agent_name.value
                task_results.append(result)

            # Grade the comparison
            grade = await comparison_grader.aevaluate(
                task_instruction=task_instruction,
                agent_outputs=task_results
            )

            results.append({
                "task_instruction": task_instruction,
                "agent_results": task_results,
                "comparison_grade": grade
            })

        return results

    async def run_openjudge_style_evaluation(
        self,
        agent_names: List[AgentName],
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run evaluation using OpenJudge's grading runner pattern.

        Args:
            agent_names: List of agent names to evaluate
            tasks: List of tasks to run

        Returns:
            Dictionary containing comprehensive evaluation results
        """
        from openjudge.agents.factory import AgentFactory
        from openjudge.runner.grading_runner import GradingRunner, GraderConfig

        # Create agents
        agents = [AgentFactory.create_agent(name) for name in agent_names]

        # Prepare dataset for grading runner
        dataset = []
        for task in tasks:
            for agent in agents:
                dataset.append({
                    "task_instruction": task.get("instruction", ""),
                    "expected_output": task.get("expected_output", ""),
                    "agent_output": "",  # Will be filled by evaluation
                    "agent_name": agent.name
                })

        # Create a custom grader that runs the agent and evaluates
        class AgentExecutionGrader(AgentPerformanceGrader):
            def __init__(self, model, agent, environment, **kwargs):
                super().__init__(model, **kwargs)
                self.agent = agent
                self.environment = environment

            async def _aevaluate(self, task_instruction: str, expected_output: str = "", **kwargs):
                # Execute the agent on the task
                from openjudge.evaluation.task_executor import TaskExecutor
                executor = TaskExecutor(self.environment)
                result = await executor.execute_task(self.agent, task_instruction)

                # Call parent evaluation with the actual output
                return await super()._aevaluate(
                    task_instruction=task_instruction,
                    agent_output=result.get("output", ""),
                    expected_output=expected_output,
                    agent_name=self.agent.name
                )

        # Set up grading runner
        grader_configs = {}
        for i, agent in enumerate(agents):
            grader_name = f"agent_{i}_{agent.name}"
            grader_configs[grader_name] = GraderConfig(
                grader=AgentExecutionGrader(
                    model=self.model,
                    agent=agent,
                    environment=self.environment,
                    name=grader_name
                )
            )

        runner = GradingRunner(grader_configs=grader_configs)

        # Run the evaluation
        results = await runner.arun(dataset)

        return {
            "raw_results": results,
            "agent_names": [name.value for name in agent_names],
            "task_count": len(tasks)
        }