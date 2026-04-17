# -*- coding: utf-8 -*-
"""Init module for OpenJudge agent evaluation framework."""

# Use lazy imports to avoid loading heavy optional dependencies at package import time.
# Core lightweight components are imported eagerly; everything else is available via
# explicit submodule imports (e.g. `from openjudge.evaluation.agent_runner import ...`).

from openjudge.agents.base_agent import BaseAgent, BaseInstalledAgent
from openjudge.agents.factory import AgentFactory
from openjudge.agents.name import AgentName
from openjudge.environments.base_env import BaseEnvironment
from openjudge.environments.factory import EnvironmentFactory
from openjudge.evaluation.task_executor import AgentEvaluator, TaskExecutor


def __getattr__(name: str):
    """Lazy-load heavy components on first access."""
    if name == "AgentEvaluationRunner":
        from openjudge.evaluation.agent_runner import AgentEvaluationRunner
        return AgentEvaluationRunner
    raise AttributeError(f"module 'openjudge' has no attribute {name!r}")


__all__ = [
    "BaseAgent",
    "BaseInstalledAgent",
    "AgentFactory",
    "AgentName",
    "BaseEnvironment",
    "EnvironmentFactory",
    "AgentEvaluationRunner",
    "AgentEvaluator",
    "TaskExecutor",
]