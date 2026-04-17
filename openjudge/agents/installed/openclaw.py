# -*- coding: utf-8 -*-
"""OpenClaw agent implementation for OpenJudge agent evaluation framework."""

import os
import shlex
from typing import Any, Dict

from openjudge.agents.base_agent import BaseInstalledAgent
from openjudge.environments.base_env import BaseEnvironment
from openjudge.agents.name import AgentName


class OpenClaw(BaseInstalledAgent):
    """OpenClaw agent implementation.

    This agent implements the OpenClaw agent for evaluation in the OpenJudge framework.
    It installs and configures OpenClaw in the execution environment.
    """

    def __init__(self, name: str = AgentName.OPENCLAW.value, **kwargs: Any):
        """Initialize the OpenClaw agent.

        Args:
            name: Name of the agent (defaults to AgentName.OPENCLAW.value)
            **kwargs: Additional configuration parameters
        """
        super().__init__(name, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install OpenClaw in the environment.

        Args:
            environment: The execution environment
        """
        # Install git; ensure Node.js 22+ is available (openclaw requires v22+)
        await environment.execute_command(
            "apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y git curl && "
            "node --version | grep -qE 'v(2[2-9]|[3-9][0-9])' || "
            "(curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs)",
            timeout=180,
        )

        # Install OpenClaw from npm registry
        await environment.execute_command("npm install -g openclaw@latest", timeout=180)

    async def run(self, instruction: str, environment: BaseEnvironment) -> Dict[str, Any]:
        """Run OpenClaw with the given instruction.

        Args:
            instruction: The task instruction for OpenClaw to execute
            environment: The execution environment

        Returns:
            Dictionary containing the result of the OpenClaw execution
        """
        escaped_instruction = shlex.quote(instruction)

        # Determine model, API key and base URL
        model = self.config.get('model', 'qwen3.5-plus')
        api_key = self.config.get('api_key') or os.environ.get("OPENAI_API_KEY", "")
        base_url = self.config.get('base_url') or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Configure openclaw with the custom endpoint via non-interactive onboard
        onboard_cmd = (
            f"export CUSTOM_API_KEY='{api_key}' && "
            f"openclaw onboard --non-interactive "
            f"--auth-choice custom-api-key "
            f"--custom-base-url '{base_url}' "
            f"--custom-model-id '{model}' "
            f"--custom-api-key '{api_key}' "
            f"--secret-input-mode plaintext "
            f"--custom-compatibility openai "
            f"--accept-risk "
            f"--skip-health "
            f"2>&1 | tee /tmp/openclaw-setup.txt || true"
        )

        onboard_result = await environment.execute_command(onboard_cmd, timeout=120)

        # Run the agent with the instruction using a fixed session ID
        run_cmd = (
            f"export OPENAI_API_KEY='{api_key}' && "
            f"export OPENAI_BASE_URL='{base_url}' && "
            f"export CUSTOM_API_KEY='{api_key}' && "
            f"timeout 280s openclaw agent --local --session-id openjudge-task --message {escaped_instruction} "
            f"2>&1 | tee /tmp/openclaw_output.txt || true"
        )

        result = await environment.execute_command(run_cmd, timeout=300)

        # Get the actual output from the file if command succeeded
        output_content = await environment.read_file("/tmp/openclaw_output.txt") or result["stdout"]

        return {
            "success": result["success"],
            "output": output_content,
            "error": result["stderr"],
            "returncode": result["returncode"],
            "metrics": {
                "execution_time": 0,  # Would need to track actual time
                "command": run_cmd
            }
        }

    async def teardown(self, environment: BaseEnvironment) -> None:
        """Clean up after OpenClaw execution.

        Args:
            environment: The execution environment
        """
        # Clean up temporary files
        await environment.execute_command("rm -f /tmp/openclaw_output.txt")

    @property
    def version(self) -> str:
        """Return the version of OpenClaw."""
        # Would need to execute version command in environment
        return "unknown"