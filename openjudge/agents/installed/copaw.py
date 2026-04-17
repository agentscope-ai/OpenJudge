# -*- coding: utf-8 -*-
"""Copaw agent implementation for OpenJudge agent evaluation framework."""

import os
import shlex
from typing import Any, Dict

from openjudge.agents.base_agent import BaseInstalledAgent
from openjudge.environments.base_env import BaseEnvironment
from openjudge.agents.name import AgentName


class Copaw(BaseInstalledAgent):
    """Copaw agent implementation.

    This agent implements the Copaw agent for evaluation in the OpenJudge framework.
    It installs and configures Copaw in the execution environment.
    """

    def __init__(self, name: str = AgentName.COPAW.value, **kwargs: Any):
        """Initialize the Copaw agent.

        Args:
            name: Name of the agent (defaults to AgentName.COPAW.value)
            **kwargs: Additional configuration parameters
        """
        super().__init__(name, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install Copaw in the environment.

        Args:
            environment: The execution environment
        """
        # Install system dependencies
        await environment.execute_command(
            "apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip git curl"
        )

        # Install QwenPaw (formerly CoPaw; still provides copaw alias)
        await environment.execute_command(
            "pip install qwenpaw --quiet 2>&1 | tail -5 || pip install copaw --quiet 2>&1 | tail -5",
            timeout=180,
        )

    async def run(self, instruction: str, environment: BaseEnvironment) -> Dict[str, Any]:
        """Run Copaw with the given instruction.

        Args:
            instruction: The task instruction for Copaw to execute
            environment: The execution environment

        Returns:
            Dictionary containing the result of the Copaw execution
        """
        escaped_instruction = shlex.quote(instruction)

        # Determine model, API key and base URL
        model = self.config.get('model', 'qwen3.5-plus')
        api_key = self.config.get('api_key') or os.environ.get("OPENAI_API_KEY", "")
        base_url = self.config.get('base_url') or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Write qwenpaw provider config files directly (non-interactive setup)
        setup_cmd = f"""set -e
mkdir -p ~/.qwenpaw.secret/providers/custom
cat > ~/.qwenpaw.secret/providers/custom/openjudge-custom.json << 'PROVIDER_EOF'
{{
  "id": "openjudge-custom",
  "name": "OpenJudge Custom Endpoint",
  "base_url": "{base_url}",
  "api_key": "{api_key}",
  "chat_model": "OpenAIChatModel",
  "models": [{{"id": "{model}", "name": "{model}"}}],
  "extra_models": [],
  "api_key_prefix": "",
  "is_local": false,
  "freeze_url": false,
  "require_api_key": true,
  "is_custom": true,
  "support_model_discovery": false,
  "support_connection_check": false,
  "generate_kwargs": {{}},
  "meta": {{}}
}}
PROVIDER_EOF
cat > ~/.qwenpaw.secret/providers/active_model.json << 'ACTIVE_EOF'
{{
  "provider_id": "openjudge-custom",
  "model": "{model}"
}}
ACTIVE_EOF
echo "QwenPaw provider config written successfully"
"""

        await environment.execute_command(setup_cmd, timeout=60)

        # Run QwenPaw headlessly with the task instruction
        run_cmd = (
            f"export OPENAI_API_KEY='{api_key}' && "
            f"export OPENAI_BASE_URL='{base_url}' && "
            f"timeout 280s qwenpaw task -i {escaped_instruction} --no-guard "
            f"2>&1 | tee /tmp/copaw_output.txt || true"
        )

        result = await environment.execute_command(run_cmd, timeout=300)

        # Get the actual output from the file if command succeeded
        output_content = await environment.read_file("/tmp/copaw_output.txt") or result["stdout"]

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
        """Clean up after Copaw execution.

        Args:
            environment: The execution environment
        """
        # Clean up temporary files
        await environment.execute_command("rm -f /tmp/copaw_output.txt")

    @property
    def version(self) -> str:
        """Return the version of Copaw."""
        # Would need to execute version command in environment
        return "unknown"