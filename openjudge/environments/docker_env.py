# -*- coding: utf-8 -*-
"""Docker environment implementation for OpenJudge agent evaluation framework."""

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from openjudge.environments.base_env import BaseEnvironment


class DockerEnvironment(BaseEnvironment):
    """Docker-based execution environment implementation.

    This implementation uses Docker containers to provide isolated execution
    environments for agents.
    """

    def __init__(
        self,
        name: str,
        image: str = "python:3.11-slim",
        volumes: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[str, str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ):
        """Initialize the Docker environment.

        Args:
            name: Unique name for the environment
            image: Docker image to use
            volumes: Volume mappings for the container
            ports: Port mappings for the container
            environment_vars: Environment variables for the container
            **kwargs: Additional configuration parameters
        """
        super().__init__(name, **kwargs)
        self.image = image
        self.volumes = volumes or {}
        self.ports = ports or {}
        self.environment_vars = environment_vars or {}
        self.container_id: Optional[str] = None
        self._is_running = False

    async def start(self) -> None:
        """Start the Docker container."""
        # Clean up any existing container with the same name
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

        cmd = ["docker", "run", "-d", "--name", self.name]

        # Add volume mounts
        for host_path, container_path in self.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Add port mappings
        for host_port, container_port in self.ports.items():
            cmd.extend(["-p", f"{host_port}:{container_port}"])

        # Add environment variables
        for key, value in self.environment_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self.image)
        # Keep container alive
        cmd.extend(["sleep", "infinity"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.container_id = result.stdout.strip()
            self._is_running = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start Docker container: {e.stderr}")

    async def stop(self) -> None:
        """Stop and remove the Docker container."""
        if self.container_id:
            try:
                subprocess.run(["docker", "stop", self.name], capture_output=True, check=True)
                subprocess.run(["docker", "rm", self.name], capture_output=True, check=True)
                self._is_running = False
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to stop Docker container: {e.stderr}")

    async def execute_command(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a command in the Docker container.

        Args:
            command: The command to execute
            timeout: Optional timeout in seconds

        Returns:
            Dictionary containing the execution result
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        # Use bash -c to properly handle shell commands with quotes, pipes, etc.
        cmd = ["docker", "exec", self.name, "bash", "-c", command]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                wait_timeout = timeout if timeout else 600
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {wait_timeout} seconds")

            return {
                "stdout": stdout.decode(errors="replace") if stdout else "",
                "stderr": stderr.decode(errors="replace") if stderr else "",
                "returncode": process.returncode,
                "success": process.returncode == 0
            }
        except TimeoutError:
            raise
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "success": False
            }

    async def copy_to(self, source: Path, destination: str) -> bool:
        """Copy a file from host to container."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        cmd = ["docker", "cp", str(source), f"{self.name}:{destination}"]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    async def copy_from(self, source: str, destination: Path) -> bool:
        """Copy a file from container to host."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        cmd = ["docker", "cp", f"{self.name}:{source}", str(destination)]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file in the container."""
        # Create a temporary file on host and copy to container
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            success = await self.copy_to(Path(temp_path), path)
            Path(temp_path).unlink()
            return success
        except:
            Path(temp_path).unlink()
            return False

    async def read_file(self, path: str) -> Optional[str]:
        """Read content from a file in the container."""
        # Copy file to temporary location and read
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            success = await self.copy_from(path, Path(temp_path))
            if success:
                with open(temp_path, 'r') as f:
                    content = f.read()
                Path(temp_path).unlink()
                return content
            else:
                Path(temp_path).unlink()
                return None
        except:
            Path(temp_path).unlink()
            return None

    @property
    def is_running(self) -> bool:
        """Check if the container is running."""
        return self._is_running