#!/usr/bin/env python3
"""
Unified Label Studio Manager - automated deployment and management of Label Studio annotation service.

Provides Docker and pip-based deployment options with automatic service management,
health checking, and configuration persistence for annotation workflows.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger as loguru_logger


class LabelStudioManager:
    """
    Comprehensive Label Studio service manager supporting Docker and pip deployment modes.

    Handles automated installation, configuration, startup, health monitoring, and shutdown
    of Label Studio annotation service with persistent configuration and logging.

    Attributes:
        port: Port number for Label Studio web interface
        username: Default admin username for Label Studio
        password: Default admin password for Label Studio
        data_dir: Directory for persistent data and configuration storage
        use_docker: Flag to prefer Docker deployment over pip installation
        container_name: Docker container name for service management
        image: Docker image name and tag for Label Studio
        server_url: Constructed server URL for API access
        process: Process handle for pip-based deployment
        api_token: Authentication token for API access
        config_file: Path to persistent configuration file
    """

    def __init__(
        self,
        port: int = 8080,
        username: str = "admin@rmgallery.com",
        password: str = "RM-Gallery",
        data_dir: str = "./log/label_studio_logs",
        use_docker: bool = True,
        container_name: str = "rm-gallery-label-studio",
        image: str = "heartexlabs/label-studio:latest",
    ):
        self.port = port
        self.username = username
        self.password = password
        self.data_dir = data_dir
        self.use_docker = use_docker
        self.container_name = container_name
        self.image = image
        self.server_url = f"http://localhost:{port}"
        self.process = None
        self.api_token = None
        self.config_file = Path(self.data_dir) / "label_studio_config.json"

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging system"""
        # Create logs directory
        logs_dir = os.path.join(os.path.abspath(self.data_dir))
        os.makedirs(logs_dir, exist_ok=True)

        # Setup loguru logger
        loguru_logger.remove()  # Remove default handler

        # Add console handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
        )

        # Add file handler
        log_file = os.path.join(logs_dir, "label_studio_manager.log")
        loguru_logger.add(
            log_file,
            rotation="10 MB",
            retention="3 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
        )

    def start_service(self) -> bool:
        """Start Label Studio service"""
        try:
            # Check if service is already running
            if self._is_service_running():
                loguru_logger.info(
                    f"Label Studio service is already running at {self.server_url}"
                )
                return True

            loguru_logger.info(f"Starting Label Studio service on port {self.port}...")

            # Choose startup method
            if self.use_docker and self._check_docker_available():
                success = self._start_with_docker()
            else:
                if self.use_docker:
                    loguru_logger.warning(
                        "Docker not available, falling back to pip installation"
                    )
                    self.use_docker = (
                        False  # Update the flag to reflect actual deployment method
                    )
                success = self._start_with_pip()

            if success:
                loguru_logger.info("✅ Label Studio service started successfully!")
                loguru_logger.info(f"🌐 Web interface: {self.server_url}")

                # Save configuration
                self._save_config()
                return True
            else:
                loguru_logger.error("❌ Failed to start Label Studio service")
                return False

        except Exception as e:
            loguru_logger.error(f"Error starting Label Studio service: {e}")
            return False

    def _check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _start_with_docker(self) -> bool:
        """Start Label Studio with Docker"""
        loguru_logger.info("Starting Label Studio with Docker...")

        # Stop existing container
        if self._container_exists():
            self._stop_container()

        # Pull image
        loguru_logger.info(f"Pulling Docker image: {self.image}...")
        result = subprocess.run(["docker", "pull", self.image], check=False)
        if result.returncode != 0:
            loguru_logger.error(f"Failed to pull Docker image: {self.image}")
            return False

        # Create data volume
        volume_name = f"{self.container_name}-data"
        subprocess.run(["docker", "volume", "create", volume_name], check=False)

        # Start container
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--restart",
            "unless-stopped",
            "-p",
            f"{self.port}:8080",
            "-v",
            f"{volume_name}:/label-studio/data",
            "-e",
            f"LABEL_STUDIO_USERNAME={self.username}",
            "-e",
            f"LABEL_STUDIO_PASSWORD={self.password}",
            "-e",
            "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true",
            self.image,
        ]

        loguru_logger.info(f"Starting container: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            loguru_logger.error(f"Failed to start container: {result.stderr}")
            return False

        # Wait for service to be ready
        return self._wait_for_service()

    def _start_with_pip(self) -> bool:
        """Start Label Studio with pip"""
        loguru_logger.info("Starting Label Studio with pip...")

        # Check if label-studio is installed using importlib (more elegant way)
        loguru_logger.info("Checking if Label Studio is installed...")
        try:
            import importlib.metadata

            version = importlib.metadata.version("label-studio")
            loguru_logger.info(f"Label Studio already installed: {version}")
        except importlib.metadata.PackageNotFoundError:
            loguru_logger.info("Label Studio not found, installing...")
            loguru_logger.info(
                "Installing Label Studio... This may take a few minutes."
            )

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "label-studio"],
                    check=False,
                    timeout=300,  # 5 minutes timeout for installation
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    loguru_logger.error(
                        f"Failed to install Label Studio: {result.stderr}"
                    )
                    return False
                else:
                    loguru_logger.info("Label Studio installed successfully!")
            except subprocess.TimeoutExpired:
                loguru_logger.error(
                    "Timeout while installing Label Studio (>5 minutes)"
                )
                return False
            except Exception as e:
                loguru_logger.error(f"Error during Label Studio installation: {e}")
                return False
        except Exception as e:
            loguru_logger.error(f"Error checking Label Studio installation: {e}")
            return False

        # Create data directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # Start command
        cmd = [
            "label-studio",
            "start",
            "--port",
            str(self.port),
            "--data-dir",
            self.data_dir,
            "--username",
            self.username,
            "--password",
            self.password,
        ]

        loguru_logger.info(f"Running command: {' '.join(cmd)}")

        # Start service
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for service to be ready
        return self._wait_for_service()

    def _is_service_running(self) -> bool:
        """Check if service is already running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _wait_for_service(self, max_wait: int = 30) -> bool:
        """Wait for service to be ready"""
        start_time = time.time()

        loguru_logger.info("Waiting for Label Studio to initialize...")

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    loguru_logger.info("Label Studio is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            # 检查Docker容器是否仍在运行
            if self.use_docker and self._check_docker_available():
                if not self._container_running():
                    loguru_logger.error("Container stopped unexpectedly")
                    self._show_container_logs()
                    return False

            time.sleep(5)

        loguru_logger.error("Label Studio failed to start within timeout")
        return False

    def _save_config(self):
        """Save configuration to file"""
        config = {
            "server_url": self.server_url,
            "username": self.username,
            "data_dir": self.data_dir,
            "use_docker": self.use_docker,
            "container_name": self.container_name if self.use_docker else None,
            "status": "running",
        }

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        loguru_logger.info(f"💾 Configuration saved to {self.config_file}")

    def _container_exists(self) -> bool:
        """Check if container exists"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return self.container_name in result.stdout.strip().split("\n")
        except:
            return False

    def _container_running(self) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return self.container_name in result.stdout.strip().split("\n")
        except:
            return False

    def _stop_container(self):
        """Stop and delete container"""
        if self._container_exists():
            loguru_logger.info(f"Stopping container: {self.container_name}...")
            subprocess.run(["docker", "stop", self.container_name], check=False)
            loguru_logger.info(f"Removing container: {self.container_name}...")
            subprocess.run(["docker", "rm", self.container_name], check=False)

    def _show_container_logs(self):
        """Show container logs"""
        try:
            result = subprocess.run(
                ["docker", "logs", self.container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                loguru_logger.error(
                    f"Container logs:\n{result.stdout}\n{result.stderr}"
                )
        except:
            pass

    def _stop_pip_service(self):
        """Stop pip-based Label Studio service"""
        stopped = False

        # Method 1: Try to stop the process if we have it
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                loguru_logger.info("Label Studio service stopped (via process object)")
                stopped = True
            except Exception as e:
                loguru_logger.warning(f"Failed to stop via process object: {e}")

        # Method 2: Find and kill by port usage (most reliable)
        if not stopped:
            try:
                # Find process using the port
                result = subprocess.run(
                    ["lsof", "-ti", f":{self.port}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    loguru_logger.info(
                        f"Found processes using port {self.port}: {pids}"
                    )

                    # Kill each process
                    for pid in pids:
                        if pid.strip():
                            try:
                                subprocess.run(
                                    ["kill", "-TERM", pid.strip()], check=False
                                )
                                loguru_logger.info(f"Sent SIGTERM to process {pid}")
                                stopped = True
                            except Exception as e:
                                loguru_logger.warning(
                                    f"Failed to kill process {pid}: {e}"
                                )

                    # Wait a moment, then force kill if needed
                    if stopped:
                        time.sleep(3)
                        # Check if processes are still running
                        result2 = subprocess.run(
                            ["lsof", "-ti", f":{self.port}"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if result2.returncode == 0 and result2.stdout.strip():
                            loguru_logger.warning(
                                "Processes still using port, force killing..."
                            )
                            for pid in result2.stdout.strip().split("\n"):
                                if pid.strip():
                                    subprocess.run(
                                        ["kill", "-9", pid.strip()], check=False
                                    )
                                    loguru_logger.info(f"Force killed process {pid}")
                else:
                    loguru_logger.info(f"No processes found using port {self.port}")

            except Exception as e:
                loguru_logger.error(f"Error finding processes by port: {e}")

        # Method 3: Find by command pattern
        if not stopped:
            try:
                # Try different patterns
                patterns = [
                    "annotation.server start",
                    "label-studio start",
                    "rm_gallery.core.data.annotation.server",
                ]

                for pattern in patterns:
                    result = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split("\n")
                        loguru_logger.info(
                            f"Found processes matching '{pattern}': {pids}"
                        )

                        for pid in pids:
                            if pid.strip():
                                try:
                                    subprocess.run(
                                        ["kill", "-TERM", pid.strip()], check=False
                                    )
                                    loguru_logger.info(f"Sent SIGTERM to process {pid}")
                                    stopped = True
                                except Exception as e:
                                    loguru_logger.warning(
                                        f"Failed to kill process {pid}: {e}"
                                    )

                        if stopped:
                            break

                if not stopped:
                    loguru_logger.info("No matching processes found by command pattern")

            except Exception as e:
                loguru_logger.error(f"Error finding processes by pattern: {e}")

        # Final check
        if self._is_service_running():
            loguru_logger.warning(
                "⚠️  Service may still be running, please check manually"
            )
            loguru_logger.info("You can check with: ps aux | grep label-studio")
            loguru_logger.info("Or check port usage: lsof -i :{}".format(self.port))
        else:
            if stopped:
                loguru_logger.info("✅ Label Studio service stopped successfully")
            else:
                loguru_logger.info("✅ Label Studio service was not running")

    def stop_service(self):
        """Stop service"""
        # Determine actual deployment method from config file or dynamic detection
        actual_use_docker = self._get_actual_deployment_method()

        if actual_use_docker:
            loguru_logger.info("Stopping Docker-based Label Studio service...")
            self._stop_container()
        else:
            loguru_logger.info("Stopping pip-based Label Studio service...")
            self._stop_pip_service()

        # Clear configuration file
        try:
            if self.config_file.exists():
                self.config_file.unlink()
                loguru_logger.info("Configuration file removed")
            else:
                loguru_logger.info("Configuration file not found")
        except Exception as e:
            loguru_logger.error(f"Error removing configuration file: {e}")

    def _get_actual_deployment_method(self) -> bool:
        """Get actual deployment method from config file or dynamic detection"""
        # Try to read from config file first
        config = self.load_config()
        if config and "use_docker" in config:
            loguru_logger.debug(
                f"Found deployment method in config: {'Docker' if config['use_docker'] else 'Pip'}"
            )
            return config["use_docker"]

        # Fallback to dynamic detection
        loguru_logger.debug("Config not found, using dynamic detection")

        # Check if Docker container exists
        if self._container_exists():
            loguru_logger.debug("Found Docker container, assuming Docker deployment")
            return True

        # Check if pip service is running
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                loguru_logger.debug("Found process using port, assuming pip deployment")
                return False
        except:
            pass

        # Default fallback
        loguru_logger.debug("Could not determine deployment method, using default")
        return self.use_docker

    def get_status(self) -> dict:
        """Get service status"""
        config = self.load_config()
        is_running = self._is_service_running()
        actual_use_docker = self._get_actual_deployment_method()

        # Additional status info for pip mode
        pip_processes = []
        port_processes = []
        if not actual_use_docker:
            try:
                # Check by port
                result = subprocess.run(
                    ["lsof", "-ti", f":{self.port}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    port_processes = result.stdout.strip().split("\n")

                # Check by command pattern
                patterns = [
                    "annotation.server start",
                    "rm_gallery.core.data.annotation.server",
                    "label-studio start",  # Match actual label-studio process
                ]
                for pattern in patterns:
                    result = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        pip_processes.extend(result.stdout.strip().split("\n"))
            except:
                pass

        status = {
            "running": is_running,
            "config": config,
            "deployment_method": "docker" if actual_use_docker else "pip",
            "port": self.port,
            "server_url": self.server_url,
        }

        if not actual_use_docker:
            if pip_processes:
                status["pip_processes"] = pip_processes
            if port_processes:
                status["port_processes"] = port_processes

        return status

    def load_config(self) -> Optional[dict]:
        """Load configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config = json.load(f)
                    loguru_logger.debug(f"Loaded configuration from {self.config_file}")
                    return config
            else:
                loguru_logger.debug(f"Configuration file {self.config_file} not found")
                return None
        except json.JSONDecodeError as e:
            loguru_logger.error(f"Invalid JSON in config file: {e}")
            return None
        except Exception as e:
            loguru_logger.error(f"Error loading config: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Label Studio Unified Manager")
    parser.add_argument(
        "action", choices=["start", "stop", "status"], help="Action to perform"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port for Label Studio service"
    )
    parser.add_argument("--username", default=None, help="Admin username")
    parser.add_argument("--password", default=None, help="Admin password")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory for Label Studio",
    )
    parser.add_argument(
        "--use-pip", action="store_true", help="Use pip instead of Docker"
    )
    parser.add_argument(
        "--container-name",
        default=None,
        help="Docker container name",
    )
    parser.add_argument("--image", default=None, help="Docker image to use")

    args = parser.parse_args()

    # Build kwargs dict, only including non-None values to use class defaults
    kwargs = {}
    if args.port is not None:
        kwargs["port"] = args.port
    if args.username is not None:
        kwargs["username"] = args.username
    if args.password is not None:
        kwargs["password"] = args.password
    if args.data_dir is not None:
        kwargs["data_dir"] = args.data_dir
    if args.container_name is not None:
        kwargs["container_name"] = args.container_name
    if args.image is not None:
        kwargs["image"] = args.image

    # Handle use_docker flag (always set based on --use-pip)
    kwargs["use_docker"] = not args.use_pip

    manager = LabelStudioManager(**kwargs)

    if args.action == "start":
        if manager.start_service():
            loguru_logger.info("🎉 Label Studio is ready for use!")
            print("\n" + "=" * 60)
            print("🚀 Label Studio Successfully Started!")
            print("=" * 60)
            print(f"🌐 Web Interface: {manager.server_url}")
            print(f"📧 Username: {manager.username}")
            print(f"🔐 Password: {manager.password}")
            print(f"📁 Data Directory: {manager.data_dir}")
            print(f"🐳 Deployment: {'Docker' if manager.use_docker else 'Pip'}")
            print("=" * 60)
        else:
            loguru_logger.error("❌ Failed to start Label Studio")
            sys.exit(1)

    elif args.action == "stop":
        manager.stop_service()
        loguru_logger.info("Service stopped")

    elif args.action == "status":
        status = manager.get_status()
        print("\n" + "=" * 50)
        print("📊 Label Studio Status")
        print("=" * 50)
        print(f"🌐 Server URL: {status['server_url']}")
        print(f"🚀 Deployment: {status['deployment_method'].upper()}")
        print(f"🔌 Port: {status['port']}")
        print(f"{'✅ Running' if status['running'] else '❌ Stopped'}")

        if status["config"]:
            print(f"📁 Data Dir: {status['config']['data_dir']}")
            print(f"👤 Username: {status['config']['username']}")
        else:
            print("⚠️  No configuration file found")

        if status.get("pip_processes"):
            print(f"🔄 Process PIDs: {', '.join(status['pip_processes'])}")
        if status.get("port_processes"):
            print(f"🔌 Port PIDs: {', '.join(status['port_processes'])}")

        print("=" * 50)

        # Additional diagnostic info
        if not status["running"] and status["config"]:
            print("\n💡 Service appears to be stopped but config exists.")
            print("   Try running 'python server.py start' to restart.")
        elif status["running"] and not status["config"]:
            print("\n⚠️  Service is running but no config found.")
            print("   This might be a manually started instance.")


if __name__ == "__main__":
    main()
