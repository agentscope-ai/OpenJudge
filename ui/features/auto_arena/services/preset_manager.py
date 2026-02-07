# -*- coding: utf-8 -*-
"""Preset manager for Auto Arena configurations.

Handles saving, loading, importing, and exporting evaluation presets.
Configuration format is compatible with cookbooks/auto_arena config.yaml.

Note: This manager now uses workspace-based paths for multi-user isolation.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _get_workspace_presets_dir() -> Path:
    """Get the presets directory for the current workspace.

    Returns:
        Path to workspace-specific presets directory
    """
    try:
        from shared.services.workspace_manager import get_current_workspace_path

        workspace_path = get_current_workspace_path()
        return workspace_path / "presets" / "auto_arena"
    except Exception:
        # Fallback to default if workspace not available
        return Path.home() / ".openjudge_studio" / "presets" / "auto_arena"


class PresetManager:
    """Manages Auto Arena configuration presets.

    Presets are stored in the workspace directory for multi-user isolation.
    The format is compatible with cookbooks config.yaml for interoperability.
    """

    # Preset naming rules: alphanumeric, underscore, hyphen only
    NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
    MAX_NAME_LENGTH = 64

    # Built-in preset prefix (cannot be deleted by user)
    BUILTIN_PREFIX = "_builtin_"

    def __init__(self, presets_dir: str | Path | None = None):
        """Initialize the preset manager.

        Args:
            presets_dir: Custom presets directory. If None, uses workspace directory.
        """
        if presets_dir:
            self.presets_dir = Path(presets_dir)
        else:
            self.presets_dir = _get_workspace_presets_dir()

        # Ensure directory exists
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        # Create built-in presets if not exists
        self._ensure_builtin_presets()

    def _ensure_builtin_presets(self) -> None:
        """Create built-in example presets if they don't exist."""
        builtin_path = self.presets_dir / f"{self.BUILTIN_PREFIX}translation_demo.yaml"
        if not builtin_path.exists():
            builtin_preset = self._create_builtin_translation_demo()
            self._save_yaml(builtin_path, builtin_preset)

    def _create_builtin_translation_demo(self) -> dict[str, Any]:
        """Create the built-in translation demo preset."""
        return {
            "_ui_preset": {
                "name": f"{self.BUILTIN_PREFIX}translation_demo",
                "display_name": "📚 Translation Demo (DashScope)",
                "description": "English to Chinese translation evaluation using DashScope models",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "builtin": True,
            },
            "task": {
                "description": (
                    "English to Chinese translation assistant, "
                    "helping users translate various types of English content into fluent and accurate Chinese"
                ),
                "scenario": "Users need to translate English articles, documents, or text into Chinese",
            },
            "query_generation": {
                "num_queries": 20,
                "seed_queries": [
                    "Please translate the following paragraph into Chinese: "
                    "'The rapid advancement of artificial intelligence has transformed numerous industries.'",
                    "Translate this sentence to Chinese: "
                    "'Climate change poses significant challenges to global food security.'",
                ],
                "temperature": 0.9,
                "max_similarity": 0.85,
                "enable_evolution": False,
            },
            "target_endpoints": {
                "qwen-max": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": "",  # User needs to fill in
                    "model": "qwen-max",
                    "system_prompt": (
                        "You are a professional English-Chinese translator. "
                        "Provide accurate and fluent translations."
                    ),
                },
                "qwen-plus": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": "",  # User needs to fill in
                    "model": "qwen-plus",
                    "system_prompt": (
                        "You are a professional English-Chinese translator. "
                        "Provide accurate and fluent translations."
                    ),
                },
            },
            "judge_endpoint": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "",  # User needs to fill in
                "model": "qwen-max",
            },
            "evaluation": {
                "max_concurrency": 10,
            },
            "output": {
                "save_queries": True,
                "save_responses": True,
                "save_details": True,
            },
            "report": {
                "enabled": True,
                "chart": {
                    "enabled": True,
                },
            },
        }

    @classmethod
    def validate_name(cls, name: str) -> tuple[bool, str]:
        """Validate a preset name.

        Args:
            name: The preset name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Preset name cannot be empty"

        if len(name) > cls.MAX_NAME_LENGTH:
            return False, f"Preset name cannot exceed {cls.MAX_NAME_LENGTH} characters"

        if not cls.NAME_PATTERN.match(name):
            return False, "Preset name can only contain letters, numbers, underscores, and hyphens"

        if name.startswith(cls.BUILTIN_PREFIX):
            return False, f"Preset name cannot start with '{cls.BUILTIN_PREFIX}'"

        return True, ""

    def list_presets(self) -> list[dict[str, Any]]:
        """List all available presets.

        Returns:
            List of preset metadata dictionaries, sorted by name
        """
        presets = []

        for yaml_file in self.presets_dir.glob("*.yaml"):
            try:
                data = self._load_yaml(yaml_file)
                ui_preset = data.get("_ui_preset", {})

                preset_info = {
                    "name": yaml_file.stem,
                    "display_name": ui_preset.get("display_name", yaml_file.stem),
                    "description": ui_preset.get("description", ""),
                    "created_at": ui_preset.get("created_at", ""),
                    "updated_at": ui_preset.get("updated_at", ""),
                    "builtin": ui_preset.get("builtin", False),
                    "path": str(yaml_file),
                }
                presets.append(preset_info)
            except Exception:
                # Skip invalid files
                continue

        # Sort: built-in first, then by name
        presets.sort(key=lambda x: (not x["builtin"], x["name"].lower()))
        return presets

    def load_preset(self, name: str) -> dict[str, Any] | None:
        """Load a preset by name.

        Args:
            name: Preset name (without .yaml extension)

        Returns:
            Preset configuration dictionary, or None if not found
        """
        preset_path = self.presets_dir / f"{name}.yaml"
        if not preset_path.exists():
            return None

        return self._load_yaml(preset_path)

    def save_preset(self, name: str, config: dict[str, Any], overwrite: bool = False) -> tuple[bool, str]:
        """Save a preset.

        Args:
            name: Preset name
            config: Configuration dictionary
            overwrite: Whether to overwrite existing preset

        Returns:
            Tuple of (success, error_message)
        """
        # Validate name
        is_valid, error_msg = self.validate_name(name)
        if not is_valid:
            return False, error_msg

        preset_path = self.presets_dir / f"{name}.yaml"

        # Check if exists
        if preset_path.exists() and not overwrite:
            return False, f"Preset '{name}' already exists. Use 'Save' to overwrite."

        # Check if trying to overwrite builtin
        if preset_path.exists():
            existing = self._load_yaml(preset_path)
            if existing.get("_ui_preset", {}).get("builtin", False):
                return False, "Cannot overwrite built-in presets"

        # Add/update UI metadata
        now = datetime.now().isoformat()
        if "_ui_preset" not in config:
            config["_ui_preset"] = {}

        config["_ui_preset"]["name"] = name
        if "display_name" not in config["_ui_preset"]:
            config["_ui_preset"]["display_name"] = name
        if "created_at" not in config["_ui_preset"]:
            config["_ui_preset"]["created_at"] = now
        config["_ui_preset"]["updated_at"] = now
        config["_ui_preset"]["builtin"] = False

        # Save
        self._save_yaml(preset_path, config)
        return True, ""

    def delete_preset(self, name: str) -> tuple[bool, str]:
        """Delete a preset.

        Args:
            name: Preset name

        Returns:
            Tuple of (success, error_message)
        """
        preset_path = self.presets_dir / f"{name}.yaml"

        if not preset_path.exists():
            return False, f"Preset '{name}' not found"

        # Check if builtin
        existing = self._load_yaml(preset_path)
        if existing.get("_ui_preset", {}).get("builtin", False):
            return False, "Cannot delete built-in presets"

        preset_path.unlink()
        return True, ""

    def import_from_file(
        self, file_content: str | bytes, preset_name: str | None = None  # pylint: disable=unused-argument
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Import a preset from YAML file content.

        Args:
            file_content: YAML file content (string or bytes)
            preset_name: Optional name for the preset. If not provided, uses name from file.

        Returns:
            Tuple of (success, error_message, config_dict)
        """
        try:
            if isinstance(file_content, bytes):
                file_content = file_content.decode("utf-8")

            config = yaml.safe_load(file_content)

            if not isinstance(config, dict):
                return False, "Invalid YAML format: expected a dictionary", None

            # Validate required fields
            if "task" not in config:
                return False, "Invalid config: missing 'task' section", None

            if "target_endpoints" not in config:
                return False, "Invalid config: missing 'target_endpoints' section", None

            return True, "", config

        except yaml.YAMLError as e:
            return False, f"YAML parsing error: {e}", None
        except Exception as e:
            return False, f"Import error: {e}", None

    def export_to_yaml(self, config: dict[str, Any], include_ui_metadata: bool = False) -> str:
        """Export configuration to YAML string.

        Args:
            config: Configuration dictionary
            include_ui_metadata: Whether to include _ui_preset metadata

        Returns:
            YAML formatted string
        """
        export_config = config.copy()

        if not include_ui_metadata and "_ui_preset" in export_config:
            del export_config["_ui_preset"]

        return yaml.dump(
            export_config,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_yaml(self, path: Path, data: dict[str, Any]) -> None:
        """Save YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    # =========================================================================
    # Config Conversion: UI State <-> YAML Config
    # =========================================================================

    @staticmethod
    def ui_state_to_config(
        sidebar_config: dict[str, Any],
        panel_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert UI state to cookbooks-compatible config format.

        Args:
            sidebar_config: Configuration from sidebar (judge model, settings)
            panel_config: Configuration from config panel (task, endpoints, query settings)

        Returns:
            Cookbooks-compatible configuration dictionary
        """
        config: dict[str, Any] = {}

        # Task
        config["task"] = {
            "description": panel_config.get("task_description", ""),
            "scenario": panel_config.get("task_scenario", ""),
        }

        # Query generation
        config["query_generation"] = {
            "num_queries": sidebar_config.get("num_queries", 20),
            "seed_queries": panel_config.get("seed_queries", []),
            "temperature": panel_config.get("query_temperature", 0.9),
            "max_similarity": panel_config.get("max_similarity", 0.85),
            "enable_evolution": panel_config.get("enable_evolution", False),
        }

        # Target endpoints
        target_endpoints = {}
        for name, ep_config in panel_config.get("target_endpoints", {}).items():
            target_endpoints[name] = {
                "base_url": ep_config.get("base_url", ""),
                "api_key": ep_config.get("api_key", ""),
                "model": ep_config.get("model", ""),
            }
            if ep_config.get("system_prompt"):
                target_endpoints[name]["system_prompt"] = ep_config["system_prompt"]

        config["target_endpoints"] = target_endpoints

        # Judge endpoint
        config["judge_endpoint"] = {
            "base_url": sidebar_config.get("judge_endpoint", ""),
            "api_key": sidebar_config.get("judge_api_key", ""),
            "model": sidebar_config.get("judge_model", ""),
        }

        # Evaluation settings
        config["evaluation"] = {
            "max_concurrency": sidebar_config.get("max_concurrency", 10),
        }

        # Output settings
        config["output"] = {
            "save_queries": sidebar_config.get("save_queries", True),
            "save_responses": sidebar_config.get("save_responses", True),
            "save_details": sidebar_config.get("save_details", True),
        }

        # Report settings
        config["report"] = {
            "enabled": sidebar_config.get("generate_report", True),
            "chart": {
                "enabled": sidebar_config.get("generate_chart", True),
            },
        }

        return config

    @staticmethod
    def config_to_ui_state(config: dict[str, Any]) -> dict[str, Any]:
        """Convert cookbooks config to UI state format.

        Args:
            config: Cookbooks-compatible configuration dictionary

        Returns:
            Dictionary with keys for populating UI state
        """
        ui_state: dict[str, Any] = {}

        # Task
        task = config.get("task", {})
        ui_state["task_description"] = task.get("description", "")
        ui_state["task_scenario"] = task.get("scenario", "")

        # Query generation
        query_gen = config.get("query_generation", {})
        ui_state["num_queries"] = query_gen.get("num_queries", 20)
        ui_state["seed_queries"] = query_gen.get("seed_queries", [])
        ui_state["query_temperature"] = query_gen.get("temperature", 0.9)
        ui_state["max_similarity"] = query_gen.get("max_similarity", 0.85)
        ui_state["enable_evolution"] = query_gen.get("enable_evolution", False)

        # Target endpoints - convert from dict format to list format for UI
        target_endpoints = []
        for name, ep_config in config.get("target_endpoints", {}).items():
            target_endpoints.append(
                {
                    "name": name,
                    "base_url": ep_config.get("base_url", ""),
                    "api_key": ep_config.get("api_key", ""),
                    "model": ep_config.get("model", ""),
                    "system_prompt": ep_config.get("system_prompt", ""),
                }
            )
        ui_state["target_endpoints"] = target_endpoints

        # Judge endpoint
        judge = config.get("judge_endpoint", {})
        ui_state["judge_endpoint"] = judge.get("base_url", "")
        ui_state["judge_api_key"] = judge.get("api_key", "")
        ui_state["judge_model"] = judge.get("model", "")

        # Evaluation settings
        evaluation = config.get("evaluation", {})
        ui_state["max_concurrency"] = evaluation.get("max_concurrency", 10)

        # Output settings
        output = config.get("output", {})
        ui_state["save_queries"] = output.get("save_queries", True)
        ui_state["save_responses"] = output.get("save_responses", True)
        ui_state["save_details"] = output.get("save_details", True)

        # Report settings
        report = config.get("report", {})
        ui_state["generate_report"] = report.get("enabled", True)
        chart = report.get("chart", {})
        ui_state["generate_chart"] = chart.get("enabled", True) if isinstance(chart, dict) else True

        return ui_state
