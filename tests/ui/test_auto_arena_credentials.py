# -*- coding: utf-8 -*-
"""Regression tests for Auto Arena credential handling.

Covers the three persistence/export paths (saved preset, exported YAML and
``ui_config.json``), preservation of in-memory credentials, and resuming both
legacy (plaintext) and sanitized configs.
"""

import json
from typing import Any

import pytest
from features.auto_arena.services.credentials import (
    apply_credentials,
    credentials_from_ui_config,
    missing_credentials,
    strip_secrets,
)
from features.auto_arena.services.pipeline_runner import PipelineRunner
from features.auto_arena.services.preset_manager import PresetManager

JUDGE_URL = "https://api.example.com/v1"
JUDGE_KEY = "sk-judge-9f2c"
KEY_A = "sk-model-a-71ab"
KEY_B = "sk-model-b-3d0e"
ALL_SECRETS = (JUDGE_KEY, KEY_A, KEY_B)


def flat_ui_config(output_dir: str = "") -> dict[str, Any]:
    """The shape written to ``ui_config.json``: judge_endpoint is a URL string."""
    return {
        "task_description": "Translate English to Chinese",
        "task_scenario": "Bilingual assistant",
        "judge_endpoint": JUDGE_URL,
        "judge_api_key": JUDGE_KEY,
        "judge_model": "judge-model",
        "target_endpoints": {
            "model-a": {"base_url": JUDGE_URL, "api_key": KEY_A, "model": "model-a"},
            "model-b": {"base_url": JUDGE_URL, "api_key": KEY_B, "model": "model-b"},
        },
        "num_queries": 20,
        "output_dir": output_dir,
    }


def nested_yaml_config(output_dir: str = "") -> dict[str, Any]:
    """The shape used by presets / cookbooks: judge_endpoint is a dict."""
    return {
        "task": {"description": "Translate English to Chinese", "scenario": "Bilingual assistant"},
        "output_dir": output_dir,
        "query_generation": {"num_queries": 20, "temperature": 0.9},
        "target_endpoints": {
            "model-a": {"base_url": JUDGE_URL, "api_key": KEY_A, "model": "model-a"},
            "model-b": {"base_url": JUDGE_URL, "api_key": KEY_B, "model": "model-b"},
        },
        "judge_endpoint": {"base_url": JUDGE_URL, "api_key": JUDGE_KEY, "model": "judge-model"},
        "evaluation": {"max_concurrency": 10},
    }


def _assert_no_secrets(content: str) -> None:
    for secret in ALL_SECRETS:
        assert secret not in content, f"{secret} leaked into persisted config"


def _write_ui_config(tmp_path, config: dict[str, Any]) -> None:
    (tmp_path / PipelineRunner.CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")


# ---------------------------------------------------------------------------
# strip_secrets: both config shapes
# ---------------------------------------------------------------------------


def test_strip_secrets_blanks_judge_api_key_in_flat_ui_config() -> None:
    config = flat_ui_config()
    scrubbed = strip_secrets(config)

    assert scrubbed["judge_api_key"] == ""
    assert scrubbed["judge_endpoint"] == JUDGE_URL  # a URL string, not a dict
    for name in ("model-a", "model-b"):
        assert scrubbed["target_endpoints"][name]["api_key"] == ""
        assert scrubbed["target_endpoints"][name]["model"] == name

    # Deep copy: the caller's in-memory config is untouched.
    assert config["judge_api_key"] == JUDGE_KEY
    assert config["target_endpoints"]["model-a"]["api_key"] == KEY_A
    scrubbed["target_endpoints"]["model-a"]["model"] = "changed"
    assert config["target_endpoints"]["model-a"]["model"] == "model-a"


def test_strip_secrets_blanks_nested_yaml_config_and_suffixed_keys() -> None:
    config = nested_yaml_config()
    config["api_token"] = "sk-service-token"
    config["service_api_key"] = "sk-service-key"
    config["target_endpoints"]["model-a"]["max_tokens"] = 4096

    scrubbed = strip_secrets(config)

    assert scrubbed["judge_endpoint"]["api_key"] == ""
    assert scrubbed["judge_endpoint"]["base_url"] == JUDGE_URL
    assert scrubbed["target_endpoints"]["model-a"]["api_key"] == ""
    assert scrubbed["target_endpoints"]["model-b"]["api_key"] == ""
    assert scrubbed["api_token"] == ""
    assert scrubbed["service_api_key"] == ""
    assert scrubbed["target_endpoints"]["model-a"]["max_tokens"] == 4096  # not a credential
    assert config["judge_endpoint"]["api_key"] == JUDGE_KEY


def test_strip_secrets_handles_list_shaped_endpoints_and_leaves_lookalikes() -> None:
    config = {
        "judge_endpoint": JUDGE_URL,
        "judge_api_key": JUDGE_KEY,
        "max_tokens": 4096,
        "api_key_env": "MY_KEY_NAME",
        "target_endpoints": [{"name": "model-a", "api_key": KEY_A, "model": "model-a"}],
    }
    scrubbed = strip_secrets(config)

    assert scrubbed["judge_api_key"] == ""
    assert scrubbed["target_endpoints"][0]["api_key"] == ""
    assert scrubbed["max_tokens"] == 4096
    assert scrubbed["api_key_env"] == "MY_KEY_NAME"


# ---------------------------------------------------------------------------
# The three persistence / export paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_factory", [flat_ui_config, nested_yaml_config])
def test_save_preset_never_writes_credentials(config_factory, tmp_path) -> None:
    manager = PresetManager(presets_dir=tmp_path)
    config = config_factory()

    ok, message = manager.save_preset("demo", config)
    assert ok, message
    _assert_no_secrets((tmp_path / "demo.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("config_factory", [flat_ui_config, nested_yaml_config])
def test_export_to_yaml_never_writes_credentials(config_factory, tmp_path) -> None:
    manager = PresetManager(presets_dir=tmp_path)
    config = config_factory()

    exported = manager.export_to_yaml(config)
    _assert_no_secrets(exported)
    # The caller's config is not mutated at all by the export path.
    if "judge_api_key" in config:
        assert config["judge_api_key"] == JUDGE_KEY
    else:
        assert config["judge_endpoint"]["api_key"] == JUDGE_KEY


@pytest.mark.parametrize("config_factory", [flat_ui_config, nested_yaml_config])
def test_save_config_never_writes_credentials(config_factory, tmp_path) -> None:
    config = config_factory(str(tmp_path))
    runner = PipelineRunner(config)

    runner.save_config()

    saved = (tmp_path / PipelineRunner.CONFIG_FILE).read_text(encoding="utf-8")
    _assert_no_secrets(saved)
    data = json.loads(saved)
    assert data["target_endpoints"]["model-a"]["api_key"] == ""
    if config_factory is flat_ui_config:
        assert data["judge_api_key"] == ""
        assert data["judge_endpoint"] == JUDGE_URL
    else:
        assert data["judge_endpoint"]["api_key"] == ""
        assert data["judge_endpoint"]["base_url"] == JUDGE_URL


def test_save_config_preserves_in_memory_credentials(tmp_path) -> None:
    runner = PipelineRunner(flat_ui_config(str(tmp_path)))

    runner.save_config()

    assert runner.config["judge_api_key"] == JUDGE_KEY
    assert runner.config["target_endpoints"]["model-a"]["api_key"] == KEY_A
    assert runner.config["target_endpoints"]["model-b"]["api_key"] == KEY_B
    assert runner.validate_config() == (True, "")


def test_credentials_from_ui_config_keeps_only_keys() -> None:
    creds = credentials_from_ui_config(flat_ui_config())
    assert set(creds) == {"judge_api_key", "target_endpoint_api_keys"}
    assert creds["judge_api_key"] == JUDGE_KEY
    assert creds["target_endpoint_api_keys"] == {"model-a": KEY_A, "model-b": KEY_B}
    assert JUDGE_URL not in str(creds)

    nested_creds = credentials_from_ui_config(nested_yaml_config())
    assert nested_creds["judge_api_key"] == JUDGE_KEY
    assert nested_creds["target_endpoint_api_keys"] == {"model-a": KEY_A, "model-b": KEY_B}


# ---------------------------------------------------------------------------
# resume(): legacy, sanitized, both shapes, and clear errors
# ---------------------------------------------------------------------------


def test_resume_legacy_config_with_plaintext_keys(tmp_path) -> None:
    """Existing saved runs store judge_endpoint as a URL string and must still resume."""
    _write_ui_config(tmp_path, flat_ui_config(str(tmp_path)))

    runner = PipelineRunner.resume(str(tmp_path))

    assert runner.config["judge_endpoint"] == JUDGE_URL
    assert runner.config["judge_api_key"] == JUDGE_KEY
    assert runner.validate_config() == (True, "")


def test_resume_sanitized_config_with_in_memory_credentials(tmp_path) -> None:
    config = flat_ui_config(str(tmp_path))
    PipelineRunner(config).save_config()

    runner = PipelineRunner.resume(str(tmp_path), credentials=credentials_from_ui_config(config))

    assert runner.config["judge_api_key"] == JUDGE_KEY
    assert runner.config["target_endpoints"]["model-a"]["api_key"] == KEY_A
    assert runner.config["target_endpoints"]["model-b"]["api_key"] == KEY_B
    assert runner.validate_config() == (True, "")


def test_resume_sanitized_config_without_credentials_raises_clear_error(tmp_path) -> None:
    PipelineRunner(flat_ui_config(str(tmp_path))).save_config()

    with pytest.raises(ValueError) as exc_info:
        PipelineRunner.resume(str(tmp_path))

    message = str(exc_info.value)
    assert "re-enter" in message.lower()
    assert "judge_api_key (judge model)" in message
    assert "target_endpoints['model-a'].api_key" in message
    assert "target_endpoints['model-b'].api_key" in message


def test_resume_partial_credentials_lists_only_what_is_missing(tmp_path) -> None:
    PipelineRunner(flat_ui_config(str(tmp_path))).save_config()

    with pytest.raises(ValueError) as exc_info:
        PipelineRunner.resume(str(tmp_path), credentials={"judge_api_key": JUDGE_KEY})

    message = str(exc_info.value)
    assert "judge_api_key (judge model)" not in message
    assert "target_endpoints['model-a'].api_key" in message
    assert "target_endpoints['model-b'].api_key" in message


def test_resume_nested_config_with_and_without_credentials(tmp_path) -> None:
    config = nested_yaml_config()
    config["output_dir"] = str(tmp_path)
    PipelineRunner(config).save_config()

    with pytest.raises(ValueError) as exc_info:
        PipelineRunner.resume(str(tmp_path))
    assert "judge_endpoint.api_key (judge model)" in str(exc_info.value)

    runner = PipelineRunner.resume(
        str(tmp_path),
        credentials={
            "judge_endpoint": {"api_key": JUDGE_KEY},
            "target_endpoints": {"model-a": {"api_key": KEY_A}, "model-b": KEY_B},
        },
    )
    assert runner.config["judge_endpoint"]["api_key"] == JUDGE_KEY
    assert runner.config["target_endpoints"]["model-a"]["api_key"] == KEY_A
    assert runner.config["target_endpoints"]["model-b"]["api_key"] == KEY_B


def test_resume_accepts_single_fallback_key(tmp_path) -> None:
    PipelineRunner(flat_ui_config(str(tmp_path))).save_config()

    runner = PipelineRunner.resume(str(tmp_path), credentials={"api_key": "sk-shared"})

    assert runner.config["judge_api_key"] == "sk-shared"
    assert runner.config["target_endpoints"]["model-a"]["api_key"] == "sk-shared"
    assert runner.config["target_endpoints"]["model-b"]["api_key"] == "sk-shared"


def test_resume_does_not_blank_legacy_keys_with_empty_credentials(tmp_path) -> None:
    _write_ui_config(tmp_path, flat_ui_config(str(tmp_path)))

    runner = PipelineRunner.resume(str(tmp_path), credentials={"judge_api_key": ""})

    assert runner.config["judge_api_key"] == JUDGE_KEY
    assert runner.config["target_endpoints"]["model-a"]["api_key"] == KEY_A


def test_resume_rejects_non_mapping_credentials(tmp_path) -> None:
    _write_ui_config(tmp_path, flat_ui_config(str(tmp_path)))

    with pytest.raises(TypeError, match="credentials must be a mapping"):
        PipelineRunner.resume(str(tmp_path), credentials="sk-not-a-mapping")


def test_resume_without_any_saved_config(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        PipelineRunner.resume(str(tmp_path))


def test_resume_results_file_fallback_reports_missing_keys(tmp_path) -> None:
    """evaluation_results.json only stores endpoint names, so keys are missing."""
    results = {
        "config": {
            "task": {"description": "Translate English to Chinese"},
            "target_endpoints": ["model-a", "model-b"],
            "num_queries": 20,
        },
    }
    (tmp_path / "evaluation_results.json").write_text(json.dumps(results), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        PipelineRunner.resume(str(tmp_path))

    message = str(exc_info.value)
    assert "judge_api_key (judge model)" in message
    assert "target_endpoints['model-a'].api_key" in message


# ---------------------------------------------------------------------------
# Helpers used directly by the UI wiring
# ---------------------------------------------------------------------------


def test_apply_credentials_never_treats_a_url_as_a_mapping() -> None:
    merged = apply_credentials(
        flat_ui_config(),
        {"judge_endpoint": JUDGE_URL, "judge_api_key": "sk-new-judge"},
    )
    assert merged["judge_endpoint"] == JUDGE_URL
    assert merged["judge_api_key"] == "sk-new-judge"


def test_missing_credentials_is_empty_for_a_complete_config() -> None:
    assert missing_credentials(flat_ui_config()) == []
    assert missing_credentials(nested_yaml_config()) == []
    assert missing_credentials(strip_secrets(nested_yaml_config())) == [
        "judge_endpoint.api_key (judge model)",
        "target_endpoints['model-a'].api_key",
        "target_endpoints['model-b'].api_key",
    ]
