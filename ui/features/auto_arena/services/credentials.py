# -*- coding: utf-8 -*-
"""Credential handling for Auto Arena configurations.

Two configuration shapes are in use in this codebase:

* **Flat UI config** (``ui_config.json`` and ``render_config_panel``): the
  judge key is ``judge_api_key``, while ``judge_endpoint`` is a URL *string*;
  ``target_endpoints`` maps endpoint names to endpoint dicts.
* **Nested YAML config** (presets and ``cookbooks/auto_arena/config.yaml``):
  ``judge_endpoint`` is a dict carrying its own ``api_key``, and
  ``target_endpoints`` is a dict of endpoint dicts.

Both shapes must be handled when scrubbing credentials before writing to disk
and when re-injecting credentials supplied in memory on resume.
"""

import copy
from typing import Any, Mapping

# Explicit names plus suffixes: ``judge_api_key`` lives at the top level of the
# flat UI config, ``api_key`` inside nested endpoint dicts. Keeping the list
# explicit avoids blanking unrelated settings such as ``max_tokens``.
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "judge_api_key",
        "api_token",
        "api_secret",
        "secret_key",
        "access_token",
        "authorization",
        "password",
    }
)
SENSITIVE_SUFFIXES = ("_api_key", "_api_token", "_api_secret", "_access_token")
SENSITIVE_KEY = "api_key"  # Backwards-compatible alias for the common case.


def is_secret_key(key: Any) -> bool:
    """Return whether a mapping key holds a credential value."""
    if not isinstance(key, str):
        return False
    normalized = key.lower()
    return normalized in SENSITIVE_KEYS or normalized.endswith(SENSITIVE_SUFFIXES)


def strip_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``config`` with every credential blanked out.

    Auto Arena configs are written to saved presets, exported YAML and
    ``ui_config.json`` in the evaluation output directory. Those files are
    routinely copied, archived or attached to bug reports, so no credential
    may reach them. The caller's in-memory config is left untouched, which
    keeps the current run (and ``PipelineRunner.config``) fully usable.
    """

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: ("" if is_secret_key(key) else _walk(value)) for key, value in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_walk(item) for item in node)
        return copy.deepcopy(node)

    return _walk(config)


def _collect_endpoint_keys(source: Any) -> dict[str, str]:
    """Extract ``{endpoint name: api key}`` from the supported input shapes."""
    keys: dict[str, str] = {}
    if isinstance(source, dict):
        pairs = list(source.items())
    elif isinstance(source, list):
        pairs = [(entry.get("name"), entry.get("api_key")) for entry in source if isinstance(entry, dict)]
    else:
        pairs = []

    for name, value in pairs:
        if isinstance(value, dict):
            value = value.get("api_key")
        if name and isinstance(value, str) and value:
            keys[name] = value
    return keys


def apply_credentials(
    config: dict[str, Any],
    credentials: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``config`` with in-memory ``credentials`` applied.

    ``credentials`` accepts both config shapes and is deliberately forgiving:

    * ``judge_api_key`` or ``judge_endpoint.api_key`` for the judge key;
    * ``target_endpoints`` / ``target_endpoint_api_keys`` as a name-to-key
      mapping, a dict of endpoint dicts, or a list of endpoint dicts;
    * ``api_key`` (or ``target_api_key``) as a single fallback key applied to
      every endpoint without an explicit key.

    Empty values never overwrite a key already present in ``config``, which
    keeps legacy saved runs (written before keys were stripped) resumable.
    """
    merged = copy.deepcopy(config)
    if not credentials:
        return merged
    if not isinstance(credentials, Mapping):
        raise TypeError("credentials must be a mapping of API keys, e.g. {'judge_api_key': 'sk-...'}.")

    fallback = credentials.get("api_key") or credentials.get("target_api_key")
    fallback = fallback if isinstance(fallback, str) else ""

    judge_key = credentials.get("judge_api_key")
    judge_credential = credentials.get("judge_endpoint")
    # A URL string is a valid value here (flat UI config); never call .get on it.
    if isinstance(judge_credential, Mapping):
        judge_key = judge_key or judge_credential.get("api_key")
    judge_key = judge_key if isinstance(judge_key, str) else ""
    judge_key = judge_key or fallback

    endpoint_keys: dict[str, str] = {}
    for source_key in ("target_endpoint_api_keys", "target_endpoints"):
        endpoint_keys.update(_collect_endpoint_keys(credentials.get(source_key)))

    judge = merged.get("judge_endpoint")
    if isinstance(judge, dict):
        if judge_key:
            judge["api_key"] = judge_key
    elif judge_key:
        merged["judge_api_key"] = judge_key

    endpoints = merged.get("target_endpoints")
    if isinstance(endpoints, dict):
        for name, entry in endpoints.items():
            key = endpoint_keys.get(name) or fallback
            if not key:
                continue
            if isinstance(entry, dict):
                entry["api_key"] = key
            elif not entry:
                endpoints[name] = {"api_key": key}
    elif isinstance(endpoints, list):
        for entry in endpoints:
            if isinstance(entry, dict):
                key = endpoint_keys.get(entry.get("name")) or fallback
                if key:
                    entry["api_key"] = key

    return merged


def missing_credentials(config: Mapping[str, Any]) -> list[str]:
    """Return readable names of the credentials still blank in ``config``."""
    missing: list[str] = []

    judge = config.get("judge_endpoint")
    if isinstance(judge, dict):
        if not judge.get("api_key"):
            missing.append("judge_endpoint.api_key (judge model)")
    elif not config.get("judge_api_key"):
        missing.append("judge_api_key (judge model)")

    endpoints = config.get("target_endpoints")
    if isinstance(endpoints, dict):
        pairs = list(endpoints.items())
    elif isinstance(endpoints, list):
        pairs = [(entry.get("name"), entry) for entry in endpoints if isinstance(entry, dict)]
        pairs += [(entry, None) for entry in endpoints if isinstance(entry, str)]
    else:
        pairs = []

    for name, entry in pairs:
        if isinstance(entry, dict):
            if not entry.get("api_key"):
                missing.append(f"target_endpoints[{name!r}].api_key")
        elif not entry:
            missing.append(f"target_endpoints[{name!r}].api_key")

    return missing


def credentials_from_ui_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only credentials from a config, for in-memory reuse on resume.

    The result contains no URLs, prompts or task data, so it is safe to keep in
    the Streamlit session state and to pass as ``credentials=`` to resume().
    """
    judge = config.get("judge_endpoint")
    if isinstance(judge, dict):
        judge_key = judge.get("api_key", "")
    else:
        judge_key = config.get("judge_api_key", "")

    return {
        "judge_api_key": judge_key if isinstance(judge_key, str) else "",
        "target_endpoint_api_keys": _collect_endpoint_keys(config.get("target_endpoints")),
    }
