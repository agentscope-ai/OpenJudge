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

Credentials reused automatically from the current UI are bound to the endpoint
URL they were entered for. On resume a bound key is applied only to the saved
endpoint with the same URL; when the URLs differ, the saved endpoint keeps its
own (possibly legacy) key or reports a clear missing-credential error instead
of silently receiving a key belonging to another provider.
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

# Keys used by ``bound_credentials_from_ui_config`` to pin each automatically
# reused credential to the endpoint URL it was entered for. They are in-memory
# only and never part of a saved config.
JUDGE_URL_BINDING = "judge_endpoint_url"
TARGET_URL_BINDINGS = "target_endpoint_urls"


def _normalize_url(url: Any) -> str:
    """Normalize an endpoint URL for comparison: trim, drop trailing '/', fold case."""
    if not isinstance(url, str):
        return ""
    return url.strip().rstrip("/").lower()


def _judge_url(config: Mapping[str, Any], *, normalize: bool = True) -> str:
    """Return the judge endpoint URL for either the flat or the nested shape."""
    judge = config.get("judge_endpoint")
    raw = judge.get("base_url") if isinstance(judge, Mapping) else judge
    if not isinstance(raw, str):
        return ""
    return _normalize_url(raw) if normalize else raw.strip()


def _target_urls(config: Mapping[str, Any], *, normalize: bool = True) -> dict[str, str]:
    """Return ``{endpoint name: base URL}`` for dict- and list-shaped endpoints."""
    urls: dict[str, str] = {}
    endpoints = config.get("target_endpoints")
    if isinstance(endpoints, Mapping):
        # The dict shape maps a name to an endpoint dict; a bare string value is
        # an API key, not a URL, so it carries no binding information.
        pairs = [
            (name, value.get("base_url") if isinstance(value, Mapping) else None) for name, value in endpoints.items()
        ]
    elif isinstance(endpoints, list):
        pairs = [
            (entry.get("name"), entry.get("base_url") if isinstance(entry, Mapping) else None)
            for entry in endpoints
            if isinstance(entry, Mapping)
        ]
    else:
        pairs = []

    for name, raw in pairs:
        if not name:
            continue
        url = raw.strip() if isinstance(raw, str) else ""
        urls[name] = _normalize_url(url) if normalize else url
    return urls


def _binding_matches(bound_url: Any, saved_url: str) -> bool:
    """Whether a URL-bound credential may be reused for a saved endpoint.

    A credential with no URL binding is an explicit ``credentials=`` value and
    is left to the caller. An automatically reused credential is only applied
    when the saved endpoint has a URL and both URLs normalize to the same value.
    """
    normalized = _normalize_url(bound_url)
    if not normalized:
        return True
    return bool(saved_url) and normalized == saved_url


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
    *,
    preserve_existing: bool = False,
) -> dict[str, Any]:
    """Return a copy of ``config`` with in-memory ``credentials`` applied.

    ``credentials`` accepts both config shapes and is deliberately forgiving:

    * ``judge_api_key`` or ``judge_endpoint.api_key`` for the judge key;
    * ``target_endpoints`` / ``target_endpoint_api_keys`` as a name-to-key
      mapping, a dict of endpoint dicts, or a list of endpoint dicts;
    * ``api_key`` (or ``target_api_key``) as a single fallback key applied to
      every endpoint without an explicit key.

    Empty values never overwrite a key already present in ``config``.

    Credentials produced by :func:`bound_credentials_from_ui_config` also carry
    ``judge_endpoint_url`` / ``target_endpoint_urls``. Such an automatically
    reused key is injected only when its URL matches the URL of the saved
    endpoint. A key for provider B therefore cannot land in provider A's
    request configuration, even when both runs use the same endpoint names.

    ``preserve_existing`` (used by ``PipelineRunner.resume``) additionally never
    replaces a non-empty key already stored in ``config``, so legacy runs keep
    the plaintext credentials they were originally saved with.
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

    saved_judge_url = _judge_url(merged)
    saved_target_urls = _target_urls(merged)
    bound_target_urls = credentials.get(TARGET_URL_BINDINGS)
    if not isinstance(bound_target_urls, Mapping):
        bound_target_urls = {}

    # A mismatched binding suppresses that endpoint's key entirely, including
    # the shared fallback, so nothing from the current UI leaks into the run.
    if not _binding_matches(credentials.get(JUDGE_URL_BINDING), saved_judge_url):
        judge_key = ""

    def _set(mapping: dict[str, Any], key: str, value: str) -> None:
        if preserve_existing and mapping.get(key):
            return
        mapping[key] = value

    def _target_key(name: Any) -> str:
        if not _binding_matches(bound_target_urls.get(name), saved_target_urls.get(name, "")):
            return ""
        return endpoint_keys.get(name) or fallback

    judge = merged.get("judge_endpoint")
    if isinstance(judge, dict):
        if judge_key:
            _set(judge, "api_key", judge_key)
    elif judge_key:
        _set(merged, "judge_api_key", judge_key)

    endpoints = merged.get("target_endpoints")
    if isinstance(endpoints, dict):
        for name, entry in endpoints.items():
            key = _target_key(name)
            if not key:
                continue
            if isinstance(entry, dict):
                _set(entry, "api_key", key)
            elif not entry:
                endpoints[name] = {"api_key": key}
    elif isinstance(endpoints, list):
        for entry in endpoints:
            if isinstance(entry, dict):
                key = _target_key(entry.get("name"))
                if key:
                    _set(entry, "api_key", key)

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


def mismatched_bindings(
    config: Mapping[str, Any],
    credentials: Mapping[str, Any] | None,
    *,
    applied: Mapping[str, Any] | None = None,
) -> list[str]:
    """Describe UI credentials bound to a different URL than the saved endpoint.

    ``applied`` is the config after :func:`apply_credentials`. When supplied,
    only endpoints that still need a credential are reported: a legacy run that
    already carries its own valid key does not need anything re-entered.
    """
    if not isinstance(credentials, Mapping):
        return []

    blank = set(missing_credentials(applied)) if applied is not None else None

    def _needed(label: str) -> bool:
        return blank is None or label in blank

    details: list[str] = []
    saved_judge_url = _judge_url(config)
    saved_judge_label = _judge_url(config, normalize=False) or "(no URL in saved config)"
    bound_judge_url = credentials.get(JUDGE_URL_BINDING)
    judge_label = (
        "judge_endpoint.api_key (judge model)"
        if isinstance(config.get("judge_endpoint"), Mapping)
        else "judge_api_key (judge model)"
    )
    if (
        isinstance(bound_judge_url, str)
        and _normalize_url(bound_judge_url)
        and not _binding_matches(bound_judge_url, saved_judge_url)
        and _needed(judge_label)
    ):
        details.append(f"judge model: current UI {bound_judge_url!r} != saved endpoint {saved_judge_label!r}")

    bound_target_urls = credentials.get(TARGET_URL_BINDINGS)
    if isinstance(bound_target_urls, Mapping):
        saved_target_urls = _target_urls(config)
        saved_target_labels = _target_urls(config, normalize=False)
        for name, url in bound_target_urls.items():
            saved_label = saved_target_labels.get(name) or "(no URL in saved config)"
            label = f"target_endpoints[{name!r}].api_key"
            if (
                isinstance(url, str)
                and _normalize_url(url)
                and not _binding_matches(url, saved_target_urls.get(name, ""))
                and _needed(label)
            ):
                details.append(f"target_endpoints[{name!r}]: current UI {url!r} != saved endpoint {saved_label!r}")
    return details


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


def bound_credentials_from_ui_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Credentials for automatic reuse on resume, pinned to their endpoint URLs.

    ``feature.py`` keeps this in session state instead of the bare keys. When a
    saved run used a different provider, the binding stops a key entered for
    the current UI from being injected into that run's request configuration;
    the user must supply a credential for the saved endpoint instead. No
    prompts, task data or model settings are included.
    """
    credentials = credentials_from_ui_config(config)
    credentials[JUDGE_URL_BINDING] = _judge_url(config, normalize=False)
    credentials[TARGET_URL_BINDINGS] = _target_urls(config, normalize=False)
    return credentials
