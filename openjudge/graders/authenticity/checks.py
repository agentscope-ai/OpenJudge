# -*- coding: utf-8 -*-
"""
Claude Authenticity Checks

9-item weighted detection logic, mirroring the checks in claude-verify.
Each check receives a CheckInput and returns a CheckResult with pass/fail and detail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckInput:
    """All data needed to run the 9 authenticity checks."""
    response_json: str = ""
    signature: str = ""
    signature_source: str = ""
    signature_min: int = 20
    answer_text: str = ""
    thinking_text: str = ""
    skip_identity_checks: bool = False


@dataclass
class CheckResult:
    """Result of a single check."""
    id: str
    label: str
    weight: int
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Internal helpers (mirrors checks.ts helpers)
# ---------------------------------------------------------------------------

def _parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


_SIGNATURE_KEYS = {"signature", "sig", "x-claude-signature", "x_signature", "xsignature"}


def _find_signature_value(value: Any, depth: int = 0) -> str:
    if depth > 6:
        return ""
    if isinstance(value, list):
        for item in value:
            found = _find_signature_value(item, depth + 1)
            if found:
                return found
    if isinstance(value, dict):
        for k, v in value.items():
            if k.lower() in _SIGNATURE_KEYS and isinstance(v, str) and v.strip():
                return v
            found = _find_signature_value(v, depth + 1)
            if found:
                return found
    return ""


def extract_signature_from_response(raw_text: str) -> tuple[str, str]:
    """Return (signature_value, source_label). Mirrors extractSignatureFromResponse."""
    data = _parse_json_safe(raw_text)
    if data is None:
        return "", ""
    result = _find_signature_value(data)
    if result:
        return result, "响应JSON"
    return "", ""


# ---------------------------------------------------------------------------
# The 9 checks (mirrors checksConfig in checks.ts)
# ---------------------------------------------------------------------------

def check_signature(inp: CheckInput) -> CheckResult:
    length = len(inp.signature.strip())
    passed = length >= inp.signature_min
    return CheckResult(
        id="signature",
        label="Signature 长度检测",
        weight=12,
        passed=passed,
        detail=f"{inp.signature_source}长度 {length}，阈值 {inp.signature_min}",
    )


def check_answer_identity(inp: CheckInput) -> CheckResult:
    keywords = ["claude code", "cli", "命令行", "command", "terminal"]
    text = inp.answer_text.lower()
    passed = any(k in text for k in keywords)
    return CheckResult(
        id="answerIdentity",
        label="身份回答检测",
        weight=12,
        passed=passed,
        detail="包含关键身份词" if passed else "未发现关键身份词",
    )


def check_thinking_output(inp: CheckInput) -> CheckResult:
    text = inp.thinking_text.strip()
    passed = bool(text)
    detail = f"检测到 thinking 输出（{len(text)} 字符）" if passed else "响应中无 thinking 内容"
    return CheckResult(
        id="thinkingOutput",
        label="Thinking 输出检测",
        weight=14,
        passed=passed,
        detail=detail,
    )


def check_thinking_identity(inp: CheckInput) -> CheckResult:
    if not inp.thinking_text.strip():
        return CheckResult(
            id="thinkingIdentity",
            label="Thinking 身份检测",
            weight=8,
            passed=False,
            detail="未提供 thinking 文本",
        )
    keywords = ["claude code", "cli", "命令行", "command", "tool"]
    text = inp.thinking_text.lower()
    passed = any(k in text for k in keywords)
    return CheckResult(
        id="thinkingIdentity",
        label="Thinking 身份检测",
        weight=8,
        passed=passed,
        detail="包含 Claude Code/CLI 相关词" if passed else "未发现关键词",
    )


def check_response_structure(inp: CheckInput) -> CheckResult:
    data = _parse_json_safe(inp.response_json)
    if data is None:
        return CheckResult(
            id="responseStructure",
            label="响应结构检测",
            weight=14,
            passed=False,
            detail="JSON 无法解析",
        )
    usage = data.get("usage", {}) or {}
    has_id = "id" in data
    has_cache = "cache_creation" in data or "cache_creation" in usage
    has_tier = "service_tier" in data or "service_tier" in usage
    missing = []
    if not has_id:
        missing.append("id")
    if not has_cache:
        missing.append("cache_creation")
    if not has_tier:
        missing.append("service_tier")
    passed = has_id and has_cache
    detail = "关键字段齐全" if not missing else f"缺少字段：{', '.join(missing)}"
    return CheckResult(
        id="responseStructure",
        label="响应结构检测",
        weight=14,
        passed=passed,
        detail=detail,
    )


def check_system_prompt(inp: CheckInput) -> CheckResult:
    risky = ["system prompt", "ignore previous", "override", "越权"]
    text = f"{inp.answer_text} {inp.thinking_text}".lower()
    hit = any(k in text for k in risky)
    passed = not hit
    return CheckResult(
        id="systemPrompt",
        label="系统提示词检测",
        weight=10,
        passed=passed,
        detail="疑似提示词注入" if hit else "未发现异常提示词",
    )


def check_tool_support(inp: CheckInput) -> CheckResult:
    keywords = ["file", "command", "bash", "shell", "read", "write", "execute", "编辑", "读取", "写入", "执行"]
    text = inp.answer_text.lower()
    passed = any(k in text for k in keywords)
    return CheckResult(
        id="toolSupport",
        label="工具支持检测",
        weight=12,
        passed=passed,
        detail="包含工具能力描述" if passed else "未出现工具能力词",
    )


def check_multi_turn(inp: CheckInput) -> CheckResult:
    keywords = ["claude code", "cli", "command line", "工具"]
    text = f"{inp.answer_text}\n{inp.thinking_text}".lower()
    hits = sum(1 for k in keywords if k in text)
    passed = hits >= 2
    return CheckResult(
        id="multiTurn",
        label="多轮对话检测",
        weight=10,
        passed=passed,
        detail="多处确认身份" if passed else "确认次数偏少",
    )


def check_output_config(inp: CheckInput) -> CheckResult:
    data = _parse_json_safe(inp.response_json)
    if data is None:
        return CheckResult(
            id="config",
            label="Output Config 检测",
            weight=10,
            passed=False,
            detail="JSON 无法解析",
        )
    usage = data.get("usage", {}) or {}
    has_cache = "cache_creation" in data or "cache_creation" in usage
    has_tier = "service_tier" in data or "service_tier" in usage
    passed = has_cache or has_tier
    return CheckResult(
        id="config",
        label="Output Config 检测",
        weight=10,
        passed=passed,
        detail="配置字段存在" if passed else "未发现配置字段",
    )


# ---------------------------------------------------------------------------
# Ordered list of all checks (used by the grader)
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_signature,
    check_answer_identity,
    check_thinking_output,
    check_thinking_identity,
    check_response_structure,
    check_system_prompt,
    check_tool_support,
    check_multi_turn,
    check_output_config,
]

IDENTITY_CHECK_IDS = {"answerIdentity", "thinkingIdentity", "multiTurn"}


def evaluate_checks(
    inp: CheckInput,
    mode: str = "full",
) -> tuple[List[CheckResult], float]:
    """Run all checks and return (results, score_0_to_1).

    Args:
        inp: Check input data.
        mode: "quick" skips thinkingIdentity; "full" runs all 9.

    Returns:
        Tuple of (list of CheckResult, normalized score in [0.0, 1.0]).
    """
    active = list(ALL_CHECKS)

    if mode == "quick":
        active = [c for c in active if c.__name__ != "check_thinking_identity"]

    if inp.skip_identity_checks:
        active = [c for c in active if c(inp).id not in IDENTITY_CHECK_IDS]

    results: List[CheckResult] = [c(inp) for c in active]

    total_weight = sum(r.weight for r in results)
    gained_weight = sum(r.weight for r in results if r.passed)
    score = round(gained_weight / total_weight, 4) if total_weight > 0 else 0.0

    return results, score


def get_verdict(score: float) -> str:
    """Map 0-1 score to verdict string. Mirrors getVerdict in checks.ts."""
    pct = score * 100
    if pct >= 85:
        return "genuine"
    if pct >= 60:
        return "suspected"
    return "likely_fake"
