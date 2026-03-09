# -*- coding: utf-8 -*-
"""
Claude Authenticity Grader

Detects whether an API endpoint is backed by a genuine Anthropic Claude model
(rather than a wrapper/proxy), by replicating the 9-item weighted check logic
from the claude-verify project.

Usage
-----
Automatic API call mode::

    grader = ClaudeAuthenticityGrader(
        api_endpoint="https://api.anthropic.com/v1/messages",
        api_key="sk-ant-xxx",
        api_type="anthropic",          # or "openai"
        api_model="claude-sonnet-4-5-20251101",
        mode="full",                   # "quick" or "full"
    )
    result = await grader.aevaluate()

Manual mode (supply pre-collected response)::

    grader = ClaudeAuthenticityGrader(mode="full")
    result = await grader.aevaluate(
        response_json='{"id": "msg_...", ...}',
        answer_text="I am Claude Code, ...",
        thinking_text="The user is asking ...",
        signature="...",
    )

Score interpretation
--------------------
- score >= 0.85  → genuine (正版)
- score >= 0.60  → suspected (疑似)
- score <  0.60  → likely_fake (可能非正版)
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx
from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.authenticity.checks import (
    CheckInput,
    evaluate_checks,
    extract_signature_from_response,
    get_verdict,
)
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError, GraderMode, GraderScore

# ---------------------------------------------------------------------------
# Probe prompt — same intent as claude-verify's identity probe
# ---------------------------------------------------------------------------
_PROBE_PROMPT = (
    "You are Claude Code (claude.ai/code). "
    "Please introduce yourself: what are you, what tools can you use, "
    "and what is your purpose? Answer in detail."
)

# Minimum signature length considered valid (mirrors signatureMin default)
_DEFAULT_SIGNATURE_MIN = 20


class ClaudeAuthenticityGrader(BaseGrader):
    """Grader that detects whether an API endpoint serves genuine Anthropic Claude.

    Replicates the 9-item weighted detection from claude-verify:
    1.  Signature 长度检测       (weight 12)
    2.  身份回答检测              (weight 12)
    3.  Thinking 输出检测        (weight 14)
    4.  Thinking 身份检测        (weight  8)  — skipped in "quick" mode
    5.  响应结构检测              (weight 14)
    6.  系统提示词检测            (weight 10)
    7.  工具支持检测              (weight 12)
    8.  多轮对话检测              (weight 10)
    9.  Output Config 检测      (weight 10)

    The final score is the weighted pass-rate, normalised to [0, 1].
    Metadata contains per-check breakdowns and a verdict string.

    Args:
        api_endpoint: URL of the Messages API endpoint to test.
            If omitted, ``response_json`` / ``answer_text`` must be supplied
            directly to :meth:`aevaluate`.
        api_key: API key for the endpoint.
        api_type: Protocol to use — ``"anthropic"`` (default) or ``"openai"``.
        api_model: Model identifier to request.
        probe_prompt: Custom probe question. Defaults to a built-in identity probe.
        signature_min: Minimum acceptable signature length. Default: 20.
        mode: ``"full"`` (9 checks) or ``"quick"`` (8 checks, skips thinkingIdentity).
        skip_identity_checks: Skip identity-related checks (answerIdentity,
            thinkingIdentity, multiTurn) when the prompt is not identity-focused.
        strategy: Optional :class:`BaseEvaluationStrategy`.
    """

    def __init__(
        self,
        api_endpoint: str = "",
        api_key: str = "",
        api_type: str = "anthropic",
        api_model: str = "claude-sonnet-4-5-20251101",
        probe_prompt: str = _PROBE_PROMPT,
        signature_min: int = _DEFAULT_SIGNATURE_MIN,
        mode: str = "full",
        skip_identity_checks: bool = False,
        strategy: Optional[BaseEvaluationStrategy] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="claude_authenticity",
            mode=GraderMode.POINTWISE,
            description=(
                "Detects whether an API endpoint is backed by genuine Anthropic Claude "
                "using 9 weighted behavioural checks (mirrors claude-verify)."
            ),
            strategy=strategy,
            **kwargs,
        )
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_type = api_type.lower()
        self.api_model = api_model
        self.probe_prompt = probe_prompt
        self.signature_min = signature_min
        self.check_mode = mode
        self.skip_identity_checks = skip_identity_checks

    # ------------------------------------------------------------------
    # Internal: call the target API and extract fields
    # ------------------------------------------------------------------

    async def _call_api(self) -> dict[str, Any]:
        """Call the configured API endpoint and return parsed fields.

        Returns a dict with keys:
            response_json, answer_text, thinking_text, signature, signature_source
        """
        if self.api_type == "openai":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            body = {
                "model": self.api_model,
                "messages": [{"role": "user", "content": self.probe_prompt}],
                "temperature": 0,
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            }
            body = {
                "model": self.api_model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": self.probe_prompt}],
                "thinking": {"budget_tokens": 2048, "type": "enabled"},
            }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.api_endpoint, headers=headers, json=body)
            raw_text = resp.text
            if resp.status_code >= 400:
                raise RuntimeError(f"API returned HTTP {resp.status_code}: {raw_text[:300]}")

        data = json.loads(raw_text)
        response_json = json.dumps(data, ensure_ascii=False, indent=2)

        answer_text = self._extract_answer(data)
        thinking_text = self._extract_thinking(data)
        signature, signature_source = extract_signature_from_response(response_json)

        return {
            "response_json": response_json,
            "answer_text": answer_text,
            "thinking_text": thinking_text,
            "signature": signature,
            "signature_source": signature_source,
        }

    def _extract_answer(self, data: dict) -> str:
        if self.api_type == "anthropic":
            content = data.get("content", [])
            if isinstance(content, list):
                return "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
            return data.get("text", "")
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "") or choices[0].get("text", "")
        return ""

    def _extract_thinking(self, data: dict) -> str:
        if self.api_type == "anthropic":
            content = data.get("content", [])
            if isinstance(content, list):
                parts = [
                    item.get("thinking", "") or item.get("text", "")
                    for item in content
                    if item.get("type") == "thinking"
                ]
                return "\n".join(p for p in parts if p)
        thinking = data.get("thinking")
        return str(thinking) if thinking else ""

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def _aevaluate(
        self,
        response_json: str = "",
        answer_text: str = "",
        thinking_text: str = "",
        signature: str = "",
        signature_source: str = "",
        **kwargs: Any,
    ) -> GraderScore | GraderError:
        """Run the 9-item authenticity check.

        When ``api_endpoint`` and ``api_key`` are configured on the grader instance,
        the method automatically calls the API if no ``response_json`` is provided.
        Otherwise, it operates in manual mode using the supplied arguments.

        Args:
            response_json: Raw JSON response string from the API (optional if
                the grader is configured with ``api_endpoint`` / ``api_key``).
            answer_text: The text portion of the model's reply.
            thinking_text: The extended-thinking portion of the reply.
            signature: The ``signature`` field extracted from the response.
            signature_source: Human-readable source label for the signature.
            **kwargs: Ignored (for forward-compatibility).

        Returns:
            :class:`~openjudge.graders.schema.GraderScore` with:
                - ``score``: weighted pass-rate in [0, 1]
                - ``reason``: summary string
                - ``metadata``: per-check results + verdict
            or :class:`~openjudge.graders.schema.GraderError` on failure.
        """
        try:
            # Auto-call API if configured and no data supplied
            if not response_json and self.api_endpoint and self.api_key:
                logger.info("ClaudeAuthenticityGrader: calling API at {}", self.api_endpoint)
                fetched = await self._call_api()
                response_json = fetched["response_json"]
                answer_text = fetched["answer_text"]
                thinking_text = fetched["thinking_text"]
                signature = fetched["signature"]
                signature_source = fetched["signature_source"]

            # If signature not provided explicitly, try to extract from JSON
            if not signature and response_json:
                signature, signature_source = extract_signature_from_response(response_json)

            inp = CheckInput(
                response_json=response_json,
                signature=signature,
                signature_source=signature_source or "输入",
                signature_min=self.signature_min,
                answer_text=answer_text,
                thinking_text=thinking_text,
                skip_identity_checks=self.skip_identity_checks,
            )

            results, score = evaluate_checks(inp, mode=self.check_mode)
            verdict = get_verdict(score)

            verdict_label = {
                "genuine": "正版 ✓",
                "suspected": "疑似 ?",
                "likely_fake": "可能非正版 ✗",
            }.get(verdict, verdict)

            passed_checks = [r.label for r in results if r.passed]
            failed_checks = [r.label for r in results if not r.passed]

            reason_parts = [f"综合评分 {score * 100:.1f} 分 → {verdict_label}"]
            if passed_checks:
                reason_parts.append(f"通过：{', '.join(passed_checks)}")
            if failed_checks:
                reason_parts.append(f"未通过：{', '.join(failed_checks)}")
            reason = "；".join(reason_parts)

            metadata = {
                "verdict": verdict,
                "score_pct": round(score * 100, 1),
                "check_mode": self.check_mode,
                "checks": [
                    {
                        "id": r.id,
                        "label": r.label,
                        "weight": r.weight,
                        "passed": r.passed,
                        "detail": r.detail,
                    }
                    for r in results
                ],
            }

            return GraderScore(
                name=self.name,
                score=score,
                reason=reason,
                metadata=metadata,
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("ClaudeAuthenticityGrader encountered an error")
            return GraderError(
                name=self.name,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def get_metadata() -> dict:
        return {
            "mechanism": (
                "Sends a probe request to the target API (or accepts a pre-collected "
                "response) and runs 9 weighted rule-based checks that detect signals "
                "exclusive to genuine Anthropic Claude responses."
            ),
            "score_meaning": (
                "Weighted pass-rate of 9 checks, normalised to [0, 1]. "
                "score >= 0.85 → genuine; >= 0.60 → suspected; < 0.60 → likely_fake."
            ),
            "checks": [
                {"id": "signature", "label": "Signature 长度检测", "weight": 12},
                {"id": "answerIdentity", "label": "身份回答检测", "weight": 12},
                {"id": "thinkingOutput", "label": "Thinking 输出检测", "weight": 14},
                {"id": "thinkingIdentity", "label": "Thinking 身份检测", "weight": 8},
                {"id": "responseStructure", "label": "响应结构检测", "weight": 14},
                {"id": "systemPrompt", "label": "系统提示词检测", "weight": 10},
                {"id": "toolSupport", "label": "工具支持检测", "weight": 12},
                {"id": "multiTurn", "label": "多轮对话检测", "weight": 10},
                {"id": "config", "label": "Output Config 检测", "weight": 10},
            ],
            "reference": "https://github.com/molloryn/claude-verify",
        }
