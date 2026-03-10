# -*- coding: utf-8 -*-
"""CLI entry point for Claude Authenticity verification.

Only requires: httpx  (pip install httpx)
No other dependencies needed — all logic is in core.py.

Usage:
    # Single model (native format)
    python -m cookbooks.claude_authenticity \
        --endpoint https://your-provider.com/v1/messages \
        --api_key sk-xxx \
        claude-sonnet-4-6

    # Multiple models in parallel
    python -m cookbooks.claude_authenticity \
        --endpoint https://your-provider.com/v1/messages \
        --api_key sk-xxx \
        claude-sonnet-4-6 claude-opus-4-6 claude-sonnet-4-6-thinking

    # OpenAI-compatible endpoint
    python -m cookbooks.claude_authenticity \
        --endpoint https://your-provider.com/v1/chat/completions \
        --api_key sk-xxx \
        --api_type openai \
        claude-sonnet-4-6

    # Also extract injected system prompt
    python -m cookbooks.claude_authenticity \
        --endpoint https://your-provider.com/v1/messages \
        --api_key sk-xxx \
        claude-sonnet-4-6 \
        --extract_prompt

    # Quick mode (skip Thinking 身份 check)
    python -m cookbooks.claude_authenticity \
        --endpoint ... --api_key ... --mode quick \
        claude-sonnet-4-6

    # Skip identity checks (when provider overrides Claude identity)
    python -m cookbooks.claude_authenticity \
        --endpoint ... --api_key ... --skip_identity \
        claude-sonnet-4-6
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Support both `python -m cookbooks.claude_authenticity` and
# `python cookbooks/claude_authenticity/__main__.py` invocations.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cookbooks.claude_authenticity.core import (  # noqa: E402
    AuthenticityResult,
    check_authenticity,
    extract_system_prompt,
)

VERDICT_ZH = {
    "genuine": "正版 ✓",
    "suspected": "疑似 ?",
    "likely_fake": "非正版 ✗",
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(model: str, result: AuthenticityResult) -> None:
    verdict = VERDICT_ZH.get(result.verdict, result.verdict)
    print(f"\n{'=' * 60}")
    print(f"模型: {model}")
    print(f"{'=' * 60}")
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    print(f"  综合得分: {result.score * 100:.1f} 分   判定: {verdict}")
    print()
    for c in result.checks:
        status = "✓" if c.passed else "✗"
        print(f"  [{status}] (权重{c.weight:2d}) {c.label}: {c.detail}")


def _print_extraction(model: str, extractions: list) -> None:
    print(f"\n{'=' * 60}")
    print(f"System Prompt 提取结果 — {model}")
    print(f"{'=' * 60}")
    for label, thinking, reply in extractions:
        print(f"\n  [{label}]")
        if thinking:
            preview = thinking[:300].replace("\n", " ")
            print(f"    thinking: {preview}")
        print(f"    reply:    {reply[:500]}")


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


async def _run(
    endpoint: str,
    api_key: str,
    models: list,
    api_type: str = "anthropic",
    mode: str = "full",
    skip_identity: bool = False,
    extract_prompt: bool = False,
) -> None:
    print(f"Testing {len(models)} model(s) in parallel on {endpoint}", file=sys.stderr)

    auth_tasks = [check_authenticity(endpoint, api_key, m, api_type, mode, skip_identity) for m in models]
    auth_results = await asyncio.gather(*auth_tasks, return_exceptions=True)

    print("\n" + "=" * 60)
    print(f"{'模型':<40} {'得分':>6}  判定")
    print("=" * 60)
    for model, result in zip(models, auth_results):
        if isinstance(result, Exception):
            print(f"{model:<40}  EXCEPTION: {result}")
            continue
        verdict = VERDICT_ZH.get(result.verdict, "?")
        score_str = f"{result.score * 100:5.1f}分"
        print(f"{model:<40} {score_str}  {verdict}")

    for model, result in zip(models, auth_results):
        if isinstance(result, Exception):
            continue
        _print_summary(model, result)

    if extract_prompt:
        print("\n\n" + "#" * 60)
        print("# System Prompt Extraction")
        print("#" * 60)
        extract_tasks = [extract_system_prompt(endpoint, api_key, m, api_type) for m in models]
        extract_results = await asyncio.gather(*extract_tasks, return_exceptions=True)
        for model, extractions in zip(models, extract_results):
            if isinstance(extractions, Exception):
                print(f"\n{model}: EXCEPTION: {extractions}")
                continue
            _print_extraction(model, extractions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify whether API endpoints serve genuine Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m cookbooks.claude_authenticity \\
      --endpoint https://your-provider.com/v1/messages \\
      --api_key sk-xxx \\
      claude-sonnet-4-6 claude-opus-4-6

  python -m cookbooks.claude_authenticity \\
      --endpoint https://your-provider.com/v1/messages \\
      --api_key sk-xxx \\
      --extract_prompt \\
      claude-sonnet-4-6
        """,
    )
    parser.add_argument("--endpoint", required=True, help="API endpoint URL (e.g. https://xxx/v1/messages)")
    parser.add_argument("--api_key", required=True, help="API key for the endpoint")
    parser.add_argument("models", nargs="+", help="One or more model IDs to test")
    parser.add_argument(
        "--api_type",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="Protocol format: anthropic (default) or openai",
    )
    parser.add_argument(
        "--mode", default="full", choices=["full", "quick"], help="full=9 checks (default), quick=skip Thinking 身份"
    )
    parser.add_argument("--skip_identity", action="store_true", help="Skip identity keyword checks")
    parser.add_argument("--extract_prompt", action="store_true", help="Also attempt system prompt extraction")

    args = parser.parse_args()
    asyncio.run(
        _run(
            args.endpoint,
            args.api_key,
            args.models,
            args.api_type,
            args.mode,
            args.skip_identity,
            args.extract_prompt,
        )
    )


if __name__ == "__main__":
    main()
