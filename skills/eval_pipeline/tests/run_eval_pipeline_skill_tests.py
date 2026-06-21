# -*- coding: utf-8 -*-
"""
Functional tests for eval-pipeline skills.

This runner checks whether a skill can guide an agent through realistic user
requests. It is different from ``evaluate_skills.py``, which grades the quality
of a skill package itself.

Usage:
    python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py
    python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py --case-id rag_eval_001_diagnose_generation_problem

Environment (any OpenAI-compatible provider):
    OPENAI_API_KEY      preferred; the OpenAI-compatible key
    OPENAI_BASE_URL     optional, for OpenAI-compatible providers
    OPENAI_MODEL        optional actor model, default: qwen3.6-plus
    OPENAI_JUDGE_MODEL  optional judge model, default: qwen3-max (stronger than actor)

Aliyun DashScope (Bailian) fallback — if OPENAI_API_KEY is unset but
DASHSCOPE_API_KEY is present, the runner uses it against
https://dashscope.aliyuncs.com/compatible-mode/v1 automatically.

The actor follows the skill; a separate (stronger) judge grades the actor output
against the acceptance criteria. Using one model for both roles inflates pass rates,
so the actor and judge default to different models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# Paths are derived from this file's own location so the runner works wherever it
# lives. It sits at skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py:
#   HERE        -> skills/eval_pipeline/tests
#   SKILL_ROOT  -> skills/eval_pipeline   (the <NN-name>/SKILL.md dirs)
#   REPO_ROOT   -> repository root        (for .env)
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CASES = HERE / "eval_pipeline_test_cases.jsonl"
DEFAULT_SKILL_ROOT = HERE.parent
DEFAULT_OUT_DIR = HERE / "results"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return cases


def load_skill_text(skill_root: Path, skill_id: str) -> str:
    """Load a skill's text. Each skill is self-contained (Anthropic Agent Skill protocol),
    so a single SKILL.md is the full unit; if a skill later bundles its own
    <skill>/references/*.md, append those too (per-skill, so they travel on install)."""
    def with_bundled_refs(skill_dir: Path, text: str) -> str:
        ref_dir = skill_dir / "references"
        if not ref_dir.is_dir():
            return text
        extra = []
        for ref in sorted(ref_dir.glob("*.md")):
            extra.append(f"\n\n----- references/{ref.name} -----\n")
            extra.append(ref.read_text(encoding="utf-8"))
        return text + "".join(extra)

    if skill_id == "eval_pipeline_collection":
        parts = []
        for skill_file in sorted(skill_root.glob("*/SKILL.md")):
            parts.append(f"\n\n===== {skill_file.parent.name} =====\n")
            parts.append(with_bundled_refs(skill_file.parent, skill_file.read_text(encoding="utf-8")))
        return "".join(parts)

    skill_file = skill_root / skill_id / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"Missing skill file: {skill_file}")
    return with_bundled_refs(skill_file.parent, skill_file.read_text(encoding="utf-8"))


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        # Fallback: Aliyun DashScope (Bailian) OpenAI-compatible endpoint.
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        if dashscope_key:
            api_key = dashscope_key
            base_url = base_url or DASHSCOPE_BASE_URL
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY (or DASHSCOPE_API_KEY) in your shell or .env. "
            "For DashScope, the base URL defaults to " + DASHSCOPE_BASE_URL
        )
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_retries: int = 5,
) -> str:
    # DashScope/OpenAI-compatible endpoints emit transient 429s (per-minute token
    # buckets). Retry with exponential backoff so a burst limit doesn't fail a run.
    delay = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - retry on any transient API error
            status = getattr(exc, "status_code", None)
            is_transient = status in (429, 500, 502, 503, 504) or "429" in str(exc)
            if attempt == max_retries or not is_transient:
                raise
            print(f"    transient error (attempt {attempt}/{max_retries}): {str(exc)[:80]} — retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return ""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:500]}")
    return json.loads(match.group(0))


def run_actor(client: OpenAI, model: str, skill_text: str, case: dict[str, Any]) -> str:
    system = (
        "You are an evaluation practitioner using the provided Skill instructions. "
        "Follow the Skill closely. Do not claim you ran external systems unless the "
        "test case provides results. Produce the concrete next-step output that the "
        "user would need."
    )
    user = f"""
Use this Skill to answer the user request.

<SKILL>
{skill_text}
</SKILL>

<TEST_CASE>
id: {case["id"]}
skill: {case["skill"]}
category: {case["category"]}
user_request: {case["user_request"]}
artifacts: {json.dumps(case.get("artifacts", {}), ensure_ascii=False, indent=2)}
</TEST_CASE>

Write a concise but complete response. If evidence is insufficient, say so and
name the minimum missing artifact or sample size.
"""
    return call_model(
        client=client,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )


def judge_actor_output(client: OpenAI, model: str, case: dict[str, Any], actor_output: str) -> dict[str, Any]:
    system = (
        "You are a strict evaluator for eval-pipeline Skill functional tests. "
        "Grade only against the acceptance criteria and failure signals. "
        "Return only a JSON object."
    )
    user = f"""
Evaluate the actor output for this test case.

case_id: {case["id"]}
skill: {case["skill"]}
user_request: {case["user_request"]}

expected_behavior:
{json.dumps(case.get("expected_behavior", []), ensure_ascii=False, indent=2)}

acceptance_criteria:
{json.dumps(case.get("acceptance_criteria", []), ensure_ascii=False, indent=2)}

failure_signals:
{json.dumps(case.get("failure_signals", []), ensure_ascii=False, indent=2)}

actor_output:
<<<
{actor_output}
>>>

Return exactly:
{{
  "verdict": "pass" | "partial" | "fail",
  "score": <number from 0.0 to 1.0>,
  "met": ["criterion text that was met"],
  "missed": ["criterion text that was missed"],
  "failure_signals_observed": ["failure signal text observed"],
  "reason": "one concise paragraph"
}}
"""
    raw = call_model(
        client=client,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    result = extract_json(raw)
    result.setdefault("raw_judge_output", raw)
    return result


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Eval Pipeline Functional Test Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Case | Skill | Verdict | Score | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for item in results:
        judge = item["judge"]
        reason = str(judge.get("reason", "")).replace("\n", " ")
        lines.append(
            f"| `{item['case']['id']}` | `{item['case']['skill']}` | "
            f"{judge.get('verdict', 'unknown')} | {float(judge.get('score', 0.0)):.2f} | {reason} |"
        )

    lines.extend(["", "## Details", ""])
    for item in results:
        case = item["case"]
        judge = item["judge"]
        lines.extend(
            [
                f"### {case['id']}",
                "",
                f"- Skill: `{case['skill']}`",
                f"- Verdict: `{judge.get('verdict', 'unknown')}`",
                f"- Score: `{float(judge.get('score', 0.0)):.2f}`",
                f"- Reason: {judge.get('reason', '')}",
                "",
                "Actor output:",
                "",
                "```markdown",
                item["actor_output"].strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def aggregate_runs(runs: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Collapse N repeat runs into one representative (actor_output, judge).

    Verdict = majority across runs (ties broken by the worse verdict, to stay
    conservative). Score = mean. The per-run verdicts/scores are kept in metadata so
    variance is visible in the report.
    """
    if len(runs) == 1:
        return runs[0]["actor_output"], runs[0]["judge"]

    order = {"fail": 0, "partial": 1, "pass": 2}
    verdicts = [str(r["judge"].get("verdict", "unknown")) for r in runs]
    scores = [float(r["judge"].get("score", 0.0)) for r in runs]
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    # Majority; ties → lower (worse) verdict.
    top = max(counts.values())
    majority = sorted([v for v, c in counts.items() if c == top], key=lambda v: order.get(v, -1))[0]
    mean_score = sum(scores) / len(scores)

    # Use the run whose verdict matches the majority as the representative actor output.
    rep = next((r for r in runs if r["judge"].get("verdict") == majority), runs[0])
    judge = dict(rep["judge"])
    judge["verdict"] = majority
    judge["score"] = round(mean_score, 3)
    judge["reason"] = (
        f"[aggregated over {len(runs)} runs: verdicts={counts}, "
        f"scores={[round(s, 2) for s in scores]}, mean={mean_score:.2f}] "
        + str(rep["judge"].get("reason", ""))
    )
    judge["repeat_verdicts"] = verdicts
    judge["repeat_scores"] = scores
    return rep["actor_output"], judge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run functional tests for eval-pipeline skills.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case-id", action="append", help="Run only the given case id. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases after filtering.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3.6-plus"),
                        help="Actor model that follows the skill.")
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL", "qwen3-max"),
                        help="Judge model that grades actor output (default stronger than actor).")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run each case N times and report the majority verdict + mean score. "
                             "Single-run verdicts are noisy (LLM actor + judge); use >=3 for robust claims.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    client = build_client()
    print(f"actor model: {args.model} | judge model: {args.judge_model}")

    cases = load_cases(args.cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No test cases selected.")

    repeat = max(1, args.repeat)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']} ({case['skill']})")
        skill_text = load_skill_text(args.skill_root, case["skill"])
        runs = []
        for rep in range(repeat):
            actor_output = run_actor(client, args.model, skill_text, case)
            judge = judge_actor_output(client, args.judge_model, case, actor_output)
            runs.append({"actor_output": actor_output, "judge": judge})
            if repeat > 1:
                print(f"    run {rep + 1}/{repeat}: {judge.get('verdict')} "
                      f"score={float(judge.get('score', 0.0)):.2f}")
        actor_output, judge = aggregate_runs(runs)
        print(f"  -> {judge.get('verdict')} score={float(judge.get('score', 0.0)):.2f}"
              + (f"  (n={repeat})" if repeat > 1 else ""))
        results.append({"case": case, "actor_output": actor_output, "judge": judge})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "eval_pipeline_functional_results.json"
    md_path = args.out_dir / "eval_pipeline_functional_report.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report(results), encoding="utf-8")

    passed = sum(1 for item in results if item["judge"].get("verdict") == "pass")
    partial = sum(1 for item in results if item["judge"].get("verdict") == "partial")
    failed = sum(1 for item in results if item["judge"].get("verdict") == "fail")
    print()
    print(f"Completed: pass={passed}, partial={partial}, fail={failed}, total={len(results)}")
    print(f"Markdown report: {md_path}")
    print(f"JSON results:    {json_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
