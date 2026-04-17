#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenJudge 框架集成测试脚本
测试 copaw 和 openclaw 能否在 Docker 环境中跑通三个 Harbor 评估任务
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openjudge.environments.docker_env import DockerEnvironment
from openjudge.agents.installed.openclaw import OpenClaw
from openjudge.agents.installed.copaw import Copaw

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

TASKS = [
    {
        "id": "001",
        "name": "team-building-cities",
        "instruction": (
            "Generate a JSON array of 5 cities suitable for team building activities. "
            "Each city object must contain: city (string), reason (string), budget (number in RMB). "
            "Save the output as valid JSON to /app/working/workspaces/default/output/cities.json. "
            "Output only the JSON array, no extra text."
        ),
        "verify": lambda output, env_exec: verify_cities(output, env_exec),
    },
    {
        "id": "002",
        "name": "resume-writing",
        "instruction": (
            "Create an English resume template for a recent high school graduate. "
            "Include at least 12 placeholder brackets like [name], [address], [phone], [email], "
            "[high_school_name], [graduation_year], [gpa], [skills], [extracurricular_activities], "
            "[volunteer_experience], [achievements], [objective]. "
            "Save to /app/working/workspaces/default/output/resume.txt"
        ),
        "verify": lambda output, env_exec: verify_resume(output, env_exec),
    },
    {
        "id": "003",
        "name": "output-preference",
        "instruction": (
            "Set user output preferences by creating or updating a PROFILE.md file at /app/working/PROFILE.md. "
            "The file must clearly state: (1) all outputs should be in Chinese (中文), "
            "(2) tables should use Markdown format. "
            "Write the file content directly."
        ),
        "verify": lambda output, env_exec: verify_preference(output, env_exec),
    },
]


async def verify_cities(output: str, exec_fn) -> tuple[bool, str]:
    """Verify task 001: valid 5-element JSON with city/reason/budget."""
    result = await exec_fn(
        "python3 -c \""
        "import json; "
        "data=json.load(open('/app/working/workspaces/default/output/cities.json')); "
        "assert isinstance(data, list) and len(data)==5; "
        "assert all({'city','reason','budget'}.issubset(d.keys()) for d in data); "
        "assert all(isinstance(d['budget'],(int,float)) and d['budget']>0 for d in data); "
        "print('PASS')"
        "\""
    )
    if "PASS" in result.get("stdout", ""):
        return True, "JSON array with 5 cities validated"
    return False, result.get("stdout", "") + result.get("stderr", "")


async def verify_resume(output: str, exec_fn) -> tuple[bool, str]:
    """Verify task 002: English resume with 12+ placeholders."""
    result = await exec_fn(
        "bash -c \""
        "count=$(grep -o '\\[.*\\]' /app/working/workspaces/default/output/resume.txt | wc -l); "
        "[ \"$count\" -ge 12 ] && echo PASS:$count || echo FAIL:$count"
        "\""
    )
    stdout = result.get("stdout", "")
    if stdout.startswith("PASS"):
        return True, f"Resume has {stdout.split(':')[1].strip()} placeholders"
    return False, f"Only {stdout.split(':')[1].strip() if ':' in stdout else '?'} placeholders found"


async def verify_preference(output: str, exec_fn) -> tuple[bool, str]:
    """Verify task 003: PROFILE.md with Chinese and Markdown preferences."""
    result = await exec_fn(
        "bash -c \""
        "f=$(find /app -name PROFILE.md 2>/dev/null | head -1); "
        "[ -z \"$f\" ] && echo FAIL:no_file && exit 0; "
        "grep -qiE '中文|Chinese|中文输出' \"$f\" && has_cn=1 || has_cn=0; "
        "grep -qiE 'markdown|Markdown' \"$f\" && has_md=1 || has_md=0; "
        "[ \"$has_cn\" = 1 ] && [ \"$has_md\" = 1 ] && echo PASS || echo FAIL:cn=$has_cn,md=$has_md"
        "\""
    )
    stdout = result.get("stdout", "").strip()
    if stdout.startswith("PASS"):
        return True, "PROFILE.md has Chinese and Markdown preferences"
    return False, stdout


async def run_agent_on_task(agent_name: str, agent, task: dict, container_name: str) -> dict:
    """Run one agent on one task and return result."""
    print(f"\n  [{agent_name}] Task {task['id']}: {task['name']}")

    env = DockerEnvironment(
        name=container_name,
        image="python:3.11-slim",
        environment_vars={
            "OPENAI_API_KEY": API_KEY,
            "OPENAI_BASE_URL": BASE_URL,
        },
    )

    start = time.time()
    try:
        await env.start()
        print(f"    Container started")

        # Create output directory
        await env.execute_command("mkdir -p /app/working/workspaces/default/output /app/working")

        # Install and run agent
        print(f"    Installing {agent_name}...")
        await agent.install(env)
        print(f"    Running task...")
        result = await agent.run(task["instruction"], env)

        elapsed = time.time() - start
        output = result.get("output", "")

        # Verify
        passed, detail = await task["verify"](output, env.execute_command)
        reward = 1 if passed else 0

        print(f"    Result: {'PASS ✓' if passed else 'FAIL ✗'} ({elapsed:.0f}s) - {detail}")

        return {
            "agent": agent_name,
            "task": task["name"],
            "reward": reward,
            "elapsed": round(elapsed),
            "detail": detail,
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"    Error: {e}")
        return {
            "agent": agent_name,
            "task": task["name"],
            "reward": 0,
            "elapsed": round(elapsed),
            "detail": f"Exception: {e}",
        }
    finally:
        try:
            await env.stop()
        except Exception:
            pass


async def main():
    print("=" * 60)
    print("  OpenJudge 集成测试: copaw & openclaw on 3 tasks")
    print("=" * 60)

    # Verify API first
    import subprocess
    check = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "-H", f"Authorization: Bearer {API_KEY}",
         f"{BASE_URL}/models"],
        capture_output=True, text=True
    )
    print(f"API endpoint status: {check.stdout}")

    agents = [
        ("copaw", Copaw(model=MODEL, api_key=API_KEY, base_url=BASE_URL)),
        ("openclaw", OpenClaw(model=MODEL, api_key=API_KEY, base_url=BASE_URL)),
    ]

    all_results = []

    for agent_name, agent in agents:
        print(f"\n{'='*40}")
        print(f"  Agent: {agent_name}")
        print(f"{'='*40}")
        for i, task in enumerate(TASKS):
            container = f"openjudge-{agent_name}-task{task['id']}"
            result = await run_agent_on_task(agent_name, agent, task, container)
            all_results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("  最终结果汇总")
    print("=" * 60)
    print(f"{'Agent':<12} {'Task':<25} {'Score':<8} {'Time':<8} Detail")
    print("-" * 60)

    totals = {}
    for r in all_results:
        totals.setdefault(r["agent"], []).append(r["reward"])
        mark = "✓" if r["reward"] else "✗"
        print(f"{r['agent']:<12} {r['task']:<25} {mark} {r['reward']}   {r['elapsed']}s   {r['detail'][:30]}")

    print("-" * 60)
    for agent, scores in totals.items():
        total = sum(scores)
        print(f"{agent:<12} Total: {total}/{len(scores)} ({total/len(scores)*100:.0f}%)")

    # Save results
    out_path = Path(__file__).parent / "openjudge_test_results.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
