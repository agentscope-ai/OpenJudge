# -*- coding: utf-8 -*-
"""Offline native-runner calibration example on explicitly constructed data.

Run from the repository root:
    python cookbooks/grader_validation/calibrated_decisions.py --output-dir /tmp/calibration-example

This verifies engineering behavior. The constructed shift is not a benchmark,
licensed human-label corpus, or evidence of improved real-agent performance.
"""

import argparse
import asyncio
from pathlib import Path

import numpy as np

from openjudge.calibration import (
    BootstrapSettings,
    CachedPartition,
    CalibrationContract,
    CalibrationScope,
    DecisionAdapter,
    FitSettings,
    SelectionSettings,
    WorkflowSettings,
    run_calibration_workflow,
)
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore
from openjudge.runner.grading_runner import GradingRunner


def scripted_judge(prediction: int, confidence: float) -> GraderScore:
    """Replay a supplied decision and confidence without access to target labels."""
    return GraderScore(
        name="scripted_judge", score=prediction, reason="constructed", metadata={"confidence": confidence}
    )


def constructed_partition(role: str, rng: np.random.Generator) -> list[dict]:
    """Keep both observations from each task in the same explicit partition."""
    rows = []
    for task in range(64):
        confidence = float(rng.uniform(0.1, 0.99))
        # Deliberately overconfident raw reports; test additionally changes the
        # correctness relationship. No method is claimed to solve this shift.
        exponent = 3 if role == "test" else 2
        for repeat in range(2):
            correct = int(rng.random() < confidence**exponent)
            rows.append(
                {
                    "id": f"{role}-{task}-{repeat}",
                    "group": f"{role}-task-{task}",
                    "domain": "constructed_shift" if role == "test" else "constructed_development",
                    "prediction": 1,
                    "label": correct,
                    "confidence": confidence,
                }
            )
    return rows


async def run(output_dir: Path) -> None:
    """Run a real native grader/runner, save caches and replay one separated workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    grader = FunctionGrader(func=scripted_judge, name="scripted_judge")
    runner = GradingRunner(
        grader_configs={
            "judge": (grader, {"prediction": "prediction", "confidence": "confidence"}),
        }
    )
    partitions = []
    for role in ("calibration", "validation", "test"):
        dataset = constructed_partition(role, rng)
        results = await runner.arun(dataset)
        cached = CachedPartition.from_runner(dataset, results, "judge", role=role)
        cached.save(output_dir / f"{role}.json")
        partitions.append(CachedPartition.load(output_dir / f"{role}.json"))
    contract = CalibrationContract(
        target_id="scripted_decision_matches_independent_constructed_label_v1",
        confidence_path="metadata.confidence",
        domain_path="domain",
    )
    scope = CalibrationScope(judge_id="scripted_v1", model_id="no_model", scope_id="constructed_arithmetic")
    settings = WorkflowSettings(
        fit=FitSettings(method="isotonic"),
        selection=SelectionSettings(risk_target=0.2, min_accepted=10),
        bootstrap=BootstrapSettings(resamples=100, seed=42),
    )
    report = run_calibration_workflow(*partitions, contract, scope, settings)
    report.save(output_dir / "report.json")
    report.fit_artifact.save(output_dir / "calibrator.json")
    report.policy.save(output_dir / "policy.json")
    adapter = DecisionAdapter(report.fit_artifact, report.policy, contract, scope)
    envelopes = adapter.decide_many(partitions[2].grader_results, partitions[2].dataset)
    print(f"Report: {output_dir / 'report.json'}")
    print(f"Test accepted: {sum(item.status == 'accepted' for item in envelopes)}/{len(envelopes)}")
    print(f"Test accepted risk: {report.test.selected.risk}")
    print("Constructed data only; empirical selection and bootstrap carry no population-risk guarantee.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output_dir))
