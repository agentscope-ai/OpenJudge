"""Native, offline fixtures for correctness-confidence experiments."""

import pytest

from openjudge.calibration.contracts import CalibrationContract, CalibrationScope
from openjudge.calibration.extraction import extract_observations
from openjudge.graders.schema import GraderScore


@pytest.fixture
def contract():
    return CalibrationContract(target_id="judge_matches_reference", confidence_path="metadata.confidence")


@pytest.fixture
def scope():
    return CalibrationScope(judge_id="judge-v1", model_id="model-v1", scope_id="arithmetic")


@pytest.fixture
def make_batch(contract, scope):
    def build(probabilities, correctness, role="calibration", prefix="fit", groups=None, domains=None):
        data = [
            {
                "id": f"{prefix}-{i}",
                "group": groups[i] if groups is not None else f"{prefix}-task-{i}",
                "label": int(y),
                **({"domain": domains[i]} if domains is not None else {}),
            }
            for i, y in enumerate(correctness)
        ]
        results = [GraderScore(name="judge", score=1, reason="", metadata={"confidence": p}) for p in probabilities]
        selected_contract = contract.model_copy(update={"domain_path": "domain"}) if domains is not None else contract
        return extract_observations(data, results, selected_contract, scope, role=role)

    return build
