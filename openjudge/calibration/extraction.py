# -*- coding: utf-8 -*-
"""Explicit extraction from aligned native runner results; never infer confidence."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from openjudge.calibration.contracts import (
    CalibrationContract, CalibrationScope, Observation, ObservationBatch, PartitionRole,
    is_decision_label, probability, valid_identifier,
)
from openjudge.graders.schema import GraderError, GraderResult, GraderScore

ConfidenceExtractor = Callable[[dict[str, Any], GraderResult], Any]


def read_path(value: Any, path: str) -> Any:
    """Read a dotted mapping/model path; absent components have value None."""
    current = value
    for component in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(component)
        elif isinstance(current, GraderResult):
            current = getattr(current, component, None)
        else:
            return None
    return current


def extract_confidence(
    sample: dict[str, Any], result: GraderResult, contract: CalibrationContract,
    confidence_extractor: ConfidenceExtractor | None = None,
) -> tuple[float | None, str | None]:
    """Read the declared probability and return a stable omission status.

    Callback exceptions propagate: a broken extractor is a programming failure,
    not an ordinary missing provider confidence.
    """
    if contract.extractor_id is not None:
        if confidence_extractor is None:
            raise ValueError(f"confidence extractor {contract.extractor_id!r} must be supplied explicitly")
        value = confidence_extractor(sample, result)
    else:
        if confidence_extractor is not None:
            raise ValueError("callback requires a contract extractor_id, not a confidence_path")
        source = result if contract.confidence_source == "result" else sample
        value = read_path(source, contract.confidence_path)
    if value is None:
        return None, "missing_confidence"
    try:
        return probability(value), None
    except ValueError:
        return None, "invalid_confidence"


def _row_values(
    sample: dict[str, Any], result: GraderResult, contract: CalibrationContract,
    confidence_extractor: ConfidenceExtractor | None,
) -> tuple[str, float | None, int | None]:
    """Assign exactly one status in documented failure/exclusion precedence."""
    if isinstance(result, GraderError):
        return "provider_error", None, None
    if contract.exclude_path is not None:
        excluded = read_path(sample, contract.exclude_path)
        if excluded is not None and not isinstance(excluded, bool):
            raise ValueError("exclude_path must reference a boolean")
        if excluded:
            return "user_excluded", None, None
    if not isinstance(result, GraderScore):
        return "unsupported_result", None, None
    label = read_path(sample, contract.label_path)
    if label is None:
        return "missing_label", None, None
    if not is_decision_label(label) or label not in contract.decision_labels:
        return "invalid_label", None, None
    prediction = read_path(result, contract.prediction_path)
    if prediction is None:
        return "missing_prediction", None, None
    if not is_decision_label(prediction) or prediction not in contract.decision_labels:
        return "invalid_prediction", None, None
    confidence, status = extract_confidence(sample, result, contract, confidence_extractor)
    if status is not None:
        return status, None, None
    return "valid", confidence, int(prediction == label)


def extract_observations(
    dataset: Sequence[dict[str, Any]], grader_results: Sequence[GraderResult],
    contract: CalibrationContract, scope: CalibrationScope, *, role: PartitionRole = "analysis",
    invalid_policy: Literal["raise", "exclude"] = "raise",
    confidence_extractor: ConfidenceExtractor | None = None,
) -> ObservationBatch:
    """Detach native results into ordered correctness-confidence observations.

    Args:
        dataset: Independently labeled samples with stable task/group identities.
        grader_results: Native results in exactly the same order.
        contract: Explicit target, decision and confidence paths.
        scope: Versioned judge/model/application identity.
        role: Data's permitted experimental role.
        invalid_policy: Fail on malformed rows, or report every omission.
        confidence_extractor: Callback matching the contract's stable extractor ID.

    Returns:
        An immutable batch, retaining every row in the original denominator.
    """
    if len(dataset) != len(grader_results):
        raise ValueError("dataset and grader_results must have the same length")
    if invalid_policy not in ("raise", "exclude"):
        raise ValueError("invalid_policy must be raise or exclude")
    if (contract.extractor_id is not None) != (confidence_extractor is not None):
        raise ValueError("declared confidence extractor and supplied callback must match")
    observations = []
    for index, (sample, result) in enumerate(zip(dataset, grader_results)):
        try:
            sample_id = valid_identifier(read_path(sample, contract.sample_id_path))
            group_id = valid_identifier(read_path(sample, contract.group_id_path))
        except ValueError as exc:
            raise ValueError(f"row {index} requires nonempty sample and group IDs") from exc
        domain = read_path(sample, contract.domain_path) if contract.domain_path is not None else None
        if domain is not None:
            domain = valid_identifier(domain)
        status, confidence, correctness = _row_values(sample, result, contract, confidence_extractor)
        if invalid_policy == "raise" and status not in ("valid", "provider_error", "user_excluded"):
            raise ValueError(f"sample {sample_id!r}: {status}")
        observations.append(Observation(
            sample_id=sample_id, group_id=group_id, domain=domain, status=status,
            probability=confidence, correctness=correctness,
        ))
    return ObservationBatch(observations=tuple(observations), contract=contract, scope=scope, role=role)


def vote_fraction(_sample: dict[str, Any], result: GraderResult) -> float | None:
    """Explicit raw vote agreement, not an assertion of calibrated correctness.

    Args:
        _sample: Unused sample, matching the extraction callback protocol.
        result: VotingEvaluationStrategy result with native original_results.

    Returns:
        Fraction of successful votes agreeing with the selected decision, or None.
        Failed votes do not enter this raw feature's denominator; the caller must
        record this interpretation in CalibrationContract.
    """
    if not isinstance(result, GraderScore):
        return None
    votes = result.metadata.get("original_results")
    if not isinstance(votes, (list, tuple)):
        return None
    scores = []
    for vote in votes:
        if isinstance(vote, GraderScore):
            scores.append(vote.score)
        elif isinstance(vote, dict) and "score" in vote and "error" not in vote:
            scores.append(vote["score"])
    if not scores:
        return None
    return sum(value == result.score for value in scores) / len(scores)
