# -*- coding: utf-8 -*-
"""Canonical provenance and fail-closed group/sample overlap checks."""

from itertools import combinations

from pydantic import field_validator, model_validator

from openjudge.calibration.contracts import (
    FrozenModel,
    ObservationBatch,
    PartitionRole,
    canonical_digest,
)


class PartitionProvenance(FrozenModel):
    """Content fingerprint and complete identity sets, including excluded rows."""

    role: PartitionRole
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    used_sample_ids: tuple[str, ...]
    sample_count: int
    valid_count: int
    sample_fingerprint: str
    group_fingerprint: str
    fingerprint: str

    @model_validator(mode="after")
    def validate_sets(self) -> "PartitionProvenance":
        """Reject inconsistent serialized provenance before overlap auditing."""
        for values in (self.sample_ids, self.group_ids, self.used_sample_ids):
            if tuple(sorted(set(values))) != values or any(not item.strip() for item in values):
                raise ValueError("provenance identity sets must be unique, nonempty and sorted")
        if self.sample_count != len(self.sample_ids) or self.valid_count != len(self.used_sample_ids):
            raise ValueError("provenance counts do not match identities")
        if not set(self.used_sample_ids).issubset(self.sample_ids):
            raise ValueError("used identities must belong to the source partition")
        if self.sample_ids and not self.group_ids:
            raise ValueError("nonempty source requires group identities")
        if self.sample_fingerprint != canonical_digest(self.sample_ids):
            raise ValueError("sample fingerprint mismatch")
        if self.group_fingerprint != canonical_digest(self.group_ids):
            raise ValueError("group fingerprint mismatch")
        return self

    @field_validator("fingerprint", "sample_fingerprint", "group_fingerprint")
    @classmethod
    def digest(cls, value: str) -> str:
        """Require a canonical SHA256 hex digest."""
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid SHA256 fingerprint")
        return value


class SplitManifest(FrozenModel):
    """A jointly audited set of disjoint partitions."""

    partitions: tuple[PartitionProvenance, ...]
    fingerprint: str

    @model_validator(mode="after")
    def validate_manifest(self) -> "SplitManifest":
        """Recheck overlap and content identity when loading saved manifests."""
        assert_disjoint(*self.partitions)
        expected = canonical_digest([part.model_dump(mode="json") for part in self.partitions])
        if self.fingerprint != expected:
            raise ValueError("manifest fingerprint mismatch")
        return self


def partition_provenance(batch: ObservationBatch) -> PartitionProvenance:
    """Fingerprint content independently of row order, without serializing labels."""
    samples = tuple(sorted(row.sample_id for row in batch.observations))
    groups = tuple(sorted({row.group_id for row in batch.observations}))
    used = tuple(sorted(row.sample_id for row in batch.valid))
    rows = sorted((row.model_dump(mode="json") for row in batch.observations), key=lambda row: row["sample_id"])
    return PartitionProvenance(
        role=batch.role,
        sample_ids=samples,
        group_ids=groups,
        used_sample_ids=used,
        sample_count=len(samples),
        valid_count=len(used),
        sample_fingerprint=canonical_digest(samples),
        group_fingerprint=canonical_digest(groups),
        fingerprint=canonical_digest(
            {
                "rows": rows,
                "contract": batch.contract.fingerprint,
                "scope": batch.scope.model_dump(mode="json"),
                "role": batch.role,
                "fit_artifact_id": batch.fit_artifact_id,
            }
        ),
    )


def assert_disjoint(*partitions: PartitionProvenance) -> None:
    """Fail before fitting/selection if any sample or complete task group overlaps."""
    for left, right in combinations(partitions, 2):
        if set(left.sample_ids).intersection(right.sample_ids):
            raise ValueError(f"sample overlap between {left.role} and {right.role}")
        if set(left.group_ids).intersection(right.group_ids):
            raise ValueError(f"group overlap between {left.role} and {right.role}")


def audit_partitions(*batches: ObservationBatch) -> SplitManifest:
    """Build a validated manifest over all rows, including errors and exclusions.

    Args:
        *batches: Explicit partitions. No random row splitting occurs here.

    Returns:
        A reproducible identity manifest; any cross-partition overlap raises.
    """
    partitions = tuple(partition_provenance(batch) for batch in batches)
    assert_disjoint(*partitions)
    return SplitManifest(
        partitions=partitions,
        fingerprint=canonical_digest([part.model_dump(mode="json") for part in partitions]),
    )
