# -*- coding: utf-8 -*-
"""Lightweight timing utilities for performance instrumentation.

This module provides a small reusable timing collector that can be used as a
context manager or decorator. Timing records are stored in memory, summarized
by operation name, and logged at DEBUG level by default to avoid cluttering
normal output.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Iterator

from loguru import logger


@dataclass(frozen=True)
class TimingRecord:
    """A single timing measurement."""

    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TimingCollector:
    """Collect timing records and expose aggregated summaries."""

    def __init__(self, log_level: str = "DEBUG") -> None:
        self.log_level = log_level.upper()
        self._records: list[TimingRecord] = []

    @contextmanager
    def measure(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        """Measure execution time for a code block."""
        start = perf_counter()
        try:
            yield
        finally:
            duration_ms = (perf_counter() - start) * 1000
            self.record(name=name, duration_ms=duration_ms, metadata=metadata)

    def record(
        self,
        name: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> TimingRecord:
        """Add a timing record and emit a debug log entry."""
        record = TimingRecord(
            name=name,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        self._records.append(record)
        logger.log(
            self.log_level,
            "Timing | {name} took {duration_ms:.3f} ms | metadata={metadata}",
            name=record.name,
            duration_ms=record.duration_ms,
            metadata=record.metadata,
        )
        return record

    def get_records(self, name: str | None = None) -> list[TimingRecord]:
        """Return collected records, optionally filtered by operation name."""
        if name is None:
            return list(self._records)
        return [record for record in self._records if record.name == name]

    def get_summary(self) -> dict[str, dict[str, float | int]]:
        """Return aggregate timing statistics grouped by operation name."""
        grouped_records: dict[str, list[float]] = defaultdict(list)
        for record in self._records:
            grouped_records[record.name].append(record.duration_ms)

        summary: dict[str, dict[str, float | int]] = {}
        for name, durations in grouped_records.items():
            summary[name] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
            }
        return summary

    def clear(self) -> None:
        """Clear all collected timing records."""
        self._records.clear()


def timed(
    name: str,
    collector: TimingCollector,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Decorator for timing sync or async functions with a collector."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with collector.measure(name, metadata=metadata):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with collector.measure(name, metadata=metadata):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


__all__ = ["TimingCollector", "TimingRecord", "timed"]
