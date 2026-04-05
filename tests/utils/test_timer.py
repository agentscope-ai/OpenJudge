# -*- coding: utf-8 -*-
"""Unit tests for timing utilities."""

import asyncio

import pytest

from openjudge.utils.timer import TimingCollector, timed


@pytest.mark.unit
class TestTimingCollector:
    """Test suite for TimingCollector."""

    def test_measure_records_context_timing(self):
        """Should record a timing entry for measured code blocks."""
        collector = TimingCollector()

        with collector.measure("test.context", metadata={"phase": "setup"}):
            sum(range(10))

        records = collector.get_records("test.context")
        assert len(records) == 1
        assert records[0].name == "test.context"
        assert records[0].duration_ms >= 0
        assert records[0].metadata == {"phase": "setup"}

    def test_summary_aggregates_multiple_records(self):
        """Should summarize repeated measurements by operation name."""
        collector = TimingCollector()
        collector.record("test.summary", 5.0)
        collector.record("test.summary", 15.0)

        summary = collector.get_summary()
        assert "test.summary" in summary
        assert summary["test.summary"]["count"] == 2
        assert summary["test.summary"]["total_ms"] == pytest.approx(20.0)
        assert summary["test.summary"]["avg_ms"] == pytest.approx(10.0)
        assert summary["test.summary"]["min_ms"] == pytest.approx(5.0)
        assert summary["test.summary"]["max_ms"] == pytest.approx(15.0)

    def test_clear_removes_all_records(self):
        """Should clear collected timing records."""
        collector = TimingCollector()
        collector.record("test.clear", 1.0)

        collector.clear()

        assert collector.get_records() == []
        assert collector.get_summary() == {}

    def test_timed_decorator_supports_sync_functions(self):
        """Should measure sync functions decorated with timed."""
        collector = TimingCollector()

        @timed("test.sync", collector, metadata={"kind": "sync"})
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

        records = collector.get_records("test.sync")
        assert len(records) == 1
        assert records[0].metadata == {"kind": "sync"}

    @pytest.mark.asyncio
    async def test_timed_decorator_supports_async_functions(self):
        """Should measure async functions decorated with timed."""
        collector = TimingCollector()

        @timed("test.async", collector, metadata={"kind": "async"})
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        assert await async_add(1, 2) == 3

        records = collector.get_records("test.async")
        assert len(records) == 1
        assert records[0].metadata == {"kind": "async"}
