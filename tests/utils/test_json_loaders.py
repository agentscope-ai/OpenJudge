#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the LLM-response JSON loaders (RFC 8259 enforcement)."""

import pytest

from openjudge.utils.utils import repair_and_load_json, trim_and_load_json


@pytest.mark.unit
class TestTrimAndLoadJson:
    """Unit tests for trim_and_load_json"""

    def test_valid_json_parsed(self):
        """Valid JSON passes through unchanged"""
        assert trim_and_load_json('{"name": "Alice", "age": 30}') == {"name": "Alice", "age": 30}

    def test_code_fence_stripped(self):
        """Markdown code fences around the JSON are removed before parsing"""
        response = '```json\n{"score": 8, "reasoning": "Good"}\n```'
        assert trim_and_load_json(response) == {"score": 8, "reasoning": "Good"}

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_literal_rejected(self, literal: str):
        """NaN/Infinity literals are invalid JSON and must fail parsing, not
        feed float('nan')/float('inf') into downstream scores and statistics."""
        with pytest.raises(ValueError, match="not a valid JSON number"):
            trim_and_load_json(f'{{"score": {literal}}}')

    def test_non_finite_literal_in_fenced_block_rejected(self):
        """The rejection also applies to fenced LLM responses"""
        with pytest.raises(ValueError, match="not a valid JSON number"):
            trim_and_load_json('```json\n{"score": NaN}\n```')


@pytest.mark.unit
class TestRepairAndLoadJson:
    """Unit tests for repair_and_load_json"""

    def test_valid_json_parsed(self):
        """Valid JSON passes through unchanged"""
        assert repair_and_load_json('{"key": "value"}') == {"key": "value"}

    def test_incomplete_json_repaired(self):
        """Incomplete JSON is repaired (existing behavior, locked)"""
        assert repair_and_load_json('{"a": "b"') == {"a": "b"}

    def test_complete_non_finite_literal_rejected(self):
        """A complete NaN literal must fail parsing instead of loading as nan"""
        with pytest.raises(ValueError, match="not a valid JSON number"):
            repair_and_load_json('{"score": NaN}')

    def test_complete_infinity_literal_rejected(self):
        """A complete Infinity literal must fail parsing instead of loading as inf"""
        with pytest.raises(ValueError, match="not a valid JSON number"):
            repair_and_load_json('{"score": Infinity}')

    def test_incomplete_non_finite_literal_repaired_to_string(self):
        """json_repair stringifies a bare NaN while repairing incomplete input
        (library behavior, locked so the grader never receives a float nan)"""
        assert repair_and_load_json('{"score": NaN') == {"score": "NaN"}
