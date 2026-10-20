# -*- coding: utf-8 -*-
"""
Tests for string match compute functions with empty grading targets
"""

import pytest

from openjudge.graders.text._utils.string_match_compute import (
    compute_contains_all,
    compute_contains_any,
    compute_prefix_match,
    compute_regex_match,
    compute_substring_match,
    compute_suffix_match,
)

WRONG_RESPONSE = "completely wrong answer"


class TestEmptyTargetDoesNotMatch:
    """An empty reference must not act as a universal match"""

    @pytest.mark.parametrize(
        "compute_fn",
        [compute_prefix_match, compute_suffix_match, compute_substring_match],
    )
    def test_empty_reference(self, compute_fn):
        """Empty reference is not a prefix/suffix/substring match"""
        score, details = compute_fn("", WRONG_RESPONSE)
        assert score == 0.0
        assert details["matched"] is False

    def test_regex_empty_reference_without_pattern(self):
        """Empty reference with no explicit pattern does not match"""
        score, details = compute_regex_match("", WRONG_RESPONSE)
        assert score == 0.0
        assert details["matched"] is False

    @pytest.mark.parametrize("compute_fn", [compute_contains_all, compute_contains_any])
    def test_contains_empty_reference_without_substrings(self, compute_fn):
        """Empty reference with no explicit substrings does not match"""
        score, details = compute_fn("", WRONG_RESPONSE)
        assert score == 0.0
        assert details["matched"] is False

    def test_substring_bidirectional_empty_response(self):
        """Empty response is not treated as contained in the reference"""
        score, details = compute_substring_match("cat", "", bidirectional=True)
        assert score == 0.0
        assert details["matched"] is False


class TestNormalMatchesPreserved:
    """Non-empty targets keep their normal matching behavior"""

    def test_prefix_match(self):
        """Valid prefix still matches"""
        assert compute_prefix_match("Hello", "Hello World")[0] == 1.0

    def test_suffix_match(self):
        """Valid suffix still matches"""
        assert compute_suffix_match("World", "Hello World")[0] == 1.0

    def test_substring_match(self):
        """Valid substring still matches, including bidirectionally"""
        assert compute_substring_match("cat", "The cat sat")[0] == 1.0
        assert compute_substring_match("The cat sat", "cat", bidirectional=True)[0] == 1.0

    def test_regex_match(self):
        """Reference used as a regex still matches"""
        assert compute_regex_match(r"\d{3}-\d{4}", "My phone is 123-4567")[0] == 1.0

    def test_contains_all(self):
        """Non-empty reference is used as the target substring"""
        assert compute_contains_all("cat", "The cat sat")[0] == 1.0

    def test_contains_any(self):
        """Non-empty reference is used as the target substring"""
        assert compute_contains_any("cat", "The cat sat")[0] == 1.0


class TestExplicitTargetsWithEmptyReference:
    """Explicit pattern/substrings are evaluated normally even if the reference is empty"""

    def test_regex_explicit_pattern(self):
        """Explicit pattern is used"""
        assert compute_regex_match("", "id: 42", pattern=r"\d+")[0] == 1.0
        assert compute_regex_match("", WRONG_RESPONSE, pattern=r"\d+")[0] == 0.0

    def test_contains_all_explicit_substrings(self):
        """Explicit substrings are used"""
        assert compute_contains_all("", "The cat sat", substrings=["cat", "sat"])[0] == 1.0
        assert compute_contains_all("", "The cat sat", substrings=["cat", "dog"])[0] == 0.5

    def test_contains_any_explicit_substrings(self):
        """Explicit substrings are used"""
        assert compute_contains_any("", "The cat sat", substrings=["dog", "cat"])[0] == 1.0
        assert compute_contains_any("", "The cat sat", substrings=["dog", "bird"])[0] == 0.0
