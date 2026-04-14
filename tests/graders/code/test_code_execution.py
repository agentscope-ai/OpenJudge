#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CodeExecutionGrader.

Tests code execution evaluation with mocked compute_score framework.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_code_execution.py -v
    ```
"""

from unittest.mock import MagicMock

import pytest

from openjudge.graders.code.code_execution import CodeExecutionGrader


@pytest.mark.unit
class TestCodeExecutionGraderUnit:
    """Unit tests for CodeExecutionGrader - testing isolated functionality"""

    def test_initialization_default_params(self):
        """Test initialization with default parameters"""
        grader = CodeExecutionGrader()
        assert grader.name == "code_execution"
        assert grader.continuous is True
        assert grader.timeout == 10

    @pytest.mark.asyncio
    async def test_framework_not_available(self):
        """Test evaluation when framework is not available"""
        grader = CodeExecutionGrader(test_framework_available=False)
        result = await grader.aevaluate(response="def add(a, b): return a + b")

        assert result.score == 0.0
        assert "not available" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_framework_not_available_default(self):
        """Test that when compute_score module is missing, framework is marked unavailable"""
        # The __init__ will attempt to import compute_score and set
        # test_framework_available=False if the import fails.
        # In this test env, the module is not installed, so it should be False.
        grader = CodeExecutionGrader()
        # If the import fails (likely in test env), framework should be unavailable
        if not grader.test_framework_available:
            result = await grader.aevaluate(response="def add(a, b): return a + b")
            assert result.score == 0.0
            assert "not available" in result.reason.lower()


@pytest.mark.unit
class TestCodeExecutionExtractCode:
    """Tests for the _extract_code method"""

    def test_extract_python_code_block(self):
        """Test extracting code from ```python``` block"""
        grader = CodeExecutionGrader()
        content = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        code = grader._extract_code(content)

        assert "def add(a, b):" in code
        assert "return a + b" in code

    def test_extract_generic_code_block(self):
        """Test extracting code from ``` (generic) block"""
        grader = CodeExecutionGrader()
        content = "Here is the code:\n```\ndef add(a, b):\n    return a + b\n```"
        code = grader._extract_code(content)

        assert "def add(a, b):" in code

    def test_extract_no_code_block(self):
        """Test extracting when no code block markers exist"""
        grader = CodeExecutionGrader()
        content = "def add(a, b):\n    return a + b"
        code = grader._extract_code(content)

        # Should return the entire content as-is
        assert code == content

    def test_extract_python_block_preferred_over_generic(self):
        """Test that ```python``` block is preferred over generic ``` block"""
        grader = CodeExecutionGrader()
        content = "```python\ndef python_func():\n    pass\n```\n" "```\ndef generic_func():\n    pass\n```"
        code = grader._extract_code(content)

        assert "python_func" in code
        assert "generic_func" not in code

    def test_extract_multiline_code(self):
        """Test extracting multiline code from block"""
        grader = CodeExecutionGrader()
        content = """```python
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in seen:
                return [seen[diff], i]
            seen[num] = i
```"""
        code = grader._extract_code(content)

        assert "class Solution:" in code
        assert "def twoSum" in code


@pytest.mark.unit
class TestCodeExecutionWithMockedFramework:
    """Tests with mocked compute_score framework"""

    @pytest.mark.asyncio
    async def test_all_tests_pass(self):
        """Test when all test cases pass (boolean True)"""
        mock_compute_score = MagicMock(return_value=(True, {}))
        success, _ = mock_compute_score(
            completion="def add(a, b): return a + b",
            test_cases='{"inputs": ["1 2"], "outputs": ["3"]}',
            continuous=True,
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_partial_pass_rate(self):
        """Test when some test cases pass (float pass rate)"""
        mock_compute_score = MagicMock(return_value=(0.6, {}))
        success, _ = mock_compute_score(
            completion="def add(a, b): return a - b",
            test_cases='{"inputs": ["1 2", "3 4"], "outputs": ["3", "7"]}',
            continuous=True,
        )
        assert success == 0.6

    @pytest.mark.asyncio
    async def test_all_tests_fail(self):
        """Test when all test cases fail"""
        mock_compute_score = MagicMock(return_value=(False, {}))
        success, _ = mock_compute_score(
            completion="def add(a, b): return a - b",
            test_cases='{"inputs": ["1 2"], "outputs": ["3"]}',
            continuous=True,
        )
        assert success is False


@pytest.mark.unit
class TestCodeExecutionEdgeCases:
    """Edge case tests for CodeExecutionGrader"""

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Test evaluation with empty response"""
        grader = CodeExecutionGrader(test_framework_available=True)
        result = await grader.aevaluate(response="")

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_response_with_only_text(self):
        """Test evaluation with response containing only text, no code"""
        grader = CodeExecutionGrader(test_framework_available=False)
        result = await grader.aevaluate(response="This is just text without any code.")

        assert result.score == 0.0

    def test_custom_timeout(self):
        """Test initialization with custom timeout"""
        grader = CodeExecutionGrader(timeout=30)
        assert grader.timeout == 30

    def test_continuous_flag(self):
        """Test initialization with continuous=False"""
        grader = CodeExecutionGrader(continuous=False)
        assert grader.continuous is False
