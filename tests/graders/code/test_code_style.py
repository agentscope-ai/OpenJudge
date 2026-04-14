#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CodeStyleGrader.

Tests code style checking including indentation consistency and naming conventions.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_code_style.py -v
    ```
"""

import pytest

from openjudge.graders.code.code_style import CodeStyleGrader


@pytest.mark.unit
class TestCodeStyleGraderUnit:
    """Unit tests for CodeStyleGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        grader = CodeStyleGrader()
        assert grader.name == "code_style"

    @pytest.mark.asyncio
    async def test_no_code_blocks(self):
        """Test response with no code blocks"""
        grader = CodeStyleGrader()
        result = await grader.aevaluate(response="Just plain text, no code here.")

        assert result.score == 0.0
        assert "No code blocks found" in result.reason

    @pytest.mark.asyncio
    async def test_perfect_style(self):
        """Test code with perfect style - consistent indentation and snake_case names"""
        grader = CodeStyleGrader()
        response = """```python
def calculate_sum(a, b):
    return a + b
```"""
        result = await grader.aevaluate(response=response)

        # 0.5 for consistent indentation + 0.5 for naming (2/2 snake_case)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_camel_case_function_name(self):
        """Test code with camelCase function name (non-snake_case)"""
        grader = CodeStyleGrader()
        response = """```python
def CalculateSum(a, b):
    return a + b
```"""
        result = await grader.aevaluate(response=response)

        # 0.5 for indentation + naming_score * 0.5
        # Only CalculateSum matched as function name; no variables matched
        # CalculateSum is not snake_case -> 0/1 naming_score = 0.0
        # total = 0.5 (indent) + 0.0 * 0.5 (naming) = 0.5
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_camel_case_variable_name(self):
        """Test code with camelCase variable name"""
        grader = CodeStyleGrader()
        response = """```python
def process_data():
    myVariable = 42
    return myVariable
```"""
        result = await grader.aevaluate(response=response)

        # process_data is snake_case, myVariable is not
        # naming_score = 1/2 = 0.5
        # total = 0.5 (indent) + 0.5 * 0.5 (naming) = 0.75
        assert result.score == 0.75

    @pytest.mark.asyncio
    async def test_all_camel_case(self):
        """Test code where all names are camelCase"""
        grader = CodeStyleGrader()
        response = """```python
def CalculateTotal():
    TotalAmount = 0
    return TotalAmount
```"""
        result = await grader.aevaluate(response=response)

        # 0 names follow snake_case out of 2
        # naming_score = 0.0
        # total = 0.5 (indent) + 0.0 * 0.5 (naming) = 0.5
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_no_names_to_check(self):
        """Test code with no function or variable names"""
        grader = CodeStyleGrader()
        response = """```python
print("hello world")
```"""
        result = await grader.aevaluate(response=response)

        # No names found, naming_score = 1.0 ("No names to check")
        # total = 0.5 (indent) + 1.0 * 0.5 (naming) = 1.0
        assert result.score == 1.0


@pytest.mark.unit
class TestCodeStyleIndentation:
    """Tests for indentation checking"""

    @pytest.mark.asyncio
    async def test_consistent_spaces(self):
        """Test code with consistent space indentation"""
        grader = CodeStyleGrader()
        response = """```python
def foo():
    if True:
        print("hello")
```"""
        result = await grader.aevaluate(response=response)

        # Consistent spaces -> indent_ok = True -> 0.5
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_no_indentation(self):
        """Test code with no indentation at all (flat code)"""
        grader = CodeStyleGrader()
        response = """```python
x = 1
y = 2
```"""
        result = await grader.aevaluate(response=response)

        # No indentation to check -> indent_ok = True -> 0.5
        assert result.score >= 0.5


@pytest.mark.unit
class TestCodeStyleNaming:
    """Tests for naming convention checking"""

    def test_snake_case_function_names(self):
        """Test _check_naming with snake_case function names"""
        grader = CodeStyleGrader()
        code = "def calculate_sum():\n    pass\n\ndef process_data():\n    pass"
        score, msg = grader._check_naming(code)
        assert score == 1.0
        assert "2/2" in msg

    def test_camel_case_function_names(self):
        """Test _check_naming with camelCase function names"""
        grader = CodeStyleGrader()
        code = "def CalculateSum():\n    pass"
        score, msg = grader._check_naming(code)
        assert score == 0.0
        assert "0/1" in msg

    def test_mixed_naming(self):
        """Test _check_naming with mixed naming styles"""
        grader = CodeStyleGrader()
        code = "def calculate_sum():\n    pass\n\ndef processData():\n    pass"
        score, msg = grader._check_naming(code)
        assert score == 0.5
        assert "1/2" in msg

    def test_snake_case_variables(self):
        """Test _check_naming with snake_case variable names"""
        grader = CodeStyleGrader()
        code = "total_count = 0\nuser_name = 'test'"
        score, msg = grader._check_naming(code)
        assert score == 1.0
        assert "2/2" in msg

    def test_no_names(self):
        """Test _check_naming with no function or variable names"""
        grader = CodeStyleGrader()
        code = "pass"
        score, msg = grader._check_naming(code)
        assert score == 1.0
        assert "No names to check" in msg


@pytest.mark.unit
class TestCodeStyleMultipleBlocks:
    """Tests for multiple code blocks"""

    @pytest.mark.asyncio
    async def test_two_blocks_averaged(self):
        """Test that score is averaged across multiple code blocks"""
        grader = CodeStyleGrader()
        response = (
            "```python\ndef calculate_sum(a, b):\n    return a + b\n```\n"
            "```python\ndef CalculateTotal():\n    TotalAmount = 0\n    return TotalAmount\n```"
        )
        result = await grader.aevaluate(response=response)

        # Block 1: 0.5 (indent) + 0.5 * 1.0 (naming: 2/2) = 1.0
        # Block 2: 0.5 (indent) + 0.5 * 0.0 (naming: 0/2) = 0.5
        # Average: (1.0 + 0.5) / 2 = 0.75
        assert result.score == 0.75
        assert result.metadata["average_score"] == 0.75

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test that metadata contains expected fields"""
        grader = CodeStyleGrader()
        response = "```python\ndef foo():\n    pass\n```"
        result = await grader.aevaluate(response=response)

        assert "code_blocks" in result.metadata
        assert "average_score" in result.metadata
        assert "details" in result.metadata
