#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for SyntaxCheckGrader.

Tests syntax checking of Python code blocks using AST parsing.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_syntax_checker.py -v
    ```
"""

import pytest

from openjudge.graders.code.syntax_checker import SyntaxCheckGrader


@pytest.mark.unit
class TestSyntaxCheckGraderUnit:
    """Unit tests for SyntaxCheckGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        grader = SyntaxCheckGrader()
        assert grader.name == "syntax_check"

    @pytest.mark.asyncio
    async def test_valid_single_code_block(self):
        """Test a single valid Python code block"""
        grader = SyntaxCheckGrader()
        response = "Here is some code:\n```python\ndef hello():\n    print('Hello')\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == 1.0
        assert result.metadata["valid_blocks"] == 1
        assert result.metadata["total_blocks"] == 1
        assert len(result.metadata["syntax_errors"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_single_code_block(self):
        """Test a single invalid Python code block with syntax error"""
        grader = SyntaxCheckGrader()
        response = "Here is bad code:\n```python\ndef hello(\n    print('Hello')\n```"
        result = await grader.aevaluate(response=response)

        # 0/1 valid + 0.5 penalty = -0.5
        assert result.score == -0.5
        assert result.metadata["valid_blocks"] == 0
        assert result.metadata["total_blocks"] == 1
        assert len(result.metadata["syntax_errors"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_code_blocks_all_valid(self):
        """Test multiple code blocks all with valid syntax"""
        grader = SyntaxCheckGrader()
        response = "```python\ndef foo():\n    pass\n```\n" "Some text\n" "```python\ndef bar():\n    return 42\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == 1.0
        assert result.metadata["valid_blocks"] == 2
        assert result.metadata["total_blocks"] == 2

    @pytest.mark.asyncio
    async def test_multiple_code_blocks_mixed(self):
        """Test multiple code blocks with some valid and some invalid"""
        grader = SyntaxCheckGrader()
        response = "```python\ndef foo():\n    pass\n```\n" "Some text\n" "```python\ndef bar(\n    return 42\n```"
        result = await grader.aevaluate(response=response)

        # 1/2 valid = 0.5, minus 0.5 penalty for syntax errors = 0.0
        assert result.score == 0.0
        assert result.metadata["valid_blocks"] == 1
        assert result.metadata["total_blocks"] == 2
        assert len(result.metadata["syntax_errors"]) == 1

    @pytest.mark.asyncio
    async def test_no_code_blocks(self):
        """Test response with no code blocks"""
        grader = SyntaxCheckGrader()
        response = "This is just plain text without any code."
        result = await grader.aevaluate(response=response)

        assert result.score == 0.0
        assert "No code blocks found" in result.reason
        assert result.metadata["code_blocks"] == []

    @pytest.mark.asyncio
    async def test_code_block_without_python_tag(self):
        """Test code block with generic (non-python) fence"""
        grader = SyntaxCheckGrader()
        response = "```\ndef hello():\n    print('Hello')\n```"
        result = await grader.aevaluate(response=response)

        # The regex matches ```python or ``` followed by code
        # Generic code block should also be parsed
        assert result.metadata["total_blocks"] >= 1

    @pytest.mark.asyncio
    async def test_syntax_error_metadata(self):
        """Test that syntax error metadata includes line and offset info"""
        grader = SyntaxCheckGrader()
        response = "```python\nif True\n    print('missing colon')\n```"
        result = await grader.aevaluate(response=response)

        assert len(result.metadata["syntax_errors"]) == 1
        error = result.metadata["syntax_errors"][0]
        assert "block" in error
        assert "error" in error
        assert "line" in error
        assert "offset" in error
        assert error["block"] == 0

    @pytest.mark.asyncio
    async def test_complex_valid_code(self):
        """Test complex but valid Python code"""
        grader = SyntaxCheckGrader()
        response = """```python
import os
from typing import List, Optional

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config

    def process(self, items: List[str]) -> Optional[str]:
        if not items:
            return None
        return items[0]
```"""
        result = await grader.aevaluate(response=response)

        assert result.score == 1.0
        assert result.metadata["valid_blocks"] == 1

    @pytest.mark.asyncio
    async def test_python_tag_case_insensitive(self):
        """A ```Python (capitalized) fence must be checked, not skipped"""
        grader = SyntaxCheckGrader()
        response = "```Python\ndef broken(:\n    pass\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == -0.5
        assert result.metadata["total_blocks"] == 1
        assert len(result.metadata["syntax_errors"]) == 1

    @pytest.mark.asyncio
    async def test_py_alias_tag_checked(self):
        """A ```py fence must be checked"""
        grader = SyntaxCheckGrader()
        response = "```py\nprint(1)\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == 1.0
        assert result.metadata["valid_blocks"] == 1

    @pytest.mark.asyncio
    async def test_non_python_fence_skipped_not_paired(self):
        """Regression: a non-Python fence must not let its closing marker
        pair with the next opening fence, capturing the prose in between as
        a code block while the real Python block goes unchecked."""
        grader = SyntaxCheckGrader()
        response = "```js\nconsole.log(1)\n```\njust prose\n```python\ndef ok():\n    pass\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == 1.0
        assert result.metadata["code_blocks"] == ["def ok():\n    pass"]
        assert result.metadata["skipped_non_python_blocks"] == 1

    @pytest.mark.asyncio
    async def test_prose_not_captured_as_code_block(self):
        """Regression: prose between an unrecognized fence and the next
        recognized fence must not be returned as a code block."""
        grader = SyntaxCheckGrader()
        response = "```Python\ndef broken(:\n    pass\n```\n" "Some prose here\n" "```python\ndef ok():\n    pass\n```"
        result = await grader.aevaluate(response=response)

        assert result.metadata["code_blocks"] == ["def broken(:\n    pass", "def ok():\n    pass"]
        # 1 of 2 Python blocks valid, with the -0.5 penalty for the syntax error
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_only_non_python_fences_is_neutral(self):
        """Only non-Python fences: neutral score with the skip count reported"""
        grader = SyntaxCheckGrader()
        response = "```js\nconsole.log(1)\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == 0.0
        assert result.metadata["code_blocks"] == []
        assert result.metadata["skipped_non_python_blocks"] == 1

    @pytest.mark.asyncio
    async def test_empty_js_block_does_not_hide_following_invalid_python(self):
        """Regression (review follow-up on #209): an empty fenced block
        (```js with no content) must be consumed as a unit. Before the fix
        its opening fence was not consumed, so the invalid Python block
        after it escaped checking and the response scored 1.0."""
        grader = SyntaxCheckGrader()
        response = (
            "```python\ndef ok():\n    return 1\n```\n"
            "```js\n```\n"
            "some prose\n"
            "```python\ndef broken(:\n    pass\n```"
        )
        result = await grader.aevaluate(response=response)

        # 1 of 2 Python blocks valid, plus the -0.5 penalty
        assert result.score == 0.0
        assert result.metadata["valid_blocks"] == 1
        assert result.metadata["total_blocks"] == 2
        assert result.metadata["skipped_non_python_blocks"] == 1
        assert result.metadata["code_blocks"] == [
            "def ok():\n    return 1",
            "def broken(:\n    pass",
        ]

    @pytest.mark.asyncio
    async def test_empty_python_block_counts_as_valid_module(self):
        """An empty Python fence parses as an empty module and stays valid"""
        grader = SyntaxCheckGrader()
        response = "```python\n```\n```python\ndef ok():\n    pass\n```"
        result = await grader.aevaluate(response=response)

        assert result.score == 1.0
        assert result.metadata["code_blocks"] == ["", "def ok():\n    pass"]
        assert result.metadata["skipped_non_python_blocks"] == 0
