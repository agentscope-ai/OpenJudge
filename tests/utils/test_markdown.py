#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Python fence extraction helper.
"""

import pytest

from openjudge.utils.markdown import (
    extract_first_python_code,
    extract_python_code_blocks,
)


@pytest.mark.unit
class TestExtractPythonCodeBlocks:
    """Unit tests for extract_python_code_blocks"""

    def test_untagged_fence_treated_as_python(self):
        """A bare ``` fence is checked, matching the previous behavior"""
        blocks, skipped = extract_python_code_blocks("```\nprint(1)\n```")
        assert blocks == ["print(1)"]
        assert skipped == 0

    def test_python_tags_case_insensitive(self):
        """```Python / ```PYTHON fences are recognized as Python"""
        blocks, skipped = extract_python_code_blocks("```Python\na = 1\n```\n```PYTHON\nb = 2\n```")
        assert blocks == ["a = 1", "b = 2"]
        assert skipped == 0

    def test_py_and_python3_aliases(self):
        """```py and ```python3 fences are recognized as Python"""
        blocks, skipped = extract_python_code_blocks("```py\na = 1\n```\n```python3\nb = 2\n```")
        assert blocks == ["a = 1", "b = 2"]
        assert skipped == 0

    def test_non_python_fences_skipped_and_counted(self):
        """Non-Python fences are skipped, not paired with the next fence"""
        text = "```js\nconsole.log(1)\n```\njust prose\n```python\ndef ok():\n    pass\n```"
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ["def ok():\n    pass"]
        assert skipped == 1

    def test_prose_between_fences_never_captured(self):
        """Regression: an unrecognized fence's closing marker must not pair
        with the next recognized opening fence, capturing the prose in
        between as a code block."""
        text = "```Python\ndef broken(:\n    pass\n```\n" "Some prose here\n" "```python\ndef ok():\n    pass\n```"
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ["def broken(:\n    pass", "def ok():\n    pass"]
        assert skipped == 0

    def test_document_order_preserved(self):
        """Blocks are returned in document order"""
        text = "```python\nfirst\n```\nprose\n```python\nsecond\n```"
        blocks, _ = extract_python_code_blocks(text)
        assert blocks == ["first", "second"]

    def test_only_non_python_fences(self):
        """All non-Python fences yields no blocks and the full skip count"""
        blocks, skipped = extract_python_code_blocks("```js\na\n```\n```bash\nb\n```")
        assert blocks == []
        assert skipped == 2

    def test_plain_text_without_fences(self):
        """Plain text without fences yields nothing"""
        blocks, skipped = extract_python_code_blocks("just some text, no fences")
        assert blocks == []
        assert skipped == 0

    def test_empty_non_python_block_is_consumed(self):
        """An empty fenced block (closing fence right after the opening one)
        must be consumed as a unit, not left to pair with a later fence."""
        text = "```js\n```\nprose\n```python\nx = 1\n```"
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ["x = 1"]
        assert skipped == 1

    def test_empty_python_block_yields_empty_content(self):
        """An empty Python fence extracts as an empty block, not as a pairing error"""
        text = "```python\n```\n```python\ny = 2\n```"
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ["", "y = 2"]
        assert skipped == 0

    def test_valid_then_empty_js_then_invalid_python(self):
        """Reviewer case (#209 follow-up): a valid Python block, an empty JS
        block, prose, then an invalid Python block — the invalid block must
        still be extracted so graders can check it."""
        text = (
            "```python\ndef ok():\n    return 1\n```\n"
            "```js\n```\n"
            "some prose\n"
            "```python\ndef broken(:\n    pass\n```"
        )
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ["def ok():\n    return 1", "def broken(:\n    pass"]
        assert skipped == 1

    def test_inline_fence_in_string_does_not_close_block(self):
        """A triple-backtick marker inside a Python string literal must not
        close the fenced block; the block runs to the real closing fence
        (review follow-up #209, inline-backtick regression)."""
        text = 'Here is the fix:\n```python\ns = "```"\nx = 1\n```\nDone.'
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ['s = "```"\nx = 1']
        assert skipped == 0

    def test_inline_fence_in_comment_does_not_close_block(self):
        """Triple-backtick markers inside comments and string literals are
        content, not closing fences (review follow-up #209)."""
        text = (
            "```python\n"
            "# docs at ```usage``` section\n"
            'def f():\n    return "```nested```"\n'
            "```\n"
            "trailing text"
        )
        blocks, skipped = extract_python_code_blocks(text)
        assert blocks == ['# docs at ```usage``` section\ndef f():\n    return "```nested```"']
        assert skipped == 0


@pytest.mark.unit
class TestExtractFirstPythonCode:
    """Unit tests for extract_first_python_code"""

    def test_tagged_block_preferred_over_earlier_untagged(self):
        """An explicitly tagged python fence wins over an earlier plain fence"""
        text = "```\nplain text block\n```\n```python\nprint(1)\n```"
        assert extract_first_python_code(text) == "print(1)"

    def test_case_insensitive_and_alias_tags(self):
        """```Python and ```py fences are recognized"""
        assert extract_first_python_code("```Python\na = 1\n```") == "a = 1"
        assert extract_first_python_code("```py\nb = 2\n```") == "b = 2"

    def test_crlf_fences_extracted(self):
        """CRLF fences still extract; the trailing \r stays in the content"""
        assert extract_first_python_code("```python\r\nprint(1)\r\n```") == "print(1)\r"

    def test_untagged_fallback(self):
        """A plain fence is used when no tagged python fence exists"""
        assert extract_first_python_code("prose\n```\ncode()\n```") == "code()"

    def test_non_python_only_returns_none(self):
        """Only non-Python fences yields None so callers can fall back"""
        assert extract_first_python_code("```js\nconsole.log(1)\n```") is None

    def test_plain_text_returns_none(self):
        """Text without fences yields None"""
        assert extract_first_python_code("just prose, no fences") is None

    def test_first_tagged_block_wins(self):
        """Document order decides among tagged blocks"""
        text = "```python\nfirst()\n```\n```python\nsecond()\n```"
        assert extract_first_python_code(text) == "first()"
