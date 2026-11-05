#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Python fence extraction helper.
"""

import pytest

from openjudge.utils.markdown import extract_python_code_blocks


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
