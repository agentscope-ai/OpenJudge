# -*- coding: utf-8 -*-
"""Extraction of Python code blocks from markdown-style fenced blocks."""

import re
from typing import List, Tuple

# Info strings (fence tags) that mark a fenced block as Python. An empty tag
# (a bare ``` fence) is treated as Python to match the previous behavior of
# the code graders.
PYTHON_FENCE_TAGS = frozenset({"", "python", "py", "python3"})

# Matches any fenced block and captures its info string (tag) so that every
# fence is consumed in document order. The previous pattern used by the code
# graders, r"```(?:python)?\s*\n(.*?)\n\s*```", only matched bare or
# lowercase-```python fences; any other fence (```Python, ```py, ```js, ...)
# was skipped by the matcher, so its closing fence could pair with the next
# recognized opening fence and the prose in between was captured as a "code
# block".
# The closing fence of an empty block follows the opening fence's newline
# directly, so the newline between content and the closing fence must be
# optional; otherwise an empty fence (```js\n```) is not consumed at all and
# its opening marker can pair with a later fence, leaving the block after it
# unchecked (review follow-up on #209).
_FENCED_BLOCK_PATTERN = re.compile(r"```([^\n`]*)\s*\n(.*?)(?:\n\s*)?```", re.DOTALL)


def extract_python_code_blocks(text: str) -> Tuple[List[str], int]:
    """Extract the contents of Python fenced code blocks from ``text``.

    Args:
        text: Response text that may contain markdown-style fenced blocks.

    Returns:
        Tuple of (python_blocks, skipped_block_count). ``python_blocks`` holds
        the content of every fence tagged as Python (case-insensitive
        ``python``/``py``/``python3``, or untagged), in document order.
        ``skipped_block_count`` is the number of fenced blocks carrying other
        tags (e.g. ```js), which are ignored.

    Example:
        >>> blocks, skipped = extract_python_code_blocks(
        ...     "```js\\nx\\n```\\nsome prose\\n```python\\nprint(1)\\n```"
        ... )
        >>> blocks, skipped
        (['print(1)'], 1)
    """
    fenced = _FENCED_BLOCK_PATTERN.findall(text)
    python_blocks = [content for tag, content in fenced if tag.strip().lower() in PYTHON_FENCE_TAGS]
    return python_blocks, len(fenced) - len(python_blocks)
