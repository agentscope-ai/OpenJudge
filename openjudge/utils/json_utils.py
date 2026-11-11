# -*- coding: utf-8 -*-
"""Shared JSON parsing helpers that enforce RFC 8259 semantics."""

from typing import NoReturn


def reject_non_finite(token: str) -> NoReturn:
    """Reject the non-JSON number literals Python's json module accepts.

    Python's ``json.loads`` parses the ``NaN``, ``Infinity`` and
    ``-Infinity`` literals that RFC 8259 does not define. Passing this
    function as ``parse_constant`` makes ``json.loads`` raise ``ValueError``
    on those literals, so graders treat such responses as invalid JSON
    instead of scoring them.

    Args:
        token (str): The constant literal found by the parser (``NaN``,
            ``Infinity`` or ``-Infinity``).

    Raises:
        ValueError: Always, so the surrounding ``json.loads`` fails.

    Example:
        >>> import json
        >>> json.loads('{"a": NaN}', parse_constant=reject_non_finite)
        Traceback (most recent call last):
            ...
        ValueError: NaN is not a valid JSON number
    """
    raise ValueError(f"{token} is not a valid JSON number")
