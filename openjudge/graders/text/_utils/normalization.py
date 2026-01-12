# -*- coding: utf-8 -*-
"""
Text Normalization Utilities

Text normalization tools for standardizing text to improve metric evaluation accuracy.
"""

import re
import string
import unicodedata
from typing import Optional


# pylint: disable=redefined-outer-name
def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_articles: bool = True,
    remove_extra_whitespace: bool = True,
    case_sensitive: bool = False,
) -> str:
    """
    Basic text normalization

    Based on OpenAI Evals framework normalization implementation.

    Args:
        text: Text to be normalized
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        remove_articles: Whether to remove English articles (a, an, the)
        remove_extra_whitespace: Whether to remove extra whitespace
        case_sensitive: Whether to preserve case sensitivity (overrides lowercase parameter)

    Returns:
        str: Normalized text

    Example:
        >>> text = "  The quick brown fox!  "
        >>> normalize_text(text)
        'quick brown fox'
    """
    if not text:
        return ""

    # Don't convert to lowercase if case sensitivity is needed
    if not case_sensitive and lowercase:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        exclude = set(string.punctuation)
        text = "".join(char for char in text if char not in exclude)

    # Remove English articles
    if remove_articles:
        # Use word boundaries to ensure only complete words are matched
        text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)

    # Remove extra whitespace
    if remove_extra_whitespace:
        text = " ".join(text.split())

    return text


def normalize_text_advanced(
    text: str,
    lowercase: bool = True,
    remove_accents: bool = True,
    remove_numbers: bool = False,
    remove_special_chars: bool = True,
    normalize_unicode: bool = True,
    strip: bool = True,
) -> str:
    """
    Advanced text normalization

    Provides more normalization options, suitable for multilingual text.

    Args:
        text: Text to be normalized
        lowercase: Whether to convert to lowercase
        remove_accents: Whether to remove accent marks
        remove_numbers: Whether to remove numbers
        remove_special_chars: Whether to remove special characters
        normalize_unicode: Whether to perform Unicode normalization
        strip: Whether to strip leading/trailing whitespace

    Returns:
        str: Normalized text

    Example:
        >>> text = "Café résumé 123"
        >>> normalize_text_advanced(text, remove_accents=True)
        'cafe resume 123'
    """
    if not text:
        return ""

    # Unicode normalization
    if normalize_unicode:
        text = unicodedata.normalize("NFKD", text)

    # Remove accent marks
    if remove_accents:
        text = "".join(char for char in text if not unicodedata.combining(char))

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    # Remove special characters (preserve letters, numbers, spaces)
    if remove_special_chars:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Strip leading/trailing whitespace
    if strip:
        text = text.strip()

    # Normalize spaces
    text = " ".join(text.split())

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace characters

    Unifies all whitespace characters (spaces, tabs, newlines, etc.) into single spaces.

    Args:
        text: Text to be processed

    Returns:
        str: Normalized text

    Example:
        >>> text = "hello\\n\\tworld  "
        >>> normalize_whitespace(text)
        'hello world'
    """
    return " ".join(text.split())


def remove_punctuation(text: str, keep_chars: Optional[str] = None) -> str:
    """
    Remove punctuation

    Args:
        text: Text to be processed
        keep_chars: Characters to preserve (optional)

    Returns:
        str: Text with punctuation removed

    Example:
        >>> remove_punctuation("Hello, world!")
        'Hello world'
        >>> remove_punctuation("Hello, world!", keep_chars=",")
        'Hello, world'
    """
    if keep_chars:
        exclude = set(string.punctuation) - set(keep_chars)
    else:
        exclude = set(string.punctuation)

    return "".join(char for char in text if char not in exclude)


def normalize_for_comparison(text: str, method: str = "standard") -> str:
    """
    Normalize text for comparison using the specified method.

    Args:
        text: Text to normalize.
        method: Normalization method.
            - "standard": Standard normalization (lowercase + remove punctuation + remove articles).
            - "minimal": Minimal normalization (only remove extra whitespace).
            - "aggressive": Aggressive normalization (all options enabled).
            - "case_only": Case normalization only.

    Returns:
        str: Normalized text.

    Example:
        >>> normalize_for_comparison("The Cat!", "standard")
        'cat'
        >>> normalize_for_comparison("The Cat!", "minimal")
        'The Cat!'
    """
    if method == "standard":
        return normalize_text(text)
    elif method == "minimal":
        return normalize_whitespace(text.strip())
    elif method == "aggressive":
        return normalize_text_advanced(
            text,
            remove_accents=True,
            remove_numbers=False,
            remove_special_chars=True,
        )
    elif method == "case_only":
        return text.lower()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_numbers(text: str, replace_with: str = " NUMBER ") -> str:
    """
    Replace numbers with a placeholder.

    Args:
        text: Text to process.
        replace_with: Placeholder string to replace numbers with.

    Returns:
        str: Text with numbers replaced.

    Example:
        >>> normalize_numbers("I have 3 apples and 5 oranges")
        'I have  NUMBER  apples and  NUMBER  oranges'
    """
    return re.sub(r"\d+\.?\d*", replace_with, text)


def normalize_urls(text: str, replace_with: str = " URL ") -> str:
    """
    Replace URLs with a placeholder.

    Args:
        text: Text to process.
        replace_with: Placeholder string to replace URLs with.

    Returns:
        str: Text with URLs replaced.

    Example:
        >>> text = "Visit https://example.com for more info"
        >>> normalize_urls(text)
        'Visit  URL  for more info'
    """
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, replace_with, text)


def normalize_emails(text: str, replace_with: str = " EMAIL ") -> str:
    """
    Replace email addresses with a placeholder.

    Args:
        text: Text to process.
        replace_with: Placeholder string to replace emails with.

    Returns:
        str: Text with email addresses replaced.

    Example:
        >>> text = "Contact me at user@example.com"
        >>> normalize_emails(text)
        'Contact me at  EMAIL '
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.sub(email_pattern, replace_with, text)


__all__ = [
    "normalize_text",
    "normalize_text_advanced",
    "normalize_whitespace",
    "remove_punctuation",
    "normalize_for_comparison",
    "normalize_numbers",
    "normalize_urls",
    "normalize_emails",
]
