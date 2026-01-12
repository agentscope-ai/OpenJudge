# -*- coding: utf-8 -*-
"""
Tokenization Utilities

Tokenization tools for breaking text into tokens.
"""

import re
from typing import List


def simple_tokenize(text: str, lowercase: bool = False) -> List[str]:
    """
    Simple tokenization based on whitespace.

    Args:
        text: Text to tokenize.
        lowercase: Whether to convert to lowercase.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> simple_tokenize("Hello, world!")
        ['Hello,', 'world!']
    """
    if lowercase:
        text = text.lower()
    return text.split()


_non_word_space_pattern = re.compile(r"[^\w\s]")
_word_punctuation_pattern = re.compile(r"\w+|[^\w\s]")


def word_tokenize(text: str, remove_punctuation: bool = True) -> List[str]:
    """
    Word-level tokenization using regex.

    Args:
        text: Text to tokenize.
        remove_punctuation: Whether to remove punctuation marks.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> word_tokenize("Hello, world!")
        ['Hello', 'world']
        >>> word_tokenize("Hello, world!", remove_punctuation=False)
        ['Hello', ',', 'world', '!']
    """
    if remove_punctuation:
        # Keep only letters, numbers, and spaces
        text = _non_word_space_pattern.sub(" ", text)
        tokens = text.split()
    else:
        # Keep punctuation as separate tokens
        tokens = _word_punctuation_pattern.findall(text)

    return [t for t in tokens if t.strip()]


def character_tokenize(text: str) -> List[str]:
    """
    Character-level tokenization.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of characters.

    Example:
        >>> character_tokenize("hello")
        ['h', 'e', 'l', 'l', 'o']
    """
    return list(text)


def ngram_tokenize(text: str, n: int = 2, char_level: bool = False) -> List[str]:
    """
    N-gram tokenization.

    Args:
        text: Text to tokenize.
        n: Size of the n-gram.
        char_level: Whether to use character-level n-grams (otherwise word-level).

    Returns:
        List[str]: List of n-grams.

    Example:
        >>> ngram_tokenize("hello world", n=2, char_level=True)
        ['he', 'el', 'll', 'lo', 'o ', ' w', 'wo', 'or', 'rl', 'ld']
        >>> ngram_tokenize("the cat sat", n=2, char_level=False)
        ['the cat', 'cat sat']
    """
    if char_level:
        tokens = list(text)
    else:
        tokens = text.split()

    if len(tokens) < n:
        return [" ".join(tokens)] if not char_level else ["".join(tokens)]

    ngrams = []
    for i in range(len(tokens) - n + 1):
        if char_level:
            ngrams.append("".join(tokens[i : i + n]))
        else:
            ngrams.append(" ".join(tokens[i : i + n]))

    return ngrams


_sentence_split_pattern = re.compile(r"(?<=[.!?])\s+")


def sentence_tokenize(text: str) -> List[str]:
    """
    Sentence tokenization based on common sentence terminators.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of sentences.

    Example:
        >>> text = "Hello world. How are you? I'm fine!"
        >>> sentence_tokenize(text)
        ['Hello world.', 'How are you?', "I'm fine!"]
    """
    sentences = _sentence_split_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


_word_pattern = re.compile(r"\b\w+\b")


def tokenize_preserving_case(text: str) -> List[str]:
    """
    Tokenization preserving original case.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> tokenize_preserving_case("Hello World")
        ['Hello', 'World']
    """
    return _word_pattern.findall(text)


def whitespace_tokenize(text: str) -> List[str]:
    """
    Tokenization based on whitespace characters.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> whitespace_tokenize("hello\\tworld\\ntest")
        ['hello', 'world', 'test']
    """
    return text.split()


def get_word_count(text: str) -> int:
    """
    Get word count from text.

    Args:
        text: Input text.

    Returns:
        int: Number of words.

    Example:
        >>> get_word_count("Hello, world! How are you?")
        5
    """
    return len(word_tokenize(text))


def get_character_count(text: str, include_spaces: bool = False) -> int:
    """
    Get character count from text.

    Args:
        text: Input text.
        include_spaces: Whether to include spaces in the count.

    Returns:
        int: Number of characters.

    Example:
        >>> get_character_count("hello world")
        10
        >>> get_character_count("hello world", include_spaces=True)
        11
    """
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))


__all__ = [
    "simple_tokenize",
    "word_tokenize",
    "character_tokenize",
    "ngram_tokenize",
    "sentence_tokenize",
    "tokenize_preserving_case",
    "whitespace_tokenize",
    "get_word_count",
    "get_character_count",
]
