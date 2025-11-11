"""
Tokenization Utilities

分词工具，用于将文本分解为词元（tokens）。
"""

import re
from typing import List


def simple_tokenize(text: str, lowercase: bool = False) -> List[str]:
    """
    简单分词（基于空格）

    Args:
        text: 待分词的文本
        lowercase: 是否转换为小写

    Returns:
        List[str]: 词元列表

    Example:
        >>> simple_tokenize("Hello, world!")
        ['Hello,', 'world!']
    """
    if lowercase:
        text = text.lower()
    return text.split()


def word_tokenize(text: str, remove_punctuation: bool = True) -> List[str]:
    """
    单词级分词

    使用正则表达式分词，可选择是否移除标点。

    Args:
        text: 待分词的文本
        remove_punctuation: 是否移除标点符号

    Returns:
        List[str]: 词元列表

    Example:
        >>> word_tokenize("Hello, world!")
        ['Hello', 'world']
        >>> word_tokenize("Hello, world!", remove_punctuation=False)
        ['Hello', ',', 'world', '!']
    """
    if remove_punctuation:
        # 只保留字母、数字和空格
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
    else:
        # 保留标点，但将其作为独立的token
        tokens = re.findall(r"\w+|[^\w\s]", text)

    return [t for t in tokens if t.strip()]


def character_tokenize(text: str) -> List[str]:
    """
    字符级分词

    Args:
        text: 待分词的文本

    Returns:
        List[str]: 字符列表

    Example:
        >>> character_tokenize("hello")
        ['h', 'e', 'l', 'l', 'o']
    """
    return list(text)


def ngram_tokenize(text: str, n: int = 2, char_level: bool = False) -> List[str]:
    """
    N-gram 分词

    Args:
        text: 待分词的文本
        n: N-gram 的大小
        char_level: 是否为字符级 n-gram（否则为词级）

    Returns:
        List[str]: N-gram 列表

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


def sentence_tokenize(text: str) -> List[str]:
    """
    句子分词

    简单的句子分割，基于常见的句子结束符。

    Args:
        text: 待分词的文本

    Returns:
        List[str]: 句子列表

    Example:
        >>> text = "Hello world. How are you? I'm fine!"
        >>> sentence_tokenize(text)
        ['Hello world.', 'How are you?', "I'm fine!"]
    """
    # 简单的句子分割规则
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_preserving_case(text: str) -> List[str]:
    """
    保持大小写的分词

    Args:
        text: 待分词的文本

    Returns:
        List[str]: 词元列表

    Example:
        >>> tokenize_preserving_case("Hello World")
        ['Hello', 'World']
    """
    return re.findall(r"\b\w+\b", text)


def whitespace_tokenize(text: str) -> List[str]:
    """
    基于空白字符的分词

    Args:
        text: 待分词的文本

    Returns:
        List[str]: 词元列表

    Example:
        >>> whitespace_tokenize("hello\\tworld\\ntest")
        ['hello', 'world', 'test']
    """
    return text.split()


def get_word_count(text: str) -> int:
    """
    获取单词数量

    Args:
        text: 文本

    Returns:
        int: 单词数量

    Example:
        >>> get_word_count("Hello, world! How are you?")
        5
    """
    return len(word_tokenize(text))


def get_character_count(text: str, include_spaces: bool = False) -> int:
    """
    获取字符数量

    Args:
        text: 文本
        include_spaces: 是否包含空格

    Returns:
        int: 字符数量

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
