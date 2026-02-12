# -*- coding: utf-8 -*-
"""
Tokenization Utilities

Tokenization tools for breaking text into tokens.
All tokenization is powered by **jieba**, which handles both CJK and Latin text
natively — English words, numbers, and punctuation are preserved as whole tokens
while Chinese/Japanese/Korean text is properly segmented.

jieba cut modes:
  - ``"accurate"`` (default): precise word segmentation, best for evaluation.
  - ``"search"``: finer-grained segmentation that further splits long words
    (e.g. "中华人民共和国" → "中华", "人民", "共和国", "中华人民共和国").
  - ``"all"``: full mode — outputs every possible word for maximum recall.

Stop-words can optionally be removed to improve metric quality.
"""

import re
import string
from typing import List, Literal, Optional, Set

import jieba

# ---------------------------------------------------------------------------
# Punctuation sets
# ---------------------------------------------------------------------------
# CJK fullwidth and common punctuation (Unicode escapes for black-safety)
CJK_PUNCTUATION = set(
    "\u3001\u3002"  # 、。
    "\uff01\uff1f"  # ！？
    "\uff0c\uff1b\uff1a"  # ，；：
    "\u201c\u201d"  # ""
    "\u2018\u2019"  # ''
    "\u3010\u3011"  # 【】
    "\uff08\uff09"  # （）
    "\u300a\u300b"  # 《》
    "\u2026\u2014\uff5e\u00b7"  # …—～·
    "\u300c\u300d"  # 「」
    "\u300e\u300f"  # 『』
    "\u3008\u3009"  # 〈〉
    "\u3014\u3015"  # 〔〕
)

# Combined punctuation (ASCII + CJK)
_ALL_PUNCTUATION = set(string.punctuation) | CJK_PUNCTUATION

# ---------------------------------------------------------------------------
# Default stop-words (Chinese high-frequency function words)
# ---------------------------------------------------------------------------
_DEFAULT_STOPWORDS: Set[str] = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "他",
    "她",
    "么",
    "那",
    "被",
    "从",
    "把",
    "它",
    "与",
    "及",
    "其",
    "或",
    "之",
    "而",
    "但",
    "对",
    "等",
    "能",
    "将",
    "可以",
    "已",
    "所",
    "为",
    "以",
    "这个",
    "那个",
    "什么",
    "怎么",
    "如何",
    "因为",
    "所以",
    "虽然",
    "但是",
    "如果",
    "应该",
    "可能",
    "没",
    "吗",
    "呢",
    "吧",
    "啊",
    "呀",
    "哦",
    "嗯",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
CutMode = Literal["accurate", "search", "all"]


def _jieba_cut(text: str, mode: CutMode = "accurate") -> List[str]:
    """
    Tokenize *text* with jieba in the given *mode*.

    Works for any language — English words/numbers are kept intact.

    Args:
        text: Input text.
        mode: ``"accurate"`` | ``"search"`` | ``"all"``.

    Returns:
        List[str]: Raw token list (may contain whitespace-only tokens).
    """
    if mode == "search":
        return list(jieba.cut_for_search(text))
    if mode == "all":
        return list(jieba.cut(text, cut_all=True))
    return list(jieba.cut(text, cut_all=False))  # accurate


def _filter_tokens(
    tokens: List[str],
    remove_punctuation: bool = False,
    remove_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Post-process a token list: strip whitespace, optionally remove punctuation
    and stop-words.

    Args:
        tokens: Raw token list.
        remove_punctuation: Drop punctuation tokens (ASCII + CJK).
        remove_stopwords: Drop stop-word tokens.
        stopwords: Custom stop-word set (falls back to built-in set).

    Returns:
        List[str]: Filtered token list.
    """
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    result: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if remove_punctuation and (tok in _ALL_PUNCTUATION or all(ch in _ALL_PUNCTUATION for ch in tok)):
            continue
        if remove_stopwords and tok in stopwords:
            continue
        result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smart_tokenize(
    text: str,
    lowercase: bool = False,
    mode: CutMode = "accurate",
    remove_stopwords: bool = False,
) -> List[str]:
    """
    Unified tokenization via jieba (recommended entry-point).

    Handles English, Chinese, mixed, and any other text that jieba supports.

    Args:
        text: Text to tokenize.
        lowercase: Convert to lowercase first.
        mode: jieba cut mode (``"accurate"`` | ``"search"`` | ``"all"``).
        remove_stopwords: Drop Chinese stop-words.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> smart_tokenize("Hello world")
        ['Hello', 'world']
        >>> smart_tokenize("机器学习是人工智能的分支")
        ['机器', '学习', '是', '人工智能', '的', '分支']
        >>> smart_tokenize("机器学习是人工智能的分支", remove_stopwords=True)
        ['机器', '学习', '人工智能', '分支']
    """
    if lowercase:
        text = text.lower()
    tokens = _jieba_cut(text, mode=mode)
    return _filter_tokens(tokens, remove_stopwords=remove_stopwords)


def simple_tokenize(text: str, lowercase: bool = False) -> List[str]:
    """
    Simple tokenization via jieba (accurate mode, no filtering).

    Args:
        text: Text to tokenize.
        lowercase: Convert to lowercase first.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> simple_tokenize("Hello, world!")
        ['Hello', ',', 'world', '!']
        >>> simple_tokenize("今天天气很好")
        ['今天天气', '很', '好']
    """
    if lowercase:
        text = text.lower()
    tokens = _jieba_cut(text, mode="accurate")
    return _filter_tokens(tokens)


def word_tokenize(
    text: str,
    remove_punctuation: bool = True,
    remove_stopwords: bool = False,
    mode: CutMode = "accurate",
) -> List[str]:
    """
    Word-level tokenization via jieba with optional punctuation/stop-word removal.

    Args:
        text: Text to tokenize.
        remove_punctuation: Drop punctuation tokens (ASCII + CJK).
        remove_stopwords: Drop stop-word tokens.
        mode: jieba cut mode.

    Returns:
        List[str]: List of word tokens.

    Example:
        >>> word_tokenize("Hello, world!")
        ['Hello', 'world']
        >>> word_tokenize("Hello, world!", remove_punctuation=False)
        ['Hello', ',', 'world', '!']
        >>> word_tokenize("机器学习是人工智能的重要分支。")
        ['机器', '学习', '是', '人工智能', '的', '重要', '分支']
        >>> word_tokenize("机器学习是人工智能的重要分支。", remove_stopwords=True)
        ['机器', '学习', '人工智能', '重要', '分支']
    """
    if not text:
        return []
    tokens = _jieba_cut(text, mode=mode)
    return _filter_tokens(
        tokens,
        remove_punctuation=remove_punctuation,
        remove_stopwords=remove_stopwords,
    )


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
    N-gram tokenization (word-level via jieba, or character-level).

    Args:
        text: Text to tokenize.
        n: Size of the n-gram.
        char_level: Use character-level n-grams (otherwise word-level).

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
        tokens = smart_tokenize(text)

    if len(tokens) < n:
        return [" ".join(tokens)] if not char_level else ["".join(tokens)]

    ngrams = []
    for i in range(len(tokens) - n + 1):
        if char_level:
            ngrams.append("".join(tokens[i : i + n]))
        else:
            ngrams.append(" ".join(tokens[i : i + n]))

    return ngrams


# Sentence split: supports English (.!?) and CJK (。！？) terminators
_sentence_split_pattern = re.compile(r"(?<=[.!?。！？])\s*")


def sentence_tokenize(text: str) -> List[str]:
    """
    Sentence tokenization based on common sentence terminators.

    Supports both English (.!?) and CJK (。！？) sentence endings.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of sentences.

    Example:
        >>> text = "Hello world. How are you? I'm fine!"
        >>> sentence_tokenize(text)
        ['Hello world.', 'How are you?', "I'm fine!"]
        >>> sentence_tokenize("今天天气很好。明天会下雨！")
        ['今天天气很好。', '明天会下雨！']
    """
    sentences = _sentence_split_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_preserving_case(text: str) -> List[str]:
    """
    Tokenization preserving original case.

    Uses jieba accurate mode without any filtering.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> tokenize_preserving_case("Hello World")
        ['Hello', 'World']
        >>> tokenize_preserving_case("机器学习很重要")
        ['机器', '学习', '很', '重要']
    """
    return smart_tokenize(text)


def whitespace_tokenize(text: str) -> List[str]:
    """
    Tokenization via jieba (replaces naive whitespace splitting).

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> whitespace_tokenize("hello world")
        ['hello', 'world']
        >>> whitespace_tokenize("自然语言处理")
        ['自然语言', '处理']
    """
    return smart_tokenize(text)


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
    return len(text.replace(" ", ""))


__all__ = [
    "smart_tokenize",
    "simple_tokenize",
    "word_tokenize",
    "character_tokenize",
    "ngram_tokenize",
    "sentence_tokenize",
    "tokenize_preserving_case",
    "whitespace_tokenize",
    "get_word_count",
    "get_character_count",
    "CJK_PUNCTUATION",
]
