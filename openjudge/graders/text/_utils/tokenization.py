# -*- coding: utf-8 -*-
"""
Tokenization Utilities

Tokenization tools for breaking text into tokens.
Supports both English (space-delimited) and CJK (Chinese/Japanese/Korean) text.

For CJK text, uses **jieba** (a required dependency) with multiple cut modes:
  - ``"accurate"`` (default): precise word segmentation, best for evaluation metrics.
  - ``"search"``: finer-grained segmentation optimized for search-engine indexing;
    long words are further segmented (e.g. "中华人民共和国" → "中华", "人民", "共和国",
    "中华人民共和国").
  - ``"all"``: full mode, outputs every possible word for maximum recall.

Chinese stop-words can optionally be removed to improve metric quality.
"""

import re
import string
from typing import List, Literal, Optional, Set

import jieba

# ---------------------------------------------------------------------------
# CJK detection
# ---------------------------------------------------------------------------
# CJK Unified Ideographs, Extension A, CJK Symbols and Punctuation,
# Fullwidth punctuation, Katakana, Hiragana, Hangul
_CJK_RANGES = (
    r"\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\u3400-\u4dbf"  # CJK Extension A
    r"\uf900-\ufaff"  # CJK Compatibility Ideographs
    r"\u3000-\u303f"  # CJK Symbols and Punctuation
    r"\u3040-\u309f"  # Hiragana
    r"\u30a0-\u30ff"  # Katakana
    r"\uac00-\ud7af"  # Hangul Syllables
)
_CJK_PATTERN = re.compile(f"[{_CJK_RANGES}]")

# CJK punctuation set (Chinese fullwidth punctuation etc.)
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
# Chinese stop-words (high-frequency function words that add little meaning)
# ---------------------------------------------------------------------------
_DEFAULT_CHINESE_STOPWORDS: Set[str] = {
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


def contains_cjk(text: str) -> bool:
    """
    Detect whether text contains CJK (Chinese/Japanese/Korean) characters.

    Args:
        text: Input text.

    Returns:
        bool: True if text contains CJK characters.

    Example:
        >>> contains_cjk("Hello world")
        False
        >>> contains_cjk("你好世界")
        True
        >>> contains_cjk("Hello 你好")
        True
    """
    return bool(_CJK_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Core CJK tokenization (jieba-based)
# ---------------------------------------------------------------------------
CutMode = Literal["accurate", "search", "all"]


def _jieba_cut(text: str, mode: CutMode = "accurate") -> List[str]:
    """
    Tokenize text using jieba with the specified cut mode.

    Args:
        text: Input text.
        mode: Cut mode —
            ``"accurate"`` (default): precise segmentation.
            ``"search"``: search-engine mode (further splits long words).
            ``"all"``: full mode (all possible words, maximum recall).

    Returns:
        List[str]: Raw token list (may contain whitespace-only tokens).
    """
    if mode == "search":
        return list(jieba.cut_for_search(text))
    elif mode == "all":
        return list(jieba.cut(text, cut_all=True))
    else:  # "accurate"
        return list(jieba.cut(text, cut_all=False))


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
        remove_punctuation: Remove punctuation tokens.
        remove_stopwords: Remove Chinese stop-word tokens.
        stopwords: Custom stop-word set (uses built-in set if *None*).

    Returns:
        List[str]: Filtered token list.
    """
    if stopwords is None:
        stopwords = _DEFAULT_CHINESE_STOPWORDS

    result = []
    for t in tokens:
        t_stripped = t.strip()
        if not t_stripped:
            continue
        if remove_punctuation and (t_stripped in _ALL_PUNCTUATION or all(ch in _ALL_PUNCTUATION for ch in t_stripped)):
            continue
        if remove_stopwords and t_stripped in stopwords:
            continue
        result.append(t_stripped)
    return result


def _cjk_tokenize(
    text: str,
    mode: CutMode = "accurate",
    remove_punctuation: bool = False,
    remove_stopwords: bool = False,
) -> List[str]:
    """
    Tokenize CJK text using jieba.

    Args:
        text: Text containing CJK characters.
        mode: jieba cut mode (``"accurate"``, ``"search"``, ``"all"``).
        remove_punctuation: Remove punctuation tokens from the result.
        remove_stopwords: Remove Chinese stop-word tokens from the result.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> _cjk_tokenize("机器学习是人工智能的重要分支。")
        ['机器', '学习', '是', '人工智能', '的', '重要', '分支', '。']
        >>> _cjk_tokenize("机器学习是人工智能的重要分支。", remove_punctuation=True, remove_stopwords=True)
        ['机器', '学习', '人工智能', '重要', '分支']
    """
    tokens = _jieba_cut(text, mode=mode)
    return _filter_tokens(
        tokens,
        remove_punctuation=remove_punctuation,
        remove_stopwords=remove_stopwords,
    )


# ---------------------------------------------------------------------------
# Public API — smart_tokenize / simple_tokenize / word_tokenize
# ---------------------------------------------------------------------------


def smart_tokenize(
    text: str,
    lowercase: bool = False,
    mode: CutMode = "accurate",
    remove_stopwords: bool = False,
) -> List[str]:
    """
    Language-aware tokenization (recommended entry point for multilingual text).

    For CJK text, uses jieba; for others, whitespace split.

    Args:
        text: Text to tokenize.
        lowercase: Whether to convert to lowercase.
        mode: jieba cut mode for CJK text (``"accurate"``, ``"search"``, ``"all"``).
        remove_stopwords: Whether to remove Chinese stop-words (CJK only).

    Returns:
        List[str]: List of tokens.

    Example:
        >>> smart_tokenize("Hello world")
        ['Hello', 'world']
        >>> smart_tokenize("机器学习是人工智能的分支")
        ['机器', '学习', '是', '人工智能', '的', '分支']
        >>> smart_tokenize("机器学习是人工智能的分支", remove_stopwords=True)
        ['机器', '学习', '人工智能', '分支']
        >>> smart_tokenize("中华人民共和国成立了", mode="search")
        ['中华', '华人', '人民', '共和', '共和国', '中华人民共和国', '成立', '了']
    """
    if lowercase:
        text = text.lower()
    if contains_cjk(text):
        return _cjk_tokenize(text, mode=mode, remove_stopwords=remove_stopwords)
    return text.split()


def simple_tokenize(text: str, lowercase: bool = False) -> List[str]:
    """
    Simple tokenization — language-aware.

    Uses jieba accurate mode for CJK, whitespace split for others.

    Args:
        text: Text to tokenize.
        lowercase: Whether to convert to lowercase.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> simple_tokenize("Hello, world!")
        ['Hello,', 'world!']
        >>> simple_tokenize("今天天气很好")
        ['今天天气', '很', '好']
    """
    if lowercase:
        text = text.lower()
    if contains_cjk(text):
        return _cjk_tokenize(text, mode="accurate")
    return text.split()


_non_word_space_pattern = re.compile(r"[^\w\s]", re.UNICODE)
_word_punctuation_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def word_tokenize(
    text: str,
    remove_punctuation: bool = True,
    remove_stopwords: bool = False,
    mode: CutMode = "accurate",
) -> List[str]:
    """
    Word-level tokenization — language-aware.

    For CJK text, uses jieba with punctuation/stop-word filtering.
    For non-CJK text, uses regex-based tokenization.

    Args:
        text: Text to tokenize.
        remove_punctuation: Whether to remove punctuation marks.
        remove_stopwords: Whether to remove Chinese stop-words (CJK only).
        mode: jieba cut mode for CJK text.

    Returns:
        List[str]: List of tokens.

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
    if contains_cjk(text):
        return _cjk_tokenize(
            text,
            mode=mode,
            remove_punctuation=remove_punctuation,
            remove_stopwords=remove_stopwords,
        )

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
    N-gram tokenization — language-aware for word-level n-grams.

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


_word_pattern = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize_preserving_case(text: str) -> List[str]:
    """
    Tokenization preserving original case — language-aware.

    For CJK text, delegates to smart_tokenize.
    For non-CJK text, uses regex word boundary matching.

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
    if contains_cjk(text):
        return smart_tokenize(text)
    return _word_pattern.findall(text)


def whitespace_tokenize(text: str) -> List[str]:
    """
    Tokenization based on whitespace characters — language-aware.

    For CJK text, delegates to smart_tokenize since whitespace splitting is
    not meaningful for languages without spaces between words.

    Args:
        text: Text to tokenize.

    Returns:
        List[str]: List of tokens.

    Example:
        >>> whitespace_tokenize("hello\\tworld\\ntest")
        ['hello', 'world', 'test']
        >>> whitespace_tokenize("自然语言处理")
        ['自然语言', '处理']
    """
    if contains_cjk(text):
        return smart_tokenize(text)
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
    "contains_cjk",
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
