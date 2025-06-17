import re
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field


class BaseTokenizer(BaseModel, ABC):
    """
    Base tokenizer class providing unified tokenization interface.

    This abstract base class defines the interface for different tokenization
    strategies including tiktoken and jieba tokenizers.
    """

    name: str = Field(..., description="Name of the tokenizer")

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into a list of tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List[str]: List of token strings
        """
        pass

    def preprocess_text(self, text: str, to_lower: bool = False) -> str:
        """
        Preprocess text before tokenization.

        Args:
            text: Input text
            to_lower: Whether to convert to lowercase

        Returns:
            str: Preprocessed text
        """
        text = text.strip()
        if to_lower:
            text = text.lower()
        return text


class TiktokenTokenizer(BaseTokenizer):
    """
    Tiktoken-based tokenizer supporting multilingual content.

    Uses tiktoken encoding for robust tokenization of Chinese, English
    and other languages. Falls back to simple splitting if tiktoken fails.
    """

    name: str = Field(default="tiktoken", description="Tiktoken tokenizer")
    encoding_name: str = Field(
        default="cl100k_base", description="Tiktoken encoding name"
    )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using tiktoken encoder.

        Args:
            text: Input text to tokenize

        Returns:
            List[str]: List of token strings
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(self.encoding_name)
            tokens = encoding.encode(text)
            # Convert token ids back to strings for comparison
            token_strings = [encoding.decode([token]) for token in tokens]
            return token_strings
        except Exception:
            # Fallback to simple splitting if tiktoken fails
            return text.split()


class JiebaTokenizer(BaseTokenizer):
    """
    Jieba-based tokenizer for Chinese text processing.

    Provides Chinese word segmentation using jieba library with optional
    Chinese character filtering and preprocessing capabilities.
    """

    name: str = Field(default="jieba", description="Jieba Chinese tokenizer")
    chinese_only: bool = Field(
        default=False, description="Whether to keep only Chinese characters"
    )

    def _preserve_chinese(self, text: str) -> str:
        """
        Preserve only Chinese characters.

        Args:
            text: Input text

        Returns:
            str: Text with only Chinese characters
        """
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return "".join(chinese_chars)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.

        Args:
            text: Input text to tokenize

        Returns:
            List[str]: List of token strings

        Raises:
            ImportError: If jieba library is not installed
        """
        try:
            import jieba

            if self.chinese_only:
                text = self._preserve_chinese(text)
            return list(jieba.cut(text))
        except ImportError:
            raise ImportError(
                "jieba library required for Chinese tokenization: pip install jieba"
            )


class SimpleTokenizer(BaseTokenizer):
    """
    Simple whitespace-based tokenizer.

    Basic tokenizer that splits text on whitespace. Used as fallback
    when other tokenizers are not available or fail.
    """

    name: str = Field(default="simple", description="Simple whitespace tokenizer")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace.

        Args:
            text: Input text to tokenize

        Returns:
            List[str]: List of token strings
        """
        return text.split()


def get_tokenizer(
    tokenizer_type: str = "tiktoken",
    encoding_name: str = "cl100k_base",
    chinese_only: bool = False,
    **kwargs,
) -> BaseTokenizer:
    """
    Factory function to create tokenizer instances.

    Args:
        tokenizer_type: Type of tokenizer ("tiktoken", "jieba", "simple")
        encoding_name: Tiktoken encoding name (for tiktoken tokenizer)
        chinese_only: Whether to keep only Chinese characters (for jieba tokenizer)
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        BaseTokenizer: Tokenizer instance

    Raises:
        ValueError: If tokenizer_type is not supported
    """
    if tokenizer_type == "tiktoken":
        return TiktokenTokenizer(encoding_name=encoding_name, **kwargs)
    elif tokenizer_type == "jieba":
        return JiebaTokenizer(chinese_only=chinese_only, **kwargs)
    elif tokenizer_type == "simple":
        return SimpleTokenizer(**kwargs)
    else:
        raise ValueError(
            f"Unsupported tokenizer type: {tokenizer_type}. "
            f"Supported types: tiktoken, jieba, simple"
        )
