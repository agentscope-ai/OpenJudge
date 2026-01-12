# -*- coding: utf-8 -*-
"""
- NgramRepetitionPenaltyGrader: Calculates N-gram repetition penalty with Chinese support
"""

import re
from collections import Counter
from typing import Any, List, Literal

from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
from openjudge.utils.tokenizer import TokenizerEnum, get_tokenizer


class NgramRepetitionPenaltyGrader(BaseGrader):
    """
    Calculate N-gram repetition penalty supporting Chinese processing and multiple penalty strategies.
    """

    def __init__(
        self,
        n: int = 3,
        penalty_threshold: float = 0.3,
        penalty_rate: float = 1.0,
        use_soft_penalty: bool = False,
        max_penalty: float = -1.0,
        min_scaling: float = 0.0,
        tokenizer_type: TokenizerEnum = TokenizerEnum.tiktoken,
        encoding_name: str = "cl100k_base",
        chinese_only: bool = False,
        analyze_scope: Literal["thought", "full"] = "full",
    ):
        """
        Initialize the NgramRepetitionPenaltyGrader.
        Args:
            n: N value for N-gram
            penalty_threshold: Threshold for hard threshold penalty
            penalty_rate: Penalty rate for each repetition
            use_soft_penalty: Use soft threshold penalty
            max_penalty: Maximum penalty value
            min_scaling: Minimum scaling factor for soft threshold penalty
            tokenizer_type: Tokenizer type (tiktoken, jieba, simple)
            encoding_name: Encoding name for tiktoken
            chinese_only: Whether to keep only Chinese characters (for jieba tokenizer)
            description: Description of the grader
        """
        super().__init__(
            name="ngram_repetition_penalty",
            mode=GraderMode.POINTWISE,
            description="Calculate N-gram repetition penalty supporting Chinese processing "
            "and multiple penalty strategies.",
        )

        self.n = n
        self.penalty_threshold = penalty_threshold
        self.penalty_rate = penalty_rate
        self.use_soft_penalty = use_soft_penalty
        self.analyze_scope = analyze_scope
        self.chinese_only = chinese_only
        self.encoding_name = encoding_name
        self.tokenizer_type = tokenizer_type
        self.chinese_only = chinese_only
        self.max_penalty = max_penalty
        self.min_scaling = min_scaling
        self.tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            encoding_name=encoding_name,
            chinese_only=chinese_only,
        )

        self._think_pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)

    def _extract_thought_process(self, content: str) -> str:
        """Extract thought process"""
        matches = self._think_pattern.findall(content)
        return " ".join(matches) if matches else ""

    def _generate_ngrams(self, tokens: List[str]) -> List[tuple]:
        """Generate N-grams"""
        if len(tokens) < self.n:
            return []

        # Use unified approach for all tokenizers
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngrams.append(tuple(tokens[i : i + self.n]))
        return ngrams

    def _calculate_penalty(self, repetition_rate: float) -> float:
        """Calculate penalty value"""
        if self.use_soft_penalty:
            # Soft penalty mode
            if self.max_penalty > 0:
                raise ValueError(
                    f"max_penalty {self.max_penalty} should not be positive",
                )

            scaling = repetition_rate
            if scaling < self.min_scaling:
                scaling = 0.0
            elif scaling > self.min_scaling:
                scaling = (scaling - self.min_scaling) / (1 - self.min_scaling)

            return scaling * self.max_penalty
        else:
            # Hard threshold mode (original logic)
            if repetition_rate > self.penalty_threshold:
                return -(repetition_rate - self.penalty_threshold) * self.penalty_rate
            return 0.0

    async def aevaluate(self, response: str, **kwargs: Any) -> GraderScore:
        """
        Calculate N-gram repetition penalty for text content.

        This method evaluates the repetitiveness of text content by calculating
        the N-gram repetition rate and applying penalties accordingly. It supports
        multiple tokenization methods and penalty strategies, including both hard
        threshold and soft penalty modes.

        Args:
            response: The text content to evaluate for N-gram repetitions.
            **kwargs: Additional keyword arguments (not used in current implementation).

        Returns:
            GraderScore: A GraderScore object containing:
                - score: The calculated penalty (negative value or 0.0)
                - reason: Explanation of the evaluation result with repetition rate and penalty
                - metadata: Dictionary with detailed information:
                    * repetition_rate: Rate of repeated N-grams
                    * unique_ngrams: Number of unique N-grams
                    * total_ngrams: Total number of N-grams
                    * penalty: The calculated penalty value
                    * most_common_ngrams: Top 5 most frequently occurring N-grams
                    * analyze_scope: Scope of analysis (thought or full)
                    * tokenizer_type: Type of tokenizer used
                    * use_soft_penalty: Whether soft penalty mode is enabled
                    * penalty_mode: Either "soft" or "hard" depending on configuration

        Examples:
            >>> grader = NgramRepetitionPenaltyGrader(n=3, penalty_threshold=0.3)
            >>> result = await grader.aevaluate("This is a test. This is a test. This is a test.")
            >>> print(result.score < 0)
            True

            >>> grader = NgramRepetitionPenaltyGrader(n=2, use_soft_penalty=True, max_penalty=-0.5)
            >>> result = await grader.aevaluate("Different words forming different bigrams here")
            >>> print(result.score)
            0.0
        """
        # Select text based on analysis scope
        if self.analyze_scope == "thought":
            text_to_analyze = self._extract_thought_process(response)
            if not text_to_analyze:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No thought process found to analyze",
                    metadata={
                        "analyze_scope": self.analyze_scope,
                        "text_to_analyze": text_to_analyze,
                    },
                )

        else:
            text_to_analyze = response

        # Tokenization using unified tokenizer
        preprocessed_text = self.tokenizer.preprocess_text(
            text_to_analyze,
            to_lower=(self.tokenizer_type != "jieba"),  # Keep case for Chinese tokenization
        )
        tokens = self.tokenizer.tokenize(preprocessed_text)

        if len(tokens) < self.n:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"Text too short for {self.n}-gram analysis",
                metadata={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Generate N-grams
        ngrams = self._generate_ngrams(tokens)

        if not ngrams:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No ngrams response",
                metadata={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Calculate repetition rate
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)
        repetition_rate = 1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0

        # Calculate penalty
        penalty = self._calculate_penalty(repetition_rate)

        # Build reason description
        penalty_mode = "soft" if self.use_soft_penalty else "hard"
        return GraderScore(
            name=self.name,
            score=penalty,
            reason=f"{self.n}-gram repetition rate: {repetition_rate:.3f}, "
            f"penalty: {penalty:.3f} ({penalty_mode} penalty, "
            f"{self.tokenizer_type} tokenizer, scope: {self.analyze_scope})",
            metadata={
                "repetition_rate": repetition_rate,
                "unique_ngrams": unique_ngrams,
                "total_ngrams": total_ngrams,
                "penalty": penalty,
                "most_common_ngrams": ngram_counts.most_common(5),
                "analyze_scope": self.analyze_scope,
                "tokenizer_type": self.tokenizer_type,
                "use_soft_penalty": self.use_soft_penalty,
                "penalty_mode": penalty_mode,
            },
        )
