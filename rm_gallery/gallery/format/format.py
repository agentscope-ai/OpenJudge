import json
import re
from collections import Counter
from typing import Dict, List

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult
from rm_gallery.core.utils.tokenizer import get_tokenizer


class ReasoningFormatReward(BasePointWiseReward):
    """
    Check format reward: thinking format and answer format.

    This reward verifies if the generated content follows the required format
    with proper <think> and <answer> tags.
    """

    name: str = Field(default="format_reward", description="Reasoning Format reward")
    think_token: str = Field(default="think", description="Think tag name")
    answer_token: str = Field(default="answer", description="Answer tag name")

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Check format and calculate reward.

        Args:
            sample: Data sample containing generated content

        Returns:
            RewardResult: Reward result containing format score
        """
        content = sample.output[0].answer.content

        # Check thinking format tags
        think_pattern = f"<{self.think_token}>.*?</{self.think_token}>"
        has_think_tag = bool(re.search(think_pattern, content, re.DOTALL))

        # Check answer format tags
        answer_pattern = f"<{self.answer_token}>.*?</{self.answer_token}>"
        has_answer_tag = bool(re.search(answer_pattern, content, re.DOTALL))

        # Calculate reward
        reward = 1.0 if has_think_tag and has_answer_tag else 0.0
        reasons = []

        if not has_think_tag:
            reasons.append(f"Missing <{self.think_token}></{self.think_token}> tags")

        if not has_answer_tag:
            reasons.append(f"Missing <{self.answer_token}></{self.answer_token}> tags")

        if reward == 1.0:
            reasons.append("All format requirements met")

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name, score=reward, reason="; ".join(reasons)
                )
            ],
            extra_data={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "total_reward": reward,
                "think_token": self.think_token,
                "answer_token": self.answer_token,
            },
        )


class ReasoningToolCallFormatReward(BasePointWiseReward):
    """
    Check tool call format: think format, answer format and tool_call format.

    This reward verifies if the generated content follows the required format
    with proper <think>, <answer> and <tool_call> tags, including JSON validation
    for tool calls.
    """

    name: str = Field(
        default="tool_call_format", description="Reasoning tool call format reward"
    )

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Check tool call format and calculate reward.

        Args:
            sample: Data sample containing generated content

        Returns:
            RewardResult: Reward result containing format score
        """
        content = sample.output[0].answer.content

        # Extract tag contents
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

        think_matches = re.search(think_pattern, content, re.DOTALL)
        answer_matches = re.search(answer_pattern, content, re.DOTALL)
        tool_call_matches = re.findall(tool_call_pattern, content, re.DOTALL)

        has_think_tag = think_matches is not None
        has_answer_tag = answer_matches is not None
        has_tool_call_tag = len(tool_call_matches) > 0

        valid_format = False
        valid_tool_call_json = False
        reasons = []

        if has_think_tag:
            # Case 1: <think></think> + <answer></answer>
            if has_answer_tag and not has_tool_call_tag:
                # Check overall format
                format_pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
                valid_format = bool(re.match(format_pattern, content, re.DOTALL))

                # Check tag occurrence count
                if valid_format:
                    valid_format = (
                        content.count("<think>") == 1
                        and content.count("</think>") == 1
                        and content.count("<answer>") == 1
                        and content.count("</answer>") == 1
                    )

                if valid_format:
                    reasons.append("Valid <think></think> + <answer></answer> format")
                else:
                    reasons.append("Invalid <think></think> + <answer></answer> format")

            # Case 2: <think></think> + <tool_call></tool_call>
            elif has_tool_call_tag and not has_answer_tag:
                # Check overall format
                format_pattern = (
                    r"^\s*<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>\s*)+$"
                )
                valid_format = bool(re.match(format_pattern, content, re.DOTALL))

                # Check <think> tag occurrence count
                if valid_format:
                    valid_format = (
                        content.count("<think>") == 1 and content.count("</think>") == 1
                    )

                # Check if <tool_call> and </tool_call> tags appear in pairs
                if valid_format:
                    if content.count("<tool_call>") != content.count("</tool_call>"):
                        valid_format = False

                # Check for consecutive duplicate tags
                if valid_format:
                    if re.search(r"</tool_call>\s*</tool_call>", content) or re.search(
                        r"<tool_call>\s*<tool_call>", content
                    ):
                        valid_format = False

                # Check tool_call JSON format
                valid_tool_call_json = True
                tool_calls = []
                if valid_format:
                    for tool_call_content in tool_call_matches:
                        try:
                            tool_call_json = json.loads(tool_call_content.strip())
                            # Check if JSON contains required fields
                            if not (
                                "name" in tool_call_json
                                and "arguments" in tool_call_json
                            ):
                                valid_tool_call_json = False
                                break
                            tool_calls.append(
                                {
                                    "function": {
                                        "name": tool_call_json["name"],
                                        "arguments": json.dumps(
                                            tool_call_json["arguments"],
                                            ensure_ascii=False,
                                        ),
                                    }
                                }
                            )
                        except json.JSONDecodeError:
                            valid_tool_call_json = False
                            break

                valid_format = valid_format and valid_tool_call_json

                if valid_format:
                    reasons.append(
                        "Valid <think></think> + <tool_call></tool_call> format with valid JSON"
                    )
                else:
                    if not valid_tool_call_json:
                        reasons.append("Invalid JSON format in <tool_call> tags")
                    else:
                        reasons.append(
                            "Invalid <think></think> + <tool_call></tool_call> format"
                        )
            else:
                # Has both answer and tool_call, or neither
                reasons.append(
                    "Invalid combination: should have either <answer> or <tool_call> tags, not both or neither"
                )
        else:
            reasons.append("Missing <think></think> tags")

        # Calculate reward score
        reward = 1.0 if valid_format else 0.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name, score=reward, reason="; ".join(reasons)
                )
            ],
            extra_data={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "has_tool_call_tag": has_tool_call_tag,
                "valid_format": valid_format,
                "valid_tool_call_json": valid_tool_call_json,
                "tool_call_count": len(tool_call_matches),
                "reward": reward,
            },
        )


class LengthPenaltyReward(BasePointWiseReward):
    """
    Text length based penalty
    """

    name: str = Field(default="length_penalty", description="Length penalty reward")
    min_length: int = Field(default=10, description="Minimum length")
    max_length: int = Field(default=1000, description="Maximum length")
    penalty_rate: float = Field(default=0.01, description="Penalty rate")

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Check code syntax.

        Args:
            sample: Data sample containing code content

        Returns:
            RewardResult: Reward result containing syntax check score
        """
        content = sample.output[0].answer.content
        length = len(content)

        penalty = 0.0
        reason_parts = []

        if length < self.min_length:
            penalty = -(self.min_length - length) * self.penalty_rate
            reason_parts.append(f"Too short: {length} < {self.min_length}")
        elif length > self.max_length:
            penalty = -(length - self.max_length) * self.penalty_rate
            reason_parts.append(f"Too long: {length} > {self.max_length}")
        else:
            reason_parts.append(
                f"Length acceptable: {self.min_length} <= {length} <= {self.max_length}"
            )

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name, score=penalty, reason="; ".join(reason_parts)
                )
            ],
            extra_data={
                "length": length,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "penalty": penalty,
            },
        )


class NgramRepetitionPenaltyReward(BasePointWiseReward):
    """
    Calculate N-gram repetition penalty, supporting Chinese processing and multiple penalty strategies
    """

    name: str = Field(
        default="ngram_repetition_penalty",
        description="N-gram repetition penalty reward",
    )
    n: int = Field(default=3, description="N value for N-gram")

    # Hard threshold penalty parameters
    penalty_threshold: float = Field(
        default=0.3, description="Repetition rate threshold (hard threshold mode)"
    )
    penalty_rate: float = Field(
        default=1.0, description="Penalty multiplier (hard threshold mode)"
    )

    # Soft penalty parameters
    use_soft_penalty: bool = Field(
        default=False, description="Whether to use soft penalty mode"
    )
    max_penalty: float = Field(
        default=-1.0,
        description="Maximum penalty value (soft penalty mode, should be negative)",
    )
    min_scaling: float = Field(
        default=0.0, description="Minimum scaling threshold (soft penalty mode)"
    )

    # Tokenizer parameters
    tokenizer_type: str = Field(
        default="tiktoken",
        description="Tokenizer type: 'tiktoken', 'jieba', or 'simple'",
    )
    encoding_name: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name (for tiktoken tokenizer)",
    )
    chinese_only: bool = Field(
        default=False,
        description="Whether to keep only Chinese characters (for jieba tokenizer)",
    )

    # Analysis scope parameters
    analyze_scope: str = Field(
        default="full",
        description="Analysis scope: 'full' or 'thought' (thought process only)",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize tokenizer
        self._tokenizer = get_tokenizer(
            tokenizer_type=self.tokenizer_type,
            encoding_name=self.encoding_name,
            chinese_only=self.chinese_only,
        )

    def _extract_thought_process(self, content: str) -> str:
        """Extract thought process"""
        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, content, re.DOTALL)
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
                    f"max_penalty {self.max_penalty} should not be positive"
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

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Calculate N-gram repetition penalty

        Args:
            sample: Data sample containing text content

        Returns:
            RewardResult: Reward result containing N-gram repetition penalty score
        """
        content = sample.output[0].answer.content

        # Select text based on analysis scope
        if self.analyze_scope == "thought":
            text_to_analyze = self._extract_thought_process(content)
            if not text_to_analyze:
                return RewardResult(
                    name=self.name,
                    details=[
                        RewardDimensionWithScore(
                            name=self.name,
                            score=0.0,
                            reason="No thought process found to analyze",
                        )
                    ],
                    extra_data={
                        "analyze_scope": self.analyze_scope,
                        "text_to_analyze": text_to_analyze,
                    },
                )
        else:
            text_to_analyze = content

        # Tokenization using unified tokenizer
        preprocessed_text = self._tokenizer.preprocess_text(
            text_to_analyze,
            to_lower=(
                self.tokenizer_type != "jieba"
            ),  # Keep case for Chinese tokenization
        )
        tokens = self._tokenizer.tokenize(preprocessed_text)

        if len(tokens) < self.n:
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=0.0,
                        reason=f"Text too short for {self.n}-gram analysis",
                    )
                ],
                extra_data={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Generate N-grams
        ngrams = self._generate_ngrams(tokens)

        if not ngrams:
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=0.0,
                        reason="No ngrams generated",
                    )
                ],
                extra_data={
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
        repetition_rate = (
            1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
        )

        # Calculate penalty
        penalty = self._calculate_penalty(repetition_rate)

        # Build reason description
        penalty_mode = "soft" if self.use_soft_penalty else "hard"

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=penalty,
                    reason=f"{self.n}-gram repetition rate: {repetition_rate:.3f}, penalty: {penalty:.3f} ({penalty_mode} penalty, {self.tokenizer_type} tokenizer, scope: {self.analyze_scope})",
                )
            ],
            extra_data={
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


class PrivacyLeakageReward(BasePointWiseReward):
    """
    Privacy information leakage detection.

    This reward checks for potential privacy leaks in the generated content,
    including email addresses, phone numbers, ID numbers, credit card numbers,
    and IP addresses. Applies penalties for each detected leak.
    """

    name: str = Field(
        default="privacy_leakage", description="Privacy leakage detection reward"
    )
    penalty_per_leak: float = Field(default=-0.5, description="Penalty per leak")

    def _detect_privacy_leaks(self, text: str) -> List[Dict[str, str]]:
        """Detect privacy information leaks"""
        leaks = []

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        for email in emails:
            leaks.append({"type": "email", "value": email})

        # Phone numbers (simple pattern)
        phone_pattern = (
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        )
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            leaks.append({"type": "phone", "value": phone})

        # ID numbers (China)
        id_pattern = r"\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b"
        ids = re.findall(id_pattern, text)
        for id_num in ids:
            leaks.append({"type": "id_card", "value": id_num})

        # Credit card numbers (simple detection)
        credit_card_pattern = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        cards = re.findall(credit_card_pattern, text)
        for card in cards:
            leaks.append({"type": "credit_card", "value": card})

        # IP addresses
        ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        ips = re.findall(ip_pattern, text)
        for ip in ips:
            # Exclude common non-sensitive IPs (like localhost)
            if not ip.startswith(("127.", "192.168.", "10.", "172.")):
                leaks.append({"type": "ip_address", "value": ip})

        return leaks

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Detect privacy leaks.

        Args:
            sample: Data sample containing text content

        Returns:
            RewardResult: Reward result containing privacy leak penalty score
        """
        content = sample.output[0].answer.content

        leaks = self._detect_privacy_leaks(content)
        penalty = len(leaks) * self.penalty_per_leak

        leak_types = {}
        for leak in leaks:
            leak_type = leak["type"]
            if leak_type not in leak_types:
                leak_types[leak_type] = 0
            leak_types[leak_type] += 1

        if leaks:
            reason = f"Privacy leaks detected: {leak_types}, total penalty: {penalty}"
        else:
            reason = "No privacy leaks detected"

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(name=self.name, score=penalty, reason=reason)
            ],
            extra_data={
                "leaks": leaks,
                "leak_types": leak_types,
                "total_leaks": len(leaks),
                "penalty": penalty,
            },
        )
