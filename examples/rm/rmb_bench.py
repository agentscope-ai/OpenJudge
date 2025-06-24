"""
RMB class inheriting from RouterComposition to manage reward model routing configurations for various task types.

Attributes:
    router (Dict[str, dict]): Mapping of task types to corresponding reward classes, supporting multiple NLP task evaluations.
    The dictionary contains:
        - Keys: Task type strings (e.g., "brainstorming", "chat")
        - Values: Dictionaries with "cls" key pointing to specific reward classes
        - Special key "general": Default HelpfulnessPointWiseReward for general cases
"""

from typing import Dict

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.composition import RouterComposition
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.brainstorming import (
    BrainstormingListWiseReward,
)
from rm_gallery.gallery.rm.alignment.helpfulness.chat import ChatListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.classification import (
    ClassificationListWiseReward,
)
from rm_gallery.gallery.rm.alignment.helpfulness.closed_qa import ClosedQAListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.code import CodeListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.generation import (
    GenerationListWiseReward,
)
from rm_gallery.gallery.rm.alignment.helpfulness.open_qa import OpenQAListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.reasoning import (
    ReasoningListWiseReward,
)
from rm_gallery.gallery.rm.alignment.helpfulness.rewrite import RewriteListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.role_playing import (
    RolePlayingListWiseReward,
)
from rm_gallery.gallery.rm.alignment.helpfulness.summarization import (
    SummarizationListWiseReward,
)
from rm_gallery.gallery.rm.alignment.helpfulness.translation import (
    TranslationListWiseReward,
)


class RMBBenchRouter(RouterComposition):
    """
    RMB class inheriting from RouterComposition to manage reward model routing configurations for various task types.

    The class provides task-specific reward model selection through a router dictionary
    that maps task types to corresponding reward implementation classes.
    """

    rewards: Dict[str, dict] = {
        "brainstorming": {"cls": BrainstormingListWiseReward},
        "chat": {"cls": ChatListWiseReward},
        "classification": {"cls": ClassificationListWiseReward},
        "closed_qa": {"cls": ClosedQAListWiseReward},
        "code": {"cls": CodeListWiseReward},
        "generation": {"cls": GenerationListWiseReward},
        "open_qa": {"cls": OpenQAListWiseReward},
        "reasoning": {"cls": ReasoningListWiseReward},
        "rewrite": {"cls": RewriteListWiseReward},
        "role_playing": {"cls": RolePlayingListWiseReward},
        "summarization": {"cls": SummarizationListWiseReward},
        "translation": {"cls": TranslationListWiseReward},
        "general": {"cls": BaseHelpfulnessListwiseReward},
    }

    def _condition(self, sample: DataSample) -> str:
        """
        Determine reward model type based on data sample metadata.

        Extracts the 'subset' field from sample metadata to identify the appropriate
        reward model type. Returns lowercase version of the subset value when available,
        falls back to 'general' type when subset field is missing.

        Args:
            sample (DataSample): Input data sample containing metadata information

        Returns:
            str: Task type identifier for reward model selection:
                - Lowercase version of sample.metadata["subset"] if present
                - "general" as default fallback when subset field is missing
        """
        try:
            cond = sample["meta"]["category_path"].split("/")[-2].lower()

        except Exception:
            # Fallback to general reward model when path extraction fails
            cond = "general"

        if cond not in self.rewards:
            cond = "general"
        return cond
