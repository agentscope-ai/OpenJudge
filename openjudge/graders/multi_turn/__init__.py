# -*- coding: utf-8 -*-
"""
Multi-turn Conversation Graders for OpenJudge.

This module provides graders for evaluating multi-turn conversation capabilities
of language models, organized according to a three-level capability framework.

Graders included:
    - ContextMemoryGrader (CM): Evaluates ability to recall early conversation details
    - AnaphoraResolutionGrader (AR): Evaluates ability to resolve pronouns
    - TopicSwitchGrader (TS): Evaluates ability to recognize sudden topic changes
    - SelfCorrectionGrader (SC): Evaluates ability to correct errors based on feedback
    - InstructionClarificationGrader (IC): Evaluates ability to ask for clarification
    - ProactiveInteractionGrader (PI): Evaluates ability to proactively engage in conversation
    - ResponseRepetitionGrader (RR): Evaluates whether response repeats historical content
"""

from openjudge.graders.multi_turn.anaphora_resolution_grader import (
    AnaphoraResolutionGrader,
)
from openjudge.graders.multi_turn.context_memory_grader import ContextMemoryGrader
from openjudge.graders.multi_turn.instruction_clarification_grader import (
    InstructionClarificationGrader,
)
from openjudge.graders.multi_turn.proactive_interaction_grader import (
    ProactiveInteractionGrader,
)
from openjudge.graders.multi_turn.response_repetition_grader import (
    ResponseRepetitionGrader,
)
from openjudge.graders.multi_turn.self_correction_grader import SelfCorrectionGrader
from openjudge.graders.multi_turn.topic_switch_grader import TopicSwitchGrader

__all__ = [
    "ContextMemoryGrader",
    "AnaphoraResolutionGrader",
    "TopicSwitchGrader",
    "SelfCorrectionGrader",
    "InstructionClarificationGrader",
    "ProactiveInteractionGrader",
    "ResponseRepetitionGrader",
]
