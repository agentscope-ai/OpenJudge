# -*- coding: utf-8 -*-
"""
Model integrations module from AgentScope
"""

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.minimax_chat_model import MiniMaxChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.qwen_vl_model import QwenVLModel

__all__ = [
    "BaseChatModel",
    "MiniMaxChatModel",
    "OpenAIChatModel",
    "QwenVLModel",
]
