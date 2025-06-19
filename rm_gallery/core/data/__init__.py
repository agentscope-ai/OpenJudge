"""
Data module initialization - centralized imports for data processing components.
Provides standardized access to data operations, loaders, and strategies.
"""
from rm_gallery.core.data.load.chat_message import ChatMessageConverter
from rm_gallery.core.data.load.huggingface import GenericConverter
from rm_gallery.core.data.process.ops.filter.conversation_turn_filter import (
    ConversationTurnFilter,
)
from rm_gallery.core.data.process.ops.filter.text_length_filter import TextLengthFilter

OPERATORS = {
    "conversation_turn_filter": ConversationTurnFilter,
    "text_length_filter": TextLengthFilter,
}

LOAD_STRATEGIES = {
    "chat_message": ChatMessageConverter,
    "huggingface": GenericConverter,
}
