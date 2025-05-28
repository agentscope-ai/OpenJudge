from ..process import OperatorFactory
from .conversation_turn_filter import create_conversation_turn_filter
from .text_length_filter import create_text_length_filter
from .group_filter import create_group_filter

# Define all operators
OPERATORS = {
    'conversation_turn_filter': create_conversation_turn_filter,
    'rm_text_length_filter': create_text_length_filter,
    'group_filter': create_group_filter
}

def register_all_operators():
    """Register all operators with the factory"""
    for op_name, op_creator in OPERATORS.items():
        OperatorFactory.register(op_name)(op_creator)

# Register operators when the module is imported
register_all_operators() 