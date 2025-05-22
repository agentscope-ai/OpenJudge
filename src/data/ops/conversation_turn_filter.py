from typing import List, Dict, Any, Optional
from loguru import logger

from ..base import BaseData
from ..base_operator import Operator, OperatorFactory

class ConversationTurnFilter(Operator[BaseData]):
    """
    Filter conversations based on the number of turns in the input.
    A turn is defined as a single message in the conversation.
    """

    def __init__(self, 
                 name: str,
                 min_turns: int = 1,
                 max_turns: int = 100,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the conversation turn filter.

        Args:
            name: Name of the operator
            min_turns: Minimum number of turns required (inclusive)
            max_turns: Maximum number of turns allowed (inclusive)
            config: Additional configuration parameters
        """
        super().__init__(name, config)
        self.min_turns = min_turns
        self.max_turns = max_turns

    def process_dataset(self, items: List[BaseData]) -> List[BaseData]:
        """
        Filter conversations based on the number of turns.

        Args:
            items: List of BaseData items to process

        Returns:
            List of BaseData items that meet the turn count criteria
        """
        try:
            filtered_items = []
            for item in items:
                # Count the number of user turns in the input
                num_turns = sum(1 for input_item in item.evaluation_sample.input 
                              if input_item.role == 'user') if item.evaluation_sample.input else 0
                
                # Check if the number of turns is within the specified range
                if self.min_turns <= num_turns <= self.max_turns:
                    filtered_items.append(item)
                else:
                    logger.debug(f"Filtered out conversation with {num_turns} user turns "
                               f"(min: {self.min_turns}, max: {self.max_turns})")
            
            return filtered_items
        except Exception as e:
            logger.error(f"Error in conversation turn filtering: {str(e)}")
            return items


# Register the operator with the factory
@OperatorFactory.register('conversation_turn_filter')
def create_conversation_turn_filter(operator_config: Dict[str, Any]) -> Operator:
    """
    Create a conversation turn filter operator from configuration.

    Args:
        operator_config: Configuration dictionary containing:
            - name: Name of the operator
            - config: Configuration dictionary containing:
                - min_turns: Minimum number of turns (default: 1)
                - max_turns: Maximum number of turns (default: 100)

    Returns:
        A configured ConversationTurnFilter operator
    """
    name = operator_config.get('name', 'conversation_turn_filter')
    config = operator_config.get('config', {})
    min_turns = config.get('min_turns', 1)
    max_turns = config.get('max_turns', 100)
    
    return ConversationTurnFilter(
        name=name,
        min_turns=min_turns,
        max_turns=max_turns,
        config=config
    ) 