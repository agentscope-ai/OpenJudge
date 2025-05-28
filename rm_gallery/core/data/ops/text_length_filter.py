from typing import List, Dict, Any, Optional
from loguru import logger
from pydantic import Field

from ..process import BaseOperator, OperatorFactory
from ..schema import DataSample

class TextLengthFilter(BaseOperator):
    """
    Filter texts based on their length.
    """

    min_length: int = Field(default=10, description="Minimum text length required (inclusive)")
    max_length: int = Field(default=1000, description="Maximum text length allowed (inclusive)")

    def __init__(self, 
                 name: str,
                 min_length: int = 10,
                 max_length: int = 1000,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text length filter.

        Args:
            name: Name of the operator
            min_length: Minimum text length required (inclusive)
            max_length: Maximum text length allowed (inclusive)
            config: Additional configuration parameters
        """
        super().__init__(
            name=name, 
            config=config,
            min_length=min_length,
            max_length=max_length
        )

    def process_dataset(self, items: List[DataSample]) -> List[DataSample]:
        """
        Filter items based on text length.

        Args:
            items: List of data items to process

        Returns:
            Filtered list of items
        """
        filtered_items = []
        for item in items:
            # get all input and output texts
            texts = []
            
            # process input from history
            if item.input:
                for input_item in item.input:
                    if input_item.content:
                        texts.append(input_item.content)
            
            # process output from answers
            if item.output:
                for output_item in item.output:
                    if hasattr(output_item, 'answer') and output_item.answer and output_item.answer.content:
                        texts.append(output_item.answer.content)
            
            # calculate total length
            total_length = sum(len(text) for text in texts)
            
            if self.min_length <= total_length <= self.max_length:
                filtered_items.append(item)
            else:
                pass
                #logger.debug(f"Filtered out item with total length {total_length}")
        return filtered_items

# Register the operator with the factory
@OperatorFactory.register('rm_text_length_filter')
def create_text_length_filter(operator_config: Dict[str, Any]) -> BaseOperator:
    """
    Create a text length filter operator from configuration.

    Args:
        operator_config: Configuration dictionary containing:
            - name: Name of the operator
            - config: Configuration dictionary containing:
                - min_length: Minimum text length (optional)
                - max_length: Maximum text length (optional)

    Returns:
        TextLengthFilter instance
    """
    name = operator_config.get('name', 'text_length_filter')
    config = operator_config.get('config', {})
    min_length = config.get('min_length', 10)
    max_length = config.get('max_length', 1000)
    
    return TextLengthFilter(
        name=name,
        min_length=min_length,
        max_length=max_length,
        config=config
    ) 