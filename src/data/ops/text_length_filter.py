from typing import List, Dict, Any, Optional
from loguru import logger

from base import BaseData
from base_operator import Operator, OperatorFactory

class TextLengthFilter(Operator[BaseData]):
    """
    Filter texts based on their length.
    """

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
        super().__init__(name, config)
        self.min_length = min_length
        self.max_length = max_length

    def process_dataset(self, items: List[BaseData]) -> List[BaseData]:
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
            
            # process input
            for input_item in item.evaluation_sample.input:
                if input_item.content:
                    texts.append(input_item.content)
            
            # process output
            for output_item in item.evaluation_sample.outputs:
                if output_item.content:
                    texts.append(output_item.content)
            
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
def create_text_length_filter(config: Dict[str, Any]) -> TextLengthFilter:
    """
    Create a text length filter operator from configuration.

    Args:
        config: Configuration dictionary containing:
            - name: Name of the operator
            - min_length: Minimum text length (optional)
            - max_length: Maximum text length (optional)

    Returns:
        TextLengthFilter instance
    """
    name = config.get('name', 'text_length_filter')
    min_length = config.get('min_length', 10)
    max_length = config.get('max_length', 1000)
    
    return TextLengthFilter(
        name=name,
        min_length=min_length,
        max_length=max_length,
        config=config
    ) 