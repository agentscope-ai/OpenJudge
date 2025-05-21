from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import random

from base import BaseData
from base_operator import Operator, OperatorFactory

class GroupFilter(Operator[BaseData]):
    """
    Filter and group data items into different sets (e.g., train/test).
    """

    def __init__(self, 
                 name: str,
                 train_ratio: float = 0.8,
                 test_ratio: float = 0.2,
                 seed: int = 42,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the group filter.

        Args:
            name: Name of the operator
            train_ratio: Ratio of data to be used for training (default: 0.8)
            test_ratio: Ratio of data to be used for testing (default: 0.2)
            seed: Random seed for reproducibility (default: 42)
            config: Additional configuration parameters
        """
        super().__init__(name, config)
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Validate ratios
        if abs(train_ratio + test_ratio - 1.0) > 1e-6:
            logger.warning(f"Train ratio ({train_ratio}) + test ratio ({test_ratio}) != 1.0")
        
        # Set random seed
        random.seed(seed)

    def process_dataset(self, items: List[BaseData]) -> List[BaseData]:
        """
        Split items into train and test sets.

        Args:
            items: List of data items to process

        Returns:
            List of data items with group information added
        """
        try:
            # Shuffle items
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            
            # Calculate split indices
            n_items = len(shuffled_items)
            n_train = int(n_items * self.train_ratio)
            
            # Split items
            train_items = shuffled_items[:n_train]
            test_items = shuffled_items[n_train:]
            
            # Add group information to items
            for item in train_items:
                if not hasattr(item, 'extra_metadata'):
                    item.extra_metadata = {}
                item.extra_metadata['group'] = 'train'
            
            for item in test_items:
                if not hasattr(item, 'extra_metadata'):
                    item.extra_metadata = {}
                item.extra_metadata['group'] = 'test'
            
            logger.info(f"Split {n_items} items into {len(train_items)} train and {len(test_items)} test items")
            
            # Return all items
            return shuffled_items
            
        except Exception as e:
            logger.error(f"Error in group filtering: {str(e)}")
            return items

# Register the operator with the factory
@OperatorFactory.register('group_filter')
def create_group_filter(config: Dict[str, Any]) -> GroupFilter:
    """
    Create a group filter operator from configuration.

    Args:
        config: Configuration dictionary containing:
            - name: Name of the operator
            - train_ratio: Ratio of data for training (optional)
            - test_ratio: Ratio of data for testing (optional)
            - seed: Random seed (optional)

    Returns:
        GroupFilter instance
    """
    name = config.get('name', 'group_filter')
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.2)
    seed = config.get('seed', 42)
    
    return GroupFilter(
        name=name,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=seed,
        config=config
    ) 