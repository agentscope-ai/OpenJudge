import pytest
from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for example strategy registration
from rm_gallery.core.data.load.base import create_load_module
from rm_gallery.core.data.process.ops.base import OperatorFactory
from rm_gallery.core.data.process.process import create_process_module
from rm_gallery.core.data.schema import BaseDataSet, DataSample


@pytest.fixture
def load_config():
    """Load configuration fixture for test data"""
    return {
        "path": "./data/reward-bench-2/data/test-00000-of-00001.parquet",
        "limit": 50,  # Use smaller limit for faster processing tests
        "huggingface_split": "test",
    }


@pytest.fixture
def dataset_info():
    """Dataset information fixture"""
    return {
        "name": "allenai/reward-bench-2",
        "type": "local",
        "data_source": "rewardbench2",
    }


@pytest.fixture
def conversation_filter_config():
    """Conversation turn filter configuration"""
    return {
        "type": "filter",
        "name": "conversation_turn_filter",
        "config": {"min_turns": 1, "max_turns": 10},
    }


@pytest.fixture
def test_dataset(dataset_info, load_config):
    """Create a test dataset for processing tests"""
    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type=dataset_info["type"],
        data_source=dataset_info["data_source"],
        config=load_config,
    )
    return load_module.run()


def test_create_process_module(conversation_filter_config):
    """Test creating process module"""
    operator = OperatorFactory.create_operator(conversation_filter_config)
    process_module = create_process_module(
        name="test-processor",
        operators=[operator],
    )

    assert process_module is not None
    logger.info("Process module created successfully")


def test_process_with_conversation_filter(test_dataset, conversation_filter_config):
    """Test data processing with conversation turn filter"""
    logger.info("Testing data processing with conversation filter...")

    # Create process module
    operator = OperatorFactory.create_operator(conversation_filter_config)
    process_module = create_process_module(
        name="conversation-filter-processor",
        operators=[operator],
    )

    # Process the data
    original_count = len(test_dataset)
    result = process_module.run(test_dataset)

    # Assertions
    assert isinstance(result, BaseDataSet), f"Expected BaseDataSet, got {type(result)}"
    assert (
        len(result) <= original_count
    ), "Processed dataset should not exceed original size"

    logger.success(f"Processing completed: {original_count} -> {len(result)} samples")


def test_process_empty_dataset(conversation_filter_config):
    """Test processing empty dataset"""
    # Create empty dataset
    empty_dataset = BaseDataSet(
        name="empty-test", metadata={"source": "test"}, datas=[]
    )

    # Create process module
    operator = OperatorFactory.create_operator(conversation_filter_config)
    process_module = create_process_module(
        name="empty-processor",
        operators=[operator],
    )

    # Process empty data
    result = process_module.run(empty_dataset)

    assert isinstance(result, BaseDataSet)
    assert len(result) == 0
    logger.info("Empty dataset processing test passed")


def test_process_with_multiple_operators(test_dataset):
    """Test processing with multiple operators"""
    # Create multiple operators
    operators = [
        OperatorFactory.create_operator(
            {
                "type": "filter",
                "name": "conversation_turn_filter",
                "config": {"min_turns": 1, "max_turns": 20},
            }
        ),
        # Add more operators as needed
    ]

    process_module = create_process_module(
        name="multi-operator-processor",
        operators=operators,
    )

    original_count = len(test_dataset)
    result = process_module.run(test_dataset)

    assert isinstance(result, BaseDataSet)
    assert len(result) <= original_count
    logger.success(
        f"Multi-operator processing: {original_count} -> {len(result)} samples"
    )


@pytest.mark.parametrize(
    "min_turns,max_turns",
    [
        (1, 5),
        (2, 8),
        (1, 15),
        (3, 10),
    ],
)
def test_conversation_filter_parametrized(test_dataset, min_turns, max_turns):
    """Parametrized test for different conversation turn filters"""
    filter_config = {
        "type": "filter",
        "name": "conversation_turn_filter",
        "config": {"min_turns": min_turns, "max_turns": max_turns},
    }

    operator = OperatorFactory.create_operator(filter_config)
    process_module = create_process_module(
        name=f"filter-{min_turns}-{max_turns}",
        operators=[operator],
    )

    original_count = len(test_dataset)
    result = process_module.run(test_dataset)

    assert isinstance(result, BaseDataSet)
    assert len(result) <= original_count

    logger.info(
        f"Filter ({min_turns}-{max_turns} turns): {original_count} -> {len(result)} samples"
    )


def test_process_preserves_metadata(test_dataset, conversation_filter_config):
    """Test that processing preserves dataset metadata"""
    operator = OperatorFactory.create_operator(conversation_filter_config)
    process_module = create_process_module(
        name="metadata-test-processor",
        operators=[operator],
    )

    result = process_module.run(test_dataset)

    assert isinstance(result, BaseDataSet)
    assert result.name is not None
    assert result.metadata is not None
    logger.info("Metadata preservation test passed")


def test_process_data_integrity(test_dataset, conversation_filter_config):
    """Test that processed data maintains integrity"""
    operator = OperatorFactory.create_operator(conversation_filter_config)
    process_module = create_process_module(
        name="integrity-test-processor",
        operators=[operator],
    )

    result = process_module.run(test_dataset)

    assert isinstance(result, BaseDataSet)

    # Check that all items in result are DataSample instances
    for item in result.datas:
        assert isinstance(item, DataSample), f"Expected DataSample, got {type(item)}"

    logger.info("Data integrity test passed")


def test_invalid_operator_config():
    """Test handling of invalid operator configuration"""
    invalid_config = {
        "type": "invalid_filter",
        "name": "nonexistent_filter",
        "config": {},
    }

    with pytest.raises(Exception):
        OperatorFactory.create_operator(invalid_config)


def test_process_with_invalid_operator():
    """Test process module with invalid operator configuration"""
    with pytest.raises(Exception):
        # This should fail when creating the operator
        operator = OperatorFactory.create_operator(
            {
                "type": "invalid_type",
                "name": "invalid_name",
                "config": {},
            }
        )
        create_process_module(
            name="invalid-processor",
            operators=[operator],
        )


def test_process_large_dataset_performance(dataset_info):
    """Test processing performance with larger dataset"""
    # Load larger dataset for performance testing
    large_load_config = {
        "path": "/Users/xielipeng/RM-Gallery/data/reward-bench-2/data/test-00000-of-00001.parquet",
        "limit": 500,  # Larger dataset
        "huggingface_split": "test",
    }

    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type=dataset_info["type"],
        data_source=dataset_info["data_source"],
        config=large_load_config,
    )

    large_dataset = load_module.run()

    # Create process module
    filter_config = {
        "type": "filter",
        "name": "conversation_turn_filter",
        "config": {"min_turns": 1, "max_turns": 10},
    }

    operator = OperatorFactory.create_operator(filter_config)
    process_module = create_process_module(
        name="performance-test-processor",
        operators=[operator],
    )

    # Measure processing
    import time

    start_time = time.time()
    result = process_module.run(large_dataset)
    processing_time = time.time() - start_time

    assert isinstance(result, BaseDataSet)
    assert processing_time < 30  # Should complete within 30 seconds

    logger.success(
        f"Performance test: processed {len(large_dataset)} samples in {processing_time:.2f}s"
    )


if __name__ == "__main__":
    # Run pytest when script is executed directly
    pytest.main([__file__, "-v"])
