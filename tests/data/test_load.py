import pytest
from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for example strategy registration
from rm_gallery.core.data.load.base import create_load_module
from rm_gallery.core.data.schema import BaseDataSet


@pytest.fixture
def load_config():
    """Load configuration fixture"""
    return {
        "path": "./data/reward-bench-2/data/test-00000-of-00001.parquet",
        "limit": 100,
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


def test_create_load_module(dataset_info, load_config):
    """Test creating load module"""
    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type=dataset_info["type"],
        data_source=dataset_info["data_source"],
        config=load_config,
    )

    assert load_module is not None
    logger.info("Load module created successfully")


def test_data_load(dataset_info, load_config):
    """Test data loading functionality"""
    logger.info("Testing data load...")

    # Create load module
    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type=dataset_info["type"],
        data_source=dataset_info["data_source"],
        config=load_config,
    )

    # Run the load operation
    result = load_module.run()

    # Assertions
    assert isinstance(result, BaseDataSet), f"Expected BaseDataSet, got {type(result)}"
    assert len(result) > 0, "Dataset should contain at least one sample"
    assert (
        len(result) <= load_config["limit"]
    ), f"Dataset size ({len(result)}) exceeds limit ({load_config['limit']})"

    logger.success(f"Successfully loaded {len(result)} samples")


def test_data_load_with_different_limits(dataset_info, load_config):
    """Test data loading with different limits"""
    test_limits = [10, 50, 100]

    for limit in test_limits:
        logger.info(f"Testing with limit: {limit}")

        # Update config with new limit
        config = load_config.copy()
        config["limit"] = limit

        # Create and run load module
        load_module = create_load_module(
            name=dataset_info["name"],
            load_strategy_type=dataset_info["type"],
            data_source=dataset_info["data_source"],
            config=config,
        )

        result = load_module.run()

        # Assertions
        assert isinstance(result, BaseDataSet)
        assert (
            len(result) <= limit
        ), f"Dataset size ({len(result)}) exceeds limit ({limit})"

        logger.success(f"Limit {limit}: loaded {len(result)} samples")


def test_data_load_empty_config(dataset_info):
    """Test data loading with minimal config should fail without required path"""
    minimal_config = {"limit": 10}

    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type=dataset_info["type"],
        data_source=dataset_info["data_source"],
        config=minimal_config,
    )

    # Should raise an exception because 'path' is required for local file strategy
    with pytest.raises(
        RuntimeError, match="File data strategy requires 'path' in config"
    ):
        load_module.run()


@pytest.mark.parametrize("limit", [1, 5, 20, 50])
def test_data_load_parametrized(dataset_info, load_config, limit):
    """Parametrized test for different data limits"""
    config = load_config.copy()
    config["limit"] = limit

    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type=dataset_info["type"],
        data_source=dataset_info["data_source"],
        config=config,
    )

    result = load_module.run()

    assert isinstance(result, BaseDataSet)
    assert len(result) <= limit
    assert len(result) > 0

    logger.info(f"Parametrized test (limit={limit}): loaded {len(result)} samples")


def test_invalid_data_source():
    """Test handling of invalid data source"""
    # Create module with invalid data source
    load_module = create_load_module(
        name="invalid/dataset",
        load_strategy_type="local",
        data_source="invalid_source",
        config={"path": "/nonexistent/path.parquet", "limit": 10},
    )

    # Should raise an exception when trying to run
    with pytest.raises(RuntimeError):
        load_module.run()


def test_invalid_load_strategy_type(dataset_info, load_config):
    """Test handling of invalid load strategy type"""
    # Create module with invalid strategy type
    load_module = create_load_module(
        name=dataset_info["name"],
        load_strategy_type="invalid_type",
        data_source=dataset_info["data_source"],
        config=load_config,
    )

    # Should raise an exception when trying to run
    with pytest.raises(RuntimeError):
        load_module.run()


if __name__ == "__main__":
    # Run pytest when script is executed directly
    pytest.main([__file__, "-v"])
