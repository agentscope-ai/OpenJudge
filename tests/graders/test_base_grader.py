import pytest

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore
from openjudge.runner.resource_executor.base_resource_executor import (
    BaseResourceExecutor,
)


class MockGrader(BaseGrader):
    """Mock grader for testing BaseGrader functionality"""

    def __init__(self, name="", mode=GraderMode.POINTWISE, description="", strategy=None, **kwargs):
        super().__init__(name=name, mode=mode, description=description, strategy=strategy, **kwargs)

    async def _aevaluate(self, **kwargs):
        # Return a simple score for testing
        return GraderScore(name=self.name, score=0.8, reason="Mock evaluation completed successfully")


class ErrorGrader(BaseGrader):
    """Mock grader that raises an exception for testing error handling"""

    async def _aevaluate(self, **kwargs):
        raise ValueError("Test error occurred")


class MockExecutor(BaseResourceExecutor):
    """Mock executor for testing executor functionality"""

    async def submit(self, fn, **kwargs):
        return await fn(**kwargs)


class MockStrategy(BaseEvaluationStrategy):
    """Mock strategy that tracks whether execute was called"""

    def __init__(self):
        super().__init__()
        self.execute_called = False
        self.last_managed_fn = None
        self.last_data = None

    async def execute(self, managed_fn, **data):
        self.execute_called = True
        self.last_managed_fn = managed_fn
        self.last_data = data
        # Just call the managed function with the data
        return await managed_fn(**data)


@pytest.mark.unit
class TestBaseGrader:
    """Test cases for BaseGrader functionality"""

    def test_initialization(self):
        """Test successful initialization with required parameters"""
        grader = MockGrader(name="test_grader", mode=GraderMode.POINTWISE, description="A test grader")

        assert grader.name == "test_grader"
        assert grader.mode == GraderMode.POINTWISE
        assert grader.description == "A test grader"
        assert grader.strategy is None

    @pytest.mark.asyncio
    async def test_aevaluate_error_handling(self):
        """Test that errors in _aevaluate are properly propagated"""
        grader = ErrorGrader(name="error_test")

        with pytest.raises(ValueError, match="Test error occurred"):
            await grader.aevaluate(query="test query", response="test response")

    @pytest.mark.asyncio
    async def test_aevaluate_copy_isolation(self):
        """Test that deepcopy isolates runtime instance from original"""
        grader = MockGrader(name="isolation_test")

        # Add an attribute to the original grader
        original_attr_value = "original_value"
        grader.original_attr = original_attr_value

        # Before calling aevaluate, capture the original state
        original_name = grader.name

        # Call aevaluate which should create a deepcopy
        result = await grader.aevaluate(query="test query", response="test response")

        # Verify the result is correct
        assert isinstance(result, GraderScore)
        assert result.score == 0.8

        # The original grader should be unchanged
        assert hasattr(grader, "original_attr")
        assert grader.name == original_name

        # This demonstrates that the runtime copy is isolated from the original

    def test_from_config_creation(self):
        """Test creating a grader from configuration"""
        config = {
            "name": "configured_grader",
            "mode": "listwise",
            "description": "A configured grader",
            "mapper": {"query": "question"},
            "custom_param": "value",
        }

        # Since we can't instantiate BaseGrader directly, we test with MockGrader
        grader = MockGrader.from_config(config)

        assert grader.name == "configured_grader"
        assert grader.mode == GraderMode.LISTWISE  # string converted to enum
        assert grader.description == "A configured grader"
        # kwargs should contain custom_param
        assert grader.kwargs.get("custom_param") == "value"

    def test_to_dict_serialization(self):
        """Test serializing grader to dictionary"""
        grader = MockGrader(name="serialized_grader", mode=GraderMode.POINTWISE, description="A serialized grader")
        grader.kwargs["custom"] = "value"

        data = grader.to_dict()

        assert data["name"] == "serialized_grader"
        assert data["mode"] == "pointwise"  # enum converted to string
        assert data["description"] == "A serialized grader"
        assert data["kwargs"]["custom"] == "value"

    def test_get_metadata_default(self):
        """Test default get_metadata implementation"""
        metadata = BaseGrader.get_metadata()
        assert "warning" in metadata
        assert "not implemented" in metadata["warning"]
