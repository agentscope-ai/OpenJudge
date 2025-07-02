import pytest

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.reward.template import BasePromptTemplate
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward


@pytest.fixture
def reward_instance():
    """Create a basic instance for testing"""

    # Create a concrete subclass with proper initialization
    class TestTemplate(BasePromptTemplate):
        """Test template implementation for validation testing"""

        pass

    class TestReward(BaseHelpfulnessListWiseReward):
        def __init__(self):
            # Initialize all required fields with proper types
            super().__init__(
                name="Test Reward",
                desc="Test Description",
                scenario="Test Scenario",
                principles=["Test Principle"],
                template=TestTemplate,  # Use valid template instance
            )

    return TestReward()


class TestBaseHelpfulnessListWiseReward:
    """Test suite with full inheritance chain coverage using proper DataSample structure"""

    def test_required_attributes_exist(self, reward_instance):
        """Test presence of required attributes from parent classes"""
        assert hasattr(reward_instance, "principles")
        assert hasattr(reward_instance, "desc")
        assert hasattr(reward_instance, "scenario")
        assert hasattr(reward_instance, "template")

    def test_initialization_with_defaults(self, reward_instance):
        """Test initialization with default values from parent classes"""
        assert reward_instance.desc == "Test Description"
        assert reward_instance.principles == ["Test Principle"]

    def test_abstract_methods_implemented(self, reward_instance):
        """Test implementation of abstract methods from parent classes"""
        assert hasattr(reward_instance, "_before_evaluate")
        assert hasattr(reward_instance, "_after_evaluate")

    def test_method_signatures(self, reward_instance):
        """Test method signatures match parent class requirements"""
        # Test _before_evaluate signature with proper DataSample structure
        sample_input = DataSample(
            unique_id="test",
            input=[ChatMessage(role="user", content="What is 2+2?")],
            output=[
                DataOutput(
                    answer=Step(
                        role="assistant", content="4", label={"preference": "chosen"}
                    )
                ),
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content="four",
                        label={"preference": "rejected"},
                    )
                ),
            ],
        )

        result = reward_instance._before_evaluate(sample=sample_input)
        assert isinstance(result, dict)
        assert "desc" in result
        assert "principles" in result
        assert "query" in result
