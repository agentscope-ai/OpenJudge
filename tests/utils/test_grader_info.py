#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for grader info utility.
"""

import pytest

from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.utils.grader_info import get_all_grader_info
from openjudge.utils.instance import InstanceConfig, init_instance_by_config


@pytest.mark.unit
class TestGraderInfoUtil:
    """Test cases for grader_info util."""

    @classmethod
    def setup_class(cls):
        """Run once before running all test cases."""
        cls.all_grader_info = get_all_grader_info()
        print(f"got the information of {len(cls.all_grader_info)} graders")

    def test_get_graders_info(self):
        """Test get_all_graders_info unti function."""
        for name, gi in self.all_grader_info.items():
            error_msg = f"name:{name}, info_obj:{gi}"
            assert name == gi.class_name, error_msg
            assert gi.module_path.startswith("openjudge.graders."), error_msg
            assert isinstance(gi.parent_class_names, list), error_msg
            assert len(gi.parent_class_names) > 0, error_msg
            assert len(gi.init_method.signature) > 0, error_msg
            assert len(gi.aevaluate_method.signature) > 0, error_msg

    def test_creating_grader_instance_using_grader_info(self):
        """Test creating grader instance using GraderInfo."""
        grader_info = self.all_grader_info[CorrectnessGrader.__name__]
        error_msg = f"grader_info:{grader_info}"
        instance_config: InstanceConfig = {
            "class_name": grader_info.class_name,
            "module_path": grader_info.module_path,
            "kwargs": {"model": {"model": "foo", "api_key": "bar"}, "threshold": 3, "language": "zh"},
        }

        grader = init_instance_by_config(instance_config)
        assert isinstance(grader, CorrectnessGrader), error_msg
        assert grader.model.model == "foo"
        assert grader.language == "zh", error_msg
        assert grader.threshold == 3, error_msg


if __name__ == "__main__":
    pytest.main([__file__])
