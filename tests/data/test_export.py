"""
Export Module Test Suite

Simple test cases for data export functionality, covering core features:
- Basic export functionality
- Multiple format support (JSON, JSONL, Parquet)
- Train/test data splitting
- Pipeline integration testing

Based on the usage patterns demonstrated in docs/tutorial/data/load.ipynb
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import pytest

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for example strategy registration
from rm_gallery.core.data.build import create_build_module
from rm_gallery.core.data.export import DataExport, create_export_module
from rm_gallery.core.data.load.base import create_load_module
from rm_gallery.core.data.schema import (
    BaseDataSet,
    ChatMessage,
    DataOutput,
    DataSample,
    Step,
)


# Module-level fixtures
@pytest.fixture
def sample_data_samples() -> List[DataSample]:
    """Create sample data for testing"""
    samples = []
    for i in range(10):
        sample = DataSample(
            unique_id=f"test_sample_{i}",
            input=[ChatMessage(role="user", content=f"Test input {i}")],
            output=[DataOutput(answer=Step(content=f"Test output {i}"))],
            source="test_source",
            task_category="test",
        )
        samples.append(sample)
    return samples


@pytest.fixture
def sample_dataset(sample_data_samples) -> BaseDataSet:
    """Create sample BaseDataSet for testing"""
    return BaseDataSet(
        name="test_dataset",
        metadata={"test_meta": "test_value"},
        datas=sample_data_samples,
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestDataExport:
    """Test suite for DataExport module"""

    def test_create_export_module(self):
        """Test basic export module creation"""
        config = {"output_dir": "./test_exports", "formats": ["jsonl"]}

        export_module = create_export_module(name="test_exporter", config=config)

        assert isinstance(export_module, DataExport)
        assert export_module.name == "test_exporter"
        assert export_module.config == config

    def test_export_jsonl_format(self, sample_dataset, temp_output_dir):
        """Test JSONL format export"""
        config = {"output_dir": str(temp_output_dir), "formats": ["jsonl"]}

        export_module = create_export_module("test_exporter", config=config)
        result = export_module.run(sample_dataset)

        # Verify result is the original dataset (passthrough)
        assert result == sample_dataset

        # Check output file exists
        output_file = temp_output_dir / "test_exporter.jsonl"
        assert output_file.exists()

        # Verify file content
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == len(sample_dataset.datas)

            # Check first line is valid JSON
            first_line = json.loads(lines[0])
            assert "unique_id" in first_line
            assert "input" in first_line
            assert "output" in first_line

    def test_export_json_format(self, sample_dataset, temp_output_dir):
        """Test JSON format export"""
        config = {"output_dir": str(temp_output_dir), "formats": ["json"]}

        export_module = create_export_module("test_exporter", config=config)
        result = export_module.run(sample_dataset)

        # Check output file exists
        output_file = temp_output_dir / "test_exporter.json"
        assert output_file.exists()

        # Verify file content is valid JSON
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "name" in data
            assert "datas" in data
            assert len(data["datas"]) == len(sample_dataset.datas)

    def test_export_parquet_format(self, sample_dataset, temp_output_dir):
        """Test Parquet format export"""
        config = {"output_dir": str(temp_output_dir), "formats": ["parquet"]}

        export_module = create_export_module("test_exporter", config=config)
        result = export_module.run(sample_dataset)

        # Check output file exists
        output_file = temp_output_dir / "test_exporter.parquet"
        assert output_file.exists()

        # Verify file content
        df = pd.read_parquet(output_file)
        assert len(df) == len(sample_dataset.datas)
        assert "unique_id" in df.columns
        assert "input" in df.columns
        assert "output" in df.columns

    def test_export_multiple_formats(self, sample_dataset, temp_output_dir):
        """Test exporting to multiple formats simultaneously"""
        config = {
            "output_dir": str(temp_output_dir),
            "formats": ["json", "jsonl", "parquet"],
        }

        export_module = create_export_module("test_exporter", config=config)
        result = export_module.run(sample_dataset)

        # Check all format files exist
        for format_type in ["json", "jsonl", "parquet"]:
            output_file = temp_output_dir / f"test_exporter.{format_type}"
            assert output_file.exists()

    def test_export_with_train_test_split(self, sample_dataset, temp_output_dir):
        """Test export with train/test split functionality"""
        config = {
            "output_dir": str(temp_output_dir),
            "formats": ["jsonl"],
            "split_ratio": {"train": 0.8, "test": 0.2},
        }

        export_module = create_export_module("test_exporter", config=config)
        result = export_module.run(sample_dataset)

        # Check split files exist
        train_file = temp_output_dir / "test_exporter_train.jsonl"
        test_file = temp_output_dir / "test_exporter_test.jsonl"

        assert train_file.exists()
        assert test_file.exists()

        # Verify split ratios (approximately)
        with open(train_file, "r") as f:
            train_lines = len(f.readlines())
        with open(test_file, "r") as f:
            test_lines = len(f.readlines())

        total_samples = len(sample_dataset.datas)
        expected_train = int(total_samples * 0.8)
        expected_test = total_samples - expected_train

        assert train_lines == expected_train
        assert test_lines == expected_test


class TestExportPipelineIntegration:
    """Test suite for export module pipeline integration"""

    @pytest.fixture
    def mock_data_file(self, temp_output_dir):
        """Create a mock data file for testing"""
        mock_data = [
            {
                "messages": [
                    {"role": "user", "content": f"Test question {i}"},
                    {"role": "assistant", "content": f"Test answer {i}"},
                ]
            }
            for i in range(5)
        ]

        # Create mock file in temp directory
        mock_file = temp_output_dir / "mock_data.json"
        with open(mock_file, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, indent=2)

        return str(mock_file)

    def test_load_and_export_pipeline(self, mock_data_file, temp_output_dir):
        """Test complete load → export pipeline as shown in load.ipynb"""
        # Create load module
        load_config = {"path": mock_data_file, "limit": 5}

        load_module = create_load_module(
            name="test_pipeline",
            load_strategy_type="local",
            data_source="chat_message",
            config=load_config,
        )

        # Create export module
        export_config = {
            "output_dir": str(temp_output_dir),
            "formats": ["jsonl"],
            "split_ratio": {"train": 0.8, "test": 0.2},
        }

        export_module = create_export_module(name="test_pipeline", config=export_config)

        # Create complete pipeline
        pipeline = create_build_module(
            name="load_export_pipeline",
            load_module=load_module,
            export_module=export_module,
        )

        # Run pipeline
        result = pipeline.run()

        # Verify pipeline result
        assert isinstance(result, BaseDataSet)
        assert len(result.datas) == 5

        # Verify export files were created
        train_file = temp_output_dir / "test_pipeline_train.jsonl"
        test_file = temp_output_dir / "test_pipeline_test.jsonl"

        assert train_file.exists()
        assert test_file.exists()

        # Verify file contents
        with open(train_file, "r") as f:
            train_lines = len(f.readlines())
        with open(test_file, "r") as f:
            test_lines = len(f.readlines())

        assert train_lines + test_lines == 5
        assert train_lines == 4  # 80% of 5 = 4
        assert test_lines == 1  # 20% of 5 = 1


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
