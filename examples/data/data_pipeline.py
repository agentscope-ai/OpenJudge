"""
Simple Data Pipeline Validation Tool

Test data modules: Load, Process, Annotation, Export

Usage:
    python data_pipeline.py [--mode MODE] [--limit LIMIT]

Examples:
    # basic pipeline
    python data_pipeline.py --mode basic

    # annotation pipeline
    python data_pipeline.py --mode annotation --api-token TOKEN

    # load only
    python data_pipeline.py --mode load-only --limit 100

    # process only
    python data_pipeline.py --mode process-only

    # export only
    python data_pipeline.py --mode export-only

    # export annotation
    python data_pipeline.py --mode export-annotation --api-token TOKEN --project-id PROJECT_ID

"""

import argparse

import requests
from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for example strategy registration
from rm_gallery.core.data.annotation.annotation import create_annotator
from rm_gallery.core.data.build import create_builder
from rm_gallery.core.data.export import create_exporter
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.process.ops.base import OperatorFactory
from rm_gallery.core.data.process.process import create_processor
from rm_gallery.core.data.schema import BaseDataSet


class DataPipelineValidator:
    """Simple data pipeline validation"""

    def __init__(self):
        self.name = "allenai/reward-bench-2"
        self.type = "local"
        self.data_source = "rewardbench2"
        self.load_config = {
            "path": "./data/reward-bench-2/data/test-00000-of-00001.parquet",
            "limit": 100,
            "huggingface_split": "test",
        }
        self.process_config = {
            "type": "filter",
            "name": "conversation_turn_filter",
            "config": {"min_turns": 1, "max_turns": 10},
        }
        self.export_config = {
            "output_dir": "./examples/data/exports",
            "formats": ["jsonl"],
            "split_ratio": {"train": 0.8, "test": 0.2},
            "preserve_structure": True,
        }

    def create_load_module(self):
        """Create load module"""
        return create_loader(
            name=self.name,
            load_strategy_type=self.type,
            data_source=self.data_source,
            config=self.load_config,
        )

    def create_process_module(self):
        """Create simple process module"""
        return create_processor(
            name="test-processor",
            operators=[OperatorFactory.create_operator(self.process_config)],
        )

    def create_export_module(self):
        """Create export module"""
        return create_exporter(name="test-exporter", config=self.export_config)

    def create_annotation_module(self, api_token: str):
        """Create annotation module"""
        return create_annotator(
            name="test-annotation",
            api_token=api_token,
            server_url="http://localhost:8080",
            project_title="Test Annotation",
            template_name=self.data_source,
        )

    def test_load_only(self) -> bool:
        """Test loading only"""
        logger.info("Testing load...")
        try:
            load_module = self.create_load_module()
            result = load_module.run()

            if isinstance(result, BaseDataSet):
                logger.success(f"Loaded {len(result)} samples")
                return True
            else:
                logger.error(f"Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def test_process_only(self) -> bool:
        """Test processing only"""
        logger.info("Testing process...")
        try:
            load_module = self.create_load_module()
            process_module = self.create_process_module()

            # Load data first
            data = load_module.run()

            # Process data
            result = process_module.run(data)

            if isinstance(result, BaseDataSet):
                logger.success(f"Processed {len(data)} -> {len(result)} samples")
                return True
            else:
                logger.error(f"Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"Process failed: {e}")
            return False

    def test_export_only(self) -> bool:
        """Test export only"""
        logger.info("Testing export...")
        try:
            load_module = self.create_load_module()
            export_module = self.create_export_module()

            # Load data first
            data = load_module.run()

            # Export data
            result = export_module.run(data)

            if isinstance(result, BaseDataSet):
                logger.success(f"Exported {len(result)} samples")
                return True
            else:
                logger.error(f"Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def run_basic_pipeline(self) -> bool:
        """Run basic pipeline: Load → Process → Export"""
        logger.info("Running basic pipeline...")
        try:
            build_module = create_builder(
                name="basic_pipeline",
                load_module=self.create_load_module(),
                process_module=self.create_process_module(),
                export_module=self.create_export_module(),
            )

            result = build_module.run()

            if isinstance(result, BaseDataSet):
                logger.success(f"Pipeline completed: {len(result)} samples")
                return True
            else:
                logger.error(f"Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

    def run_annotation_pipeline(self, api_token: str) -> bool:
        """Run annotation pipeline: Load → Process → Annotation → Export"""
        logger.info("Running annotation pipeline...")

        # Check service
        if not self.check_service():
            logger.error("Label Studio service unavailable")
            return False

        try:
            build_module = create_builder(
                name="annotation_pipeline",
                load_module=self.create_load_module(),
                process_module=self.create_process_module(),
                annotation_module=self.create_annotation_module(api_token),
                export_module=self.create_export_module(),
            )

            result = build_module.run()

            if isinstance(result, BaseDataSet):
                logger.success(f"Annotation pipeline completed: {len(result)} samples")
                return True
            else:
                logger.error(f"Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"Annotation pipeline failed: {e}")
            return False

    def check_service(self) -> bool:
        """Check Label Studio service"""
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def export_annotation_data(self, api_token: str, project_id: int = None) -> bool:
        """Export completed annotation data from Label Studio"""
        logger.info("Exporting annotations...")

        # Check service
        if not self.check_service():
            logger.error("Label Studio service unavailable")
            return False

        try:
            # Create annotation module
            annotation_module = self.create_annotation_module(api_token)

            # Set project ID if provided, otherwise find the most recent project
            if project_id:
                annotation_module.project_id = project_id
            else:
                if annotation_module.client:
                    projects = annotation_module.client.get_projects()
                    if projects:
                        annotation_module.project_id = projects[-1]["id"]
                    else:
                        logger.error("No projects found")
                        return False
                else:
                    logger.error("Cannot initialize client")
                    return False

            # Export annotations to dataset format
            annotated_dataset = annotation_module.export_annotations_to_dataset()

            # Verify the exported dataset
            if isinstance(annotated_dataset, BaseDataSet):
                if len(annotated_dataset) > 0:
                    logger.success(f"Exported {len(annotated_dataset)} annotations")

                    # Also export in various formats using export module
                    export_module = self.create_export_module()
                    export_module.run(annotated_dataset)
                    return True
                else:
                    logger.warning("No annotated data found")
                    return False
            else:
                logger.error(f"Expected BaseDataSet, got {type(annotated_dataset)}")
                return False

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Data Pipeline Validator")

    parser.add_argument(
        "--mode",
        choices=[
            "basic",
            "annotation",
            "load-only",
            "process-only",
            "export-only",
            "export-annotation",
        ],
        default="basic",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        help="Label Studio API token",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        help="Label Studio project ID",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    validator = DataPipelineValidator()

    logger.info(f"Pipeline mode: {args.mode}")

    success = False

    if args.mode == "load-only":
        success = validator.test_load_only()
    elif args.mode == "process-only":
        success = validator.test_process_only()
    elif args.mode == "export-only":
        success = validator.test_export_only()
    elif args.mode == "basic":
        success = validator.run_basic_pipeline()
    elif args.mode == "annotation":
        if not args.api_token:
            logger.error("--api-token required")
            return
        success = validator.run_annotation_pipeline(args.api_token)
    elif args.mode == "export-annotation":
        if not args.api_token:
            logger.error("--api-token required")
            return
        success = validator.export_annotation_data(args.api_token, args.project_id)

    if success:
        logger.success("Completed successfully")
    else:
        logger.error("Failed")


if __name__ == "__main__":
    main()
