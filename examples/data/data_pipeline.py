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

import rm_gallery.core.data  # noqa: F401 - needed for strategy registration
from rm_gallery.core.data.annotation.annotation import create_annotation_module
from rm_gallery.core.data.build import create_build_module
from rm_gallery.core.data.export import create_export_module
from rm_gallery.core.data.load.base import create_load_module
from rm_gallery.core.data.process.ops.base import OperatorFactory
from rm_gallery.core.data.process.process import create_process_module
from rm_gallery.core.data.schema import BaseDataSet


class DataPipelineValidator:
    """Simple data pipeline validation"""

    def __init__(self, limit: int = 100):
        self.limit = limit
        self.data_path = (
            "./data/preference-test-sets/data/anthropic_helpful-00000-of-00001.parquet"
        )
        self.export_dir = "./rm_gallery/examples/data/exports"
        self.data_source = "rewardbench"
        self.format = "parquet"

    def create_load_module(self, limit: int = None):
        """Create load module"""
        return create_load_module(
            name="test-data-loader",
            load_strategy_type="local",
            data_source=self.data_source,
            config={
                "path": self.data_path,
                "limit": limit or self.limit,
            },
        )

    def create_process_module(self):
        """Create simple process module"""
        return create_process_module(
            name="test-processor",
            operators=[
                OperatorFactory.create_operator(
                    {
                        "type": "filter",
                        "name": "conversation_turn_filter",
                        "config": {"min_turns": 1, "max_turns": 10},
                    }
                ),
                OperatorFactory.create_operator(
                    {
                        "type": "filter",
                        "name": "text_length_filter",
                        "config": {"min_length": 10, "max_length": 5000},
                    }
                ),
            ],
        )

    def create_export_module(self):
        """Create export module"""
        return create_export_module(
            name="test-exporter",
            config={
                "output_dir": self.export_dir,
                "formats": [self.format],
                "split_ratio": {"train": 0.8, "test": 0.2},
            },
        )

    def create_annotation_module(self, api_token: str):
        """Create annotation module"""
        return create_annotation_module(
            name="test-annotation",
            api_token=api_token,
            server_url="http://localhost:8080",
            project_title="Test Annotation",
            template_name="reward_bench",
        )

    def test_load_only(self, limit: int = None) -> bool:
        """Test loading only"""
        logger.info("🔍 Testing load module...")
        try:
            load_module = self.create_load_module(limit)
            result = load_module.run()

            # Verify result is BaseDataSet
            if isinstance(result, BaseDataSet):
                logger.success(
                    f"✅ Loaded {len(result)} samples into BaseDataSet: {result.name}"
                )
                logger.info(f"   Metadata: {result.metadata}")
                if len(result) > 0:
                    sample = result[0]
                    logger.info(
                        f"   Sample structure: ID={sample.unique_id}, Input={len(sample.input)} messages, Output={len(sample.output)} items"
                    )
                return True
            else:
                logger.error(f"❌ Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
            return False

    def test_process_only(self) -> bool:
        """Test processing only"""
        logger.info("🔍 Testing process module...")
        try:
            load_module = self.create_load_module(10)
            process_module = self.create_process_module()

            # Load data first
            data = load_module.run()
            logger.info(f"📥 Input: {len(data)} samples from dataset '{data.name}'")

            # Process data
            result = process_module.run(data)

            # Verify result is BaseDataSet
            if isinstance(result, BaseDataSet):
                logger.success(
                    f"✅ Processed to {len(result)} samples in dataset '{result.name}'"
                )
                logger.info(f"   Metadata: {result.metadata}")
                return True
            else:
                logger.error(f"❌ Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Process failed: {e}")
            return False

    def test_export_only(self) -> bool:
        """Test export only"""
        logger.info("🔍 Testing export module...")
        try:
            load_module = self.create_load_module(5)
            export_module = self.create_export_module()

            # Load data first
            data = load_module.run()
            logger.info(f"📥 Input: {len(data)} samples from dataset '{data.name}'")

            # Export data
            result = export_module.run(data)

            # Verify export result
            if isinstance(result, BaseDataSet):
                logger.success(f"✅ Exported {len(result)} samples to {self.export_dir}")
                logger.info(f"   Dataset: {result.name}")
                return True
            else:
                logger.error(f"❌ Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False

    def run_basic_pipeline(self) -> bool:
        """Run basic pipeline: Load → Process → Export"""
        logger.info("🚀 Running basic pipeline: Load → Process → Export")
        try:
            build_module = create_build_module(
                name="basic_pipeline",
                load_module=self.create_load_module(),
                process_module=self.create_process_module(),
                export_module=self.create_export_module(),
            )

            result = build_module.run()

            # Verify final result
            if isinstance(result, BaseDataSet):
                logger.success(
                    f"✅ Basic pipeline completed: {len(result)} samples in dataset '{result.name}'"
                )
                logger.info(f"   Final metadata: {result.metadata}")
                return True
            else:
                logger.error(f"❌ Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Basic pipeline failed: {e}")
            return False

    def run_annotation_pipeline(self, api_token: str) -> bool:
        """Run annotation pipeline: Load → Process → Annotation → Export"""
        logger.info(
            "🚀 Running annotation pipeline: Load → Process → Annotation → Export"
        )

        # Check service
        if not self.check_service():
            logger.error("❌ Label Studio service not available")
            return False

        try:
            build_module = create_build_module(
                name="annotation_pipeline",
                load_module=self.create_load_module(),
                process_module=self.create_process_module(),
                annotation_module=self.create_annotation_module(api_token),
                export_module=self.create_export_module(),
            )

            result = build_module.run()

            # Verify final result
            if isinstance(result, BaseDataSet):
                logger.success(
                    f"✅ Annotation pipeline completed: {len(result)} samples in dataset '{result.name}'"
                )
                logger.info(f"   Final metadata: {result.metadata}")
                return True
            else:
                logger.error(f"❌ Expected BaseDataSet, got {type(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Annotation pipeline failed: {e}")
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
        logger.info("🚀 Exporting annotation data from Label Studio...")

        # Check service
        if not self.check_service():
            logger.error("❌ Label Studio service not available")
            return False

        try:
            # Create annotation module
            logger.info("📝 Creating annotation module...")
            annotation_module = self.create_annotation_module(api_token)
            logger.info("✅ Annotation module created successfully")

            # Set project ID if provided, otherwise try to find the most recent project
            if project_id:
                annotation_module.project_id = project_id
                logger.info(f"📊 Using specified project ID: {project_id}")
            else:
                # Try to get project ID from client
                logger.info("🔍 Searching for existing projects...")
                if annotation_module.client:
                    projects = annotation_module.client.get_projects()
                    if projects:
                        # Use the most recent project
                        annotation_module.project_id = projects[-1]["id"]
                        logger.info(
                            f"📊 Using most recent project ID: {annotation_module.project_id}"
                        )
                    else:
                        logger.error("❌ No projects found in Label Studio")
                        return False
                else:
                    logger.error("❌ Cannot initialize Label Studio client")
                    return False

            # Export annotations to dataset format
            logger.info("📤 Exporting annotations from Label Studio...")
            annotated_dataset = annotation_module.export_annotations_to_dataset()

            # Verify the exported dataset
            if isinstance(annotated_dataset, BaseDataSet):
                logger.info(
                    f"✅ Successfully retrieved {len(annotated_dataset)} samples into dataset '{annotated_dataset.name}'"
                )
                logger.info(f"   Metadata: {annotated_dataset.metadata}")

                if len(annotated_dataset) > 0:
                    logger.success(
                        f"✅ Exported {len(annotated_dataset)} annotated samples"
                    )
                    logger.info(f"📁 Saved to: {self.export_dir}")

                    # Also export in various formats using export module
                    logger.info("📦 Exporting to additional formats...")
                    export_module = self.create_export_module()
                    export_result = export_module.run(annotated_dataset)

                    if isinstance(export_result, BaseDataSet):
                        logger.info(
                            f"📦 Additional export formats completed for dataset '{export_result.name}'"
                        )
                        logger.info(f"📦 Files saved to: {self.export_dir}")

                    return True
                else:
                    logger.warning("⚠️  No annotated data found to export")
                    logger.info(
                        "💡 Make sure you have completed some annotations in Label Studio"
                    )
                    return False
            else:
                logger.error(
                    f"❌ Expected BaseDataSet from annotation export, got {type(annotated_dataset)}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Export annotation data failed: {e}")
            import traceback

            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple Data Pipeline Validator")

    parser.add_argument(
        "--mode",
        choices=[
            "basic",
            "annotation",
            "load-only",
            "process-only",
            "export-only",
            "export-annotation",
            "yaml-config",
        ],
        default="basic",
        help="Pipeline mode (default: basic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to process (default: 100)",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        help="Label Studio API token (required for annotation and export-annotation modes)",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        help="Label Studio project ID (optional for export-annotation mode)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to YAML configuration file (required for yaml-config mode)",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    validator = DataPipelineValidator(limit=args.limit)

    logger.info("🚀 Data Pipeline Validator")
    logger.info("=" * 40)
    logger.info(f"🔧 Mode: {args.mode}")
    logger.info(f"📊 Limit: {args.limit}")
    logger.info(f"📁 Export dir: {validator.export_dir}")
    logger.info("=" * 40)

    success = False

    if args.mode == "load-only":
        success = validator.test_load_only(args.limit)

    elif args.mode == "process-only":
        success = validator.test_process_only()

    elif args.mode == "export-only":
        success = validator.test_export_only()

    elif args.mode == "basic":
        success = validator.run_basic_pipeline()

    elif args.mode == "annotation":
        if not args.api_token:
            logger.error("❌ --api-token required for annotation mode")
            logger.info("💡 Get your token from: http://localhost:8080")
            return
        success = validator.run_annotation_pipeline(args.api_token)

    elif args.mode == "export-annotation":
        if not args.api_token:
            logger.error("❌ --api-token required for export-annotation mode")
            logger.info("💡 Get your token from: http://localhost:8080")
            return
        success = validator.export_annotation_data(args.api_token, args.project_id)

    elif args.mode == "yaml-config":
        if not args.config_path:
            logger.error("❌ --config-path required for yaml-config mode")
            logger.info("💡 Specify path to YAML configuration file")
            return
        success = validator.run_yaml_config_pipeline(args.config_path)

    if success:
        logger.success(f"\n🎉 {args.mode} completed successfully!")
        logger.info(f"📁 Check results in: {validator.export_dir}")
    else:
        logger.error(f"\n❌ {args.mode} failed!")


if __name__ == "__main__":
    main()
