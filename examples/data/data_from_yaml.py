"""
ORM Data Loading and Processing Script

Load and process ORM dataset using YAML configuration with integrated export functionality.

Usage:
    python orm_data_load.py
"""

from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for strategy registration
from rm_gallery.core.data.build import create_build_module_from_yaml


def load_and_process_dataset():
    """Load and process ORM dataset using YAML configuration"""
    config_path = "./rm_gallery/examples/data/data_config.yaml"

    try:
        logger.info("🚀 Starting ORM data processing...")
        logger.info(f"📄 Loading config: {config_path}")

        # Create builder from YAML config
        builder = create_build_module_from_yaml(config_path)
        logger.info(f"🔧 Created builder: {builder.name}")

        # Run the complete pipeline (Load → Process → Export)
        logger.info("⚡ Starting pipeline execution...")
        dataset = builder.run()

        logger.success("✅ Pipeline completed successfully!")
        logger.info(f"📊 Dataset: {dataset.name}")
        logger.info(f"📈 Processed: {len(dataset)} samples")

        return dataset

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main function"""
    logger.info("🎯 ORM Data Processing Pipeline")
    logger.info("=" * 50)

    dataset = load_and_process_dataset()

    if dataset:
        logger.success("\n🎉 ORM data processing completed successfully!")
        logger.info(
            "💡 Export files should be available in the configured export directory"
        )
        logger.info("📁 Check your YAML config for export settings")
    else:
        logger.error("\n❌ ORM data processing failed!")
        logger.info("💡 Check the error messages above for troubleshooting")


if __name__ == "__main__":
    main()
