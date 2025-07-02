"""
Data Loading and Processing Script

Load and process dataset using YAML configuration with integrated export functionality.

Usage:
    python data_from_yaml.py [--config CONFIG_PATH]

Examples:
    python data_from_yaml.py
    python data_from_yaml.py --config examples/train/pointwise/data_config.yaml
    python data_from_yaml.py --config /absolute/path/to/config.yaml
"""

import argparse

from loguru import logger

import rm_gallery.core.data  # noqa: F401 - needed for core strategy registration
import rm_gallery.gallery.data  # noqa: F401 - needed for gallery strategy registration
from rm_gallery.core.data.build import create_builder_from_yaml


def load_and_process_dataset(config_path):
    """Load and process dataset using YAML configuration

    Args:
        config_path (str): Path to the YAML configuration file
    """
    try:
        logger.info("🚀 Starting data processing...")
        logger.info(f"📄 Loading config: {config_path}")

        # Create builder from YAML config
        builder = create_builder_from_yaml(config_path)
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Load and process dataset using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_from_yaml.py
  python data_from_yaml.py --config examples/train/pointwise/data_config.yaml
  python data_from_yaml.py --config /absolute/path/to/config.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="examples/train/pairwise/data_config.yaml",
        help="Path to YAML configuration file (default: examples/train/pairwise/data_config.yaml)",
    )

    return parser.parse_args()


def main():
    """Main function"""
    logger.info("🎯 Data Processing Pipeline")

    # Parse command line arguments
    args = parse_args()

    logger.info(f"🔧 Using config file: {args.config}")

    dataset = load_and_process_dataset(args.config)

    if dataset:
        logger.success("🎉 data processing completed successfully!")
        logger.info(
            "💡 Export files should be available in the configured export directory"
        )
        logger.info("📁 Check your YAML config for export settings")
    else:
        logger.error("\n❌ data processing failed!")
        logger.info("💡 Check the error messages above for troubleshooting")


if __name__ == "__main__":
    main()
