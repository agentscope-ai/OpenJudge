import json
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

import rm_gallery.core.data.ops  # noqa: F401 - needed for strategy registration
from rm_gallery.core.data.build import create_build_module_from_yaml


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def convert_numpy_in_dict(obj):
    """Recursively convert numpy arrays and other non-serializable types in nested dictionaries and lists"""
    if isinstance(obj, dict):
        return {k: convert_numpy_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_in_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


def convert_to_jsonl_format(data):
    """Convert BaseData object to JSONL format"""
    try:
        # First, try to get the raw dict and clean it
        try:
            raw_data = data.dict()
        except:
            raw_data = data.model_dump()

        # Convert any numpy arrays and datetime objects
        cleaned_data = convert_numpy_in_dict(raw_data)
        return cleaned_data

    except Exception as e:
        logger.error(f"Error in converting data to JSON format: {str(e)}")
        logger.error(f"Data type: {type(data)}")
        raise e


def load_and_process_dataset():
    """load and process orm dataset"""
    config_path = "./rm_gallery/examples/data/orm_data_load.yaml"

    try:
        logger.info(f"Loading dataset from config: {config_path}")

        builder = create_build_module_from_yaml(config_path)
        logger.info(f"Created builder: {builder.name}")

        logger.info("Starting pipeline execution...")
        dataset = builder.run()

        output_path = f"{dataset.name.replace('-', '_')}_output.jsonl"
        export_dataset(dataset, output_path)
        return dataset

    except Exception as e:
        logger.error(f"❌ Error in dataset processing: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def export_dataset(dataset, output_path):
    if not dataset or not dataset.datas:
        logger.warning("No data to export")
        return

    output_path = Path(output_path)
    written_count = 0

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for data in dataset.datas:
                try:
                    json_data = convert_to_jsonl_format(data)
                    f.write(
                        json.dumps(json_data, ensure_ascii=False, cls=NumpyEncoder)
                        + "\n"
                    )
                    written_count += 1
                except Exception as e:
                    logger.error(
                        f"Error processing data item {written_count}: {str(e)}"
                    )
                    logger.error(f"Data item type: {type(data)}")
                    if hasattr(data, "metadata"):
                        logger.error(f"Data metadata: {data.metadata}")
                    continue

        logger.info(f"✅ Successfully exported {written_count} items to {output_path}")

    except Exception as e:
        logger.error(f"❌ Error exporting to JSONL: {str(e)}")


def main():
    logger.info("🚀 Starting ORM data processing...")
    dataset = load_and_process_dataset()

    if dataset:
        logger.info("✅ ORM data processing completed successfully!")
    else:
        logger.error("❌ ORM data processing failed!")


if __name__ == "__main__":
    main()
