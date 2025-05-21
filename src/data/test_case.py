import json
from loguru import logger
from pathlib import Path

from .data_builder import load_dataset_from_yaml

def convert_to_jsonl_format(data):
    """Convert BaseData object to JSONL format"""
    return data.model_dump(mode='json')  # 使用json模式进行序列化，会自动处理datetime

# 加载数据集
try:
    dataset = load_dataset_from_yaml("/Users/xielipeng/RM-Gallery/src/data/data_load.yaml")
    logger.info(f"Dataset loaded successfully. Type: {type(dataset)}")
    logger.info(f"Dataset name: {dataset.name}")
    logger.info(f"Dataset description: {dataset.description}")
    logger.info(f"Number of data items: {len(dataset.datas)}")

    
    # 输出JSONL文件
    output_path = Path("anthropic_helpful_0521.jsonl")
    written_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in dataset.datas:
            json_data = convert_to_jsonl_format(data)
            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
            written_count += 1
    
    logger.info(f"Data has been written to {output_path}")
    
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise