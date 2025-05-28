import json
from loguru import logger
from pathlib import Path
from datetime import datetime 

# 使用最新的数据加载接口
from rm_gallery.core.data.build import create_build_module_from_yaml

def convert_to_jsonl_format(data):
    """Convert BaseData object to JSONL format"""
    return data.model_dump(mode='json')

def load_and_process_dataset():
    """load and process prm dataset"""
    config_path = "./src/data/example/prm_data_load.yaml"
    
    try:
        logger.info(f"Loading PRM dataset from config: {config_path}")
        
        # 创建构建模块
        builder = create_build_module_from_yaml(config_path)
        logger.info(f"Created builder: {builder.name}")
    
        # 运行完整流程
        logger.info("Starting pipeline execution...")
        dataset = builder.run()
        
        # 导出数据
        output_path = f"{dataset.name.replace('-', '_')}_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        export_dataset(dataset, output_path)
        
        return dataset
            
    except Exception as e:
        logger.error(f"❌ Error in PRM dataset processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def export_dataset(dataset, output_path):
    """export prm dataset to jsonl file"""
    if not dataset or not dataset.datas:
        logger.warning("No PRM data to export")
        return
        
    output_path = Path(output_path)
    written_count = 0
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in dataset.datas:
                json_data = convert_to_jsonl_format(data)
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
                written_count += 1
        
        logger.info(f"✅ Successfully exported {written_count} PRM items to {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error exporting PRM data to JSONL: {str(e)}")

def main():
    logger.info("🚀 Starting PRM data processing...")
    dataset = load_and_process_dataset()
    
    if dataset:
        logger.info("✅ PRM data processing completed successfully!")
    else:
        logger.error("❌ PRM data processing failed!")

if __name__ == "__main__":
    main() 