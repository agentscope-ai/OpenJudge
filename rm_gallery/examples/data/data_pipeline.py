from rm_gallery.core.data.load import create_load_module
from rm_gallery.core.data.process import OperatorFactory, create_process_module
from rm_gallery.core.data.build import create_build_module

def example_data_pipeline():
    
    load_module = create_load_module(
        name="preference-test-sets",
        config={},
        load_strategy_type="local",
        data_source="rewardbench",
        dimension="helpfulness",
        load_config={"path": "./data/preference-test-sets/data/anthropic_helpful-00000-of-00001.parquet", "limit": 2000}
    )

    process_module = create_process_module(
        name="preference-test-sets-processor",
        config={},
        operators=[
            OperatorFactory.create_operator({
                "type": "filter",
                "name": "conversation_turn_filter",
                "config": {"min_turns": 2, "max_turns": 6}
            }),
            OperatorFactory.create_operator({
                "type": "filter", 
                "name": "rm_text_length_filter",
                "config": {"min_length": 10, "max_length": 1000}
            }),
            OperatorFactory.create_operator({
                "type": "data_juicer",
                "name": "text_length_filter", 
                "config": {"min_len": 50, "max_len": 100}
            }),
            OperatorFactory.create_operator({
                "type": "group",
                "name": "group_filter",
                "config": {"train_ratio": 0.7, "test_ratio": 0.3}
            }),
        ]
    )
    
    build_module = create_build_module(
        name="simple_pipeline",
        config={},
        load_module=load_module,
        process_module=process_module
    )

    result = build_module.run()
    
    print(f"Pipeline completed successfully!")
    print(f"Processed {len(result)} items")
    print(f"Output dataset: {result.name}")

if __name__ == "__main__":
    example_data_pipeline()