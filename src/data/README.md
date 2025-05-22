# RM-Gallery Data Processing System

## Overview

The RM-Gallery data processing system is designed to handle the loading, processing, and transformation of evaluation data for reinforcement learning models. The system is built with flexibility and extensibility in mind, allowing for easy addition of new data sources and processing steps.

## 概述

RM-Gallery 数据处理系统设计用于处理强化学习模型的评估数据的加载、处理和转换。该系统以灵活性和可扩展性为核心，支持轻松添加新的数据源和处理步骤。

## System Architecture

### System Diagram

### Core Components

1. **Base Data Structure (`base.py`)**
   - Defines the base data classes using Pydantic models
   - Includes `BaseData` and `BaseDataSet` classes
   - Provides common functionality for data management

2. **Data Schema (`data_schema.py`)**
   - Defines the core data structures using Pydantic models
   - Includes `ContentDict`, `ContextDict`, `Reward`, `DataInfo`, and `EvaluationSample`
   - Ensures data validation and type safety

3. **Base Operator (`base_operator.py`)**
   - Defines the core operator framework
   - Contains two main classes:
     - `Operator`: Abstract base class for all operators
       - Requires implementation of `process_dataset` method
       - Provides common functionality for all operators
     - `OperatorFactory`: Factory class for operator creation
       - Implements the Factory pattern for operator instantiation
       - Maintains a registry of operator creation functions
       - Supports dynamic operator creation from configuration
       - Handles both custom operators and built-in operators (filter, map, reward)
       - Includes special handling for data-juicer operators

4. **Data Load Strategy (`data_load_strategy.py`)**
   - Implements the Strategy pattern for data loading
   - Supports multiple data sources (local files, Huggingface)
   - Uses a registry system with wildcard matching for strategy management
   - Includes built-in strategies for JSON, JSONL, and Parquet files

5. **Data Processor (`data_processor.py`)**
   - Implements the Pipeline pattern for data processing
   - Provides a framework for adding custom operators
   - Includes built-in operators for filtering, mapping, and reward processing
   - Supports integration with data-juicer operators
   - Features a unified operator registration system
   - Supports multiple operator types (filter, map, group, reward)
   - Enables easy operator composition and chaining

6. **Data Builder (`data_builder.py`)**
   - Handles the construction of evaluation datasets
   - Manages data loading and saving operations
   - Provides a clean interface for data manipulation

7. **Operators (`ops/`)**
   - Contains various data processing operators
   - Each operator is implemented as a separate module
   - Current operators include:
     - `text_length_filter.py`: Filters data based on text length
     - `conversation_turn_filter.py`: Filters conversations based on turn count

### Design Patterns

1. **Strategy Pattern**
   - Used for data loading strategies
   - Allows easy addition of new data sources
   - Provides flexible strategy selection with wildcard matching

2. **Pipeline Pattern**
   - Used for data processing
   - Enables modular processing steps through operators
   - Supports both item-level and dataset-level processing
   - Allows flexible operator composition and chaining
   - Supports multiple operator types in a single pipeline

3. **Factory Pattern**
   - Used for operator creation
   - Provides a registry system for operator registration
   - Enables dynamic operator creation from configuration
   - Supports both custom and built-in operators
   - Handles special cases like data-juicer integration
   - Supports multiple operator types through a unified interface

4. **Template Method Pattern**
   - Used in the operator framework
   - Provides a common interface for all operators
   - Allows for easy extension of operator functionality
   - Supports multiple operator types through inheritance

## Usage Examples

### Basic Usage

```python
from data_builder import load_dataset_from_yaml

# Load and process dataset using YAML configuration
dataset = load_dataset_from_yaml('data_load.yaml')
```

### YAML Configuration

```yaml
dataset:
    name: "preference-test-sets"
    description: "Preference test dataset"
    version: "1.0.0"
    configs:
        type: local
        source: rewardbench
        dimension: helpfulness
        path: /path/to/data.parquet
        limit: 2000
    processors:
        - type: filter
          name: conversation_turn_filter
          config:
            min_turns: 2
            max_turns: 6
        - type: filter
          name: text_length_filter
          config:
            min_length: 10
            max_length: 1000
        - type: map
          name: text_cleaner
          config:
            remove_urls: true
        - type: reward
          name: reward_calculator
          config:
            model: "gpt-4"
```

### Adding a New Operator

1. Create a new operator file in the `ops/` directory:
```python
from base_operator import Operator, OperatorFactory

class NewOperator(Operator):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        
    def process_dataset(self, items):
        # Implement your processing logic
        return items

# Register the operator
@OperatorFactory.register('new_operator')
def create_new_operator(config):
    return NewOperator(
        name=config.get('name', 'new_operator'),
        config=config.get('config', {})
    )
```

2. Add the operator to your YAML configuration:
```yaml
processors:
    - type: filter
      name: new_operator
      config:
        # Your operator configuration
```

## Best Practices

1. **Error Handling**
   - Use try-except blocks in operators
   - Log errors using loguru
   - Provide meaningful error messages

2. **Configuration**
   - Validate configurations in strategy classes
   - Provide default values where appropriate
   - Document configuration options

3. **Performance**
   - Use batch processing when possible
   - Implement efficient data loading strategies
   - Consider memory usage for large datasets

4. **Testing**
   - Write unit tests for operators
   - Test with different data sources
   - Include edge cases in tests

5. **Operator Design**
   - Keep operators focused on a single responsibility
   - Use the appropriate operator type for your use case
   - Consider operator composition for complex processing
   - Document operator behavior and configuration options

## Dependencies

- pydantic
- loguru
- pandas (for Parquet support)
- data-juicer (for advanced data processing)

## 系统架构

### 核心组件

1. **基础数据结构 (`base.py`)**
   - 使用 Pydantic 模型定义基础数据类
   - 包含 `BaseData` 和 `BaseDataSet` 类
   - 提供数据管理的通用功能

2. **数据模式 (`data_schema.py`)**
   - 使用 Pydantic 模型定义核心数据结构
   - 包含 `ContentDict`、`ContextDict`、`Reward`、`DataInfo` 和 `EvaluationSample`
   - 确保数据验证和类型安全

3. **基础算子 (`base_operator.py`)**
   - 定义核心算子框架
   - 包含两个主要类：
     - `Operator`：所有算子的抽象基类
       - 要求实现 `process_dataset` 方法
       - 为所有算子提供通用功能
     - `OperatorFactory`：算子创建工厂类
       - 实现工厂模式进行算子实例化
       - 维护算子创建函数注册表
       - 支持从配置动态创建算子
       - 处理自定义算子和内置算子（过滤、映射、奖励）
       - 包含对 data-juicer 算子的特殊处理

4. **数据加载策略 (`data_load_strategy.py`)**
   - 实现数据加载的策略模式
   - 支持多个数据源（本地文件、Huggingface）
   - 使用带通配符匹配的注册系统进行策略管理
   - 包含 JSON、JSONL 和 Parquet 文件的内置策略

5. **数据处理器 (`data_processor.py`)**
   - 实现数据处理的管道模式
   - 提供添加自定义算子的框架
   - 包含用于过滤、映射和奖励处理的内置算子
   - 支持与 data-juicer 算子的集成
   - 提供统一的算子注册系统
   - 支持多种算子类型（过滤、映射、分组、奖励）
   - 支持算子的组合和链接

6. **数据构建器 (`data_builder.py`)**
   - 处理评估数据集的构建
   - 管理数据加载和保存操作
   - 提供清晰的数据操作接口

7. **算子 (`ops/`)**
   - 包含各种数据处理算子
   - 每个算子作为单独的模块实现
   - 当前算子包括：
     - `text_length_filter.py`：基于文本长度过滤数据
     - `conversation_turn_filter.py`：基于对话轮次过滤对话
     - `group_filter.py`：区分正负样本集

### 设计模式

1. **策略模式**
   - 用于数据加载策略
   - 允许轻松添加新的数据源
   - 提供灵活的策略选择，支持通配符匹配

2. **管道模式**
   - 用于数据处理
   - 通过算子实现模块化处理步骤
   - 支持项目级和数据集级处理
   - 允许灵活的算子组合和链接
   - 支持在单个管道中使用多种算子类型

3. **工厂模式**
   - 用于算子创建
   - 提供算子注册系统
   - 支持从配置动态创建算子
   - 支持自定义和内置算子
   - 处理特殊用例，如 data-juicer 集成
   - 通过统一接口支持多种算子类型

4. **模板方法模式**
   - 用于算子框架
   - 为所有算子提供通用接口
   - 允许轻松扩展算子功能
   - 通过继承支持多种算子类型

## 最佳实践

1. **错误处理**
   - 在算子中使用 try-except 块
   - 使用 loguru 记录错误
   - 提供有意义的错误消息

2. **配置**
   - 在策略类中验证配置
   - 在适当的地方提供默认值
   - 记录配置选项

3. **性能**
   - 尽可能使用批处理
   - 实现高效的数据加载策略
   - 考虑大数据集的内存使用

4. **测试**
   - 为算子编写单元测试
   - 使用不同的数据源进行测试
   - 在测试中包含边缘情况

5. **算子设计**
   - 保持算子专注于单一职责
   - 为您的用例使用适当的算子类型
   - 考虑使用算子组合进行复杂处理
   - 记录算子行为和配置选项 