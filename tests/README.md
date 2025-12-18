# 项目测试策略

本文档概述了 RM-Gallery 框架的测试策略，涵盖了测试类型、测试方法和质量保证要求。

## 1. 概述

为了确保 RM-Gallery 项目能够稳定可靠地运行，我们需要建立一套完整的测试体系。项目测试分为单元测试和质量测试两种类型：
- 所有项目关键组件都需要完成单元测试
- 所有LLMGrader需要完成单元测试和质量测试

## 2. 测试类型

### 2.1 单元测试 (Unit Testing)

单元测试是最基础的测试类型，针对项目中各个组件的单个功能进行验证。

##### 测试原则
- **隔离性**: 每个测试应独立运行，不依赖其他测试
- **针对性**: 每个测试应专注于验证一个特定功能
- **自动化**: 测试应能自动运行并给出明确结果

##### 离线测试框架
所有外部服务（特别是 LLM API）必须进行模拟以实现离线测试：

```python
from unittest.mock import AsyncMock, patch
import pytest

# 模拟 LLM 调用的示例
@pytest.mark.unit
class TestLLMGraderOffline:
    @pytest.mark.asyncio
    async def test_llm_based_grader_offline(self):
        # 设置模拟，注意LLMGrader期望从metadata中获取结果
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 5,
            "reason": "Perfect"
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # 使用模拟初始化 grader
        grader = MyLLMGrader(model=mock_model)

        # 执行测试
        result = await grader.aevaluate(query="test", response="test response")

        # 断言
        assert result.score == 5
        assert result.reason == "Perfect"

        # 验证模型调用
        mock_model.achat.assert_called_once()
```

##### 标准测试结构
每个组件都应按以下方式组织测试：

``python
# tests/[component_type]/[category]/test_[component_name].py
import pytest
from unittest.mock import AsyncMock, patch

from rm_gallery.core.[component_type].[category].[component_module] import [ComponentClass]

@pytest.mark.unit
class Test[ComponentClass]:
    """Test suite for [ComponentClass]"""

    def test_initialization(self):
        """测试成功初始化"""
        mock_dependency = AsyncMock()
        component = [ComponentClass](dependency=mock_dependency, **kwargs)
        assert component.name == "[expected_name]"
        assert component.mode == [expected_mode]
        # 注意：只有异步方法才需要@pytest.mark.asyncio装饰器

    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """测试使用有效输入的成功操作"""
        # 设置模拟，注意LLMGrader期望的返回格式
        mock_response = AsyncMock()

        mock_response.parsed = {
            "score": 5,
            "reason": "Good response"
        }

        mock_dependency = AsyncMock()
        mock_dependency.method = AsyncMock(return_value=mock_response)

        component = [ComponentClass](dependency=mock_dependency)
        result = await component.operation(**valid_inputs)
        assert isinstance(result, [ExpectedResultType])
        assert 0 <= result.score <= 5  # 或适当范围

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """测试边缘情况，如空输入、极值"""
        mock_dependency = AsyncMock()
        component = [ComponentClass](dependency=mock_dependency)
        result = await component.operation(**edge_case_inputs)
        # 适当的断言

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试优雅的错误处理"""
        # 设置模拟以引发异常
        mock_dependency = AsyncMock()
        mock_dependency.method = AsyncMock(side_effect=Exception("API Error"))

        component = [ComponentClass](dependency=mock_dependency)
        result = await component.operation(**invalid_inputs)
        # 应返回有意义的错误结果，而不是抛出异常
        assert result.score == 0  # 或其他适当的错误值
        assert "API Error" in result.reason
```


### 2.2 质量测试 (Quality Testing)

质量测试用于确保 LLMGrader 的评估质量符合使用要求。这类测试关注的是 LLMGrader 是否能够正确地区分高质量和低质量的响应。质量测试必须配置 API_KEY，基于真实数据进行在线验证评估质量。通过与 [GradingRunner](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/runner/grading_runner.py#L31-L175) 结合使用，可以显著提高测试执行效率。

#### 环境变量检测

质量测试需要配置真实的模型 API 密钥，测试脚本应检测环境变量以确认是否执行测试：

```python
import os
import pytest

# 检测是否配置了必要的 API 密钥和基础URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 根据环境变量决定是否运行质量测试
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# 如果没有配置 API 密钥和基础URL，则跳过所有质量测试
pytestmark = pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
```

#### 基于黄金标准数据集的质量测试

``python
# tests/quality/test_grader_quality.py
import os
import pytest
import pandas as pd
import numpy as np

from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.validation import (
    AccuracyAnalyzer,
    ConsistencyAnalyzer
)
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# 检测是否配置了必要的 API 密钥和基础URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 根据环境变量决定是否运行质量测试
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# 如果没有配置 API 密钥和基础URL，则跳过所有质量测试
@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestGraderQuality:
    """测试 Grader 的评估质量"""

    @pytest.fixture
    def dataset(self):
        """加载黄金标准数据集"""
        # 这里应该加载一个包含查询、响应和人工标注质量分数的数据集
        # 数据格式示例：
        # [
        #   {
        #     "query": "什么是光合作用？",
        #     "response": "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖的过程。",
        #     "human_score": 5  # 5表示高质量（1-5评分）
        #   },
        #   ...
        # ]
        return [
            {
                "query": "什么是光合作用？",
                "response": "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖的过程，这是生态系统中非常重要的一个环节。",
                "context": "生物学基础知识",
                "human_score": 5  # 5表示高质量（1-5评分）
            },
            {
                "query": "什么是光合作用？",
                "response": "光合作用是一种化学反应。",
                "context": "生物学基础知识",
                "human_score": 2  # 2表示低质量
            }
        ]

    @pytest.fixture
    def model(self):
        """根据环境变量返回OpenAIChatModel实例"""
        if OPENAI_API_KEY:
            config = {"model": "qwen-max", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # 这种情况不应该发生，因为测试已经被跳过了
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, dataset, model):
        """测试 Grader 区分高低质量响应的能力（使用 Runner）"""
        # 创建使用真实模型的 Grader，传入OpenAIChatModel实例
        grader = HelpfulnessGrader(model=model)

        # 使用 mapper 配置数据转换
        grader_configs = {
            "helpfulness": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 准备测试数据
        test_data = dataset
        human_scores = [item["human_score"] for item in dataset]

        # 使用 Runner 执行批量评估
        results = await runner.arun(dataset=test_data)

        # 使用 AccuracyAnalyzer 计算准确率指标
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score"
        )

        # 使用 ConsistencyAnalyzer 计算一致性指标
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            first_run_results=results["helpfulness"],
            second_run_results=results["helpfulness"]
        )

        # 断言质量指标达到预期阈值
        assert accuracy_result.accuracy >= 0.7, f"准确率低于阈值: {accuracy_result.accuracy}"
        assert consistency_result.consistency >= 0.9, f"评估一致性不足: {consistency_result.consistency}"

        # 验证分析结果包含必要的元数据
        assert "explanation" in accuracy_result.metadata
        assert "explanation" in consistency_result.metadata

        assert accuracy_result.name == "Accuracy Analysis"
        assert consistency_result.name == "Consistency Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """测试 Grader 评估的一致性（使用 Runner）"""
        # 创建使用真实模型的 Grader，传入OpenAIChatModel实例
        grader = HelpfulnessGrader(model=model)

        # 使用重复配置实现一致性测试
        grader_configs = {
            "helpfulness_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context"
                }
            ),
            "helpfulness_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 准备测试数据
        test_data = dataset

        # 使用 Runner 执行批量评估
        results = await runner.arun(dataset=test_data)

        # 使用 ConsistencyAnalyzer 计算一致性指标
        from rm_gallery.core.analyzer.validation import ConsistencyAnalyzer
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            first_run_results=results["helpfulness_run1"],
            second_run_results=results["helpfulness_run2"]
        )

        # 断言一致性指标达到预期阈值
        assert consistency_result.consistency >= 0.9, f"评估一致性不足: {consistency_result.consistency}"

        # 验证分析结果包含必要的元数据
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"
```

#### 基于对抗样本的质量测试

``python
# tests/quality/test_adversarial_examples.py
import os
import pytest
from unittest.mock import AsyncMock

from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.validation import (
    FalsePositiveAnalyzer,
    FalseNegativeAnalyzer
)
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# 检测是否配置了必要的 API 密钥和基础URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 根据环境变量决定是否运行质量测试
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# 如果没有配置 API 密钥和基础URL，则跳过所有质量测试
@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
@pytest.mark.quality
class TestAdversarialExamples:
    """测试 Grader 对对抗样本的鲁棒性"""

    @pytest.fixture
    def dataset(self):
        """加载对抗样本数据集"""
        # 这里应该加载一个包含查询、正确响应、错误响应及其标签的数据集
        # 数据格式示例：
        # [
        #   {
        #     "query": "地球是什么？",
        #     "correct_response": "地球是太阳系中的行星之一",
        #     "incorrect_response": "地球是宇宙中唯一有生命的星球",
        #     "context": "基础天文学知识",
        #     "correct_label": 5,
        #     "incorrect_label": 2
        #   },
        #   ...
        # ]
        return [
            {
                "query": "地球是什么？",
                "correct_response": "地球是太阳系中的行星之一",
                "incorrect_response": "地球是宇宙中唯一有生命的星球",
                "context": "基础天文学知识",
                "correct_label": 5,   # 高质量评分
                "incorrect_label": 2  # 低质量评分
            },
            {
                "query": "水的化学分子式是什么？",
                "correct_response": "水的化学分子式是H2O",
                "incorrect_response": "水的化学分子式是CO2",
                "context": "基础化学知识",
                "correct_label": 5,   # 高质量评分
                "incorrect_label": 2  # 低质量评分
            }
        ]

    @pytest.fixture
    def model(self):
        """根据环境变量返回OpenAIChatModel实例"""
        if OPENAI_API_KEY:
            config = {"model": "qwen-max", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # 这种情况不应该发生，因为测试已经被跳过了
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_helpfulness_with_runner(self, dataset, model):
        """测试对帮助性对抗样本的识别能力（使用 Runner）"""
        # 创建使用真实模型的 Grader，传入OpenAIChatModel实例
        grader = HelpfulnessGrader(model=model)

        # 使用 mapper 配置数据转换
        # 通过配置GraderConfig实现同时评估正确和错误答案
        grader_configs = {
            "helpfulness_correct": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "correct_response",
                    "context": "context"
                }
            ),
            "helpfulness_incorrect": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "incorrect_response",
                    "context": "context"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 准备测试数据
        test_data = dataset

        # 使用 Runner 执行批量评估
        results = await runner.arun(dataset=test_data)

        # 使用 FalsePositiveAnalyzer 计算误报率指标
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness_incorrect"],  # 错误响应的评估结果
            label_path="incorrect_label"
        )

        # 使用 FalseNegativeAnalyzer 计算漏报率指标
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness_correct"],  # 正确响应的评估结果
            label_path="correct_label"
        )

        # 断言误报率和漏报率达到预期阈值
        assert fp_result.false_positive_rate <= 0.3, f"误报率过高: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"漏报率过高: {fn_result.false_negative_rate}"

        # 验证分析结果包含必要的元数据
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
```

## 3. 测试执行

### 3.1 本地开发环境

```
# 运行所有测试
pytest tests/ -v

# 运行特定类别
pytest tests/graders/text/ -v

# 运行带覆盖率
pytest tests/ --cov=rm_gallery --cov-report=html

# 运行性能测试
pytest tests/performance/ --benchmark-only

# 运行质量测试（仅模拟测试）
pytest tests/ -v -m quality

# 运行所有质量测试（包括真实模型测试，如果配置了API密钥）
pytest tests/ -v -m quality --run-live

# 运行特定类型的测试（使用标记）
pytest tests/ -m unit       # 运行单元测试
pytest tests/ -m quality     # 运行质量测试
```

### 3.2 持续集成 (CI/CD)

每次推送和拉取请求时，测试都会通过 GitHub Actions 自动运行，结果报告给 Codecov 以进行覆盖率跟踪。

#### GitHub Actions 工作流

```
# .github/workflows/project-tests.yml
name: Project Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install

    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=rm_gallery -m unit

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v -m integration

    - name: Run quality tests (simulation only)
      run: |
        pytest tests/ -v -m quality

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

## 4. 质量保证要求

### 4.1 代码覆盖率

- **行覆盖率**：90%
- **分支覆盖率**：80%
- **函数覆盖率**：100%

### 4.2 测试用例要求

每个组件必须有涵盖以下内容的测试用例：

1. **正常操作**：产生预期输出的典型输入
2. **边界条件**：边缘情况，如空字符串、最大长度输入
3. **错误条件**：无效输入、缺少参数
4. **配置变化**：不同的初始化参数

## 5. 测试维护

### 5.1 添加新组件时

1. 在 `tests/[component_type]/[category]/` 中创建相应的测试文件
2. 实现标准测试套件（初始化、评估、边缘情况、错误处理）
3. 确保达到最低覆盖率阈值
4. 对于LLMGrader，还需要准备黄金标准数据集用于质量测试
5. 更新文档

### 5.2 修改现有组件时

1. 更新现有测试以匹配新行为
2. 验证向后兼容性或相应更新测试
3. 如果行为有显著改变，对于LLMGrader需要更新质量测试
4. 重新运行完整测试套件以确保没有回归