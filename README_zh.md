<div align="center">

<img src="./docs/images/logo.svg" alt="Open-Judge Logo" width="500">

<br/>

<h3>
  <em>全面评估，质量驱动：提升应用效果</em>
</h3>

<p>
  🌟 <em>如果您觉得 OpenJudge 有帮助，请给我们一个 <b>Star</b>！</em> 🌟
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://pypi.org/project/py-openjudge/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-blue?logo=pypi)](https://pypi.org/project/py-openjudge/)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://agentscope-ai.github.io/OpenJudge/)

[📖 文档](https://agentscope-ai.github.io/OpenJudge/) | [🤝 贡献指南](https://agentscope-ai.github.io/OpenJudge/community/contributing/) | [English](./README.md)

</div>




OpenJudge 是一个 **开源评估框架**，用于 **AI 应用**（如智能体或聊天机器人）的**质量评估**，并驱动**持续优化**。

> 在实践中，应用卓越依赖可信的评估流程：收集测试数据 → 定义评分器 → 规模化运行评估 → 分析缺陷 → 快速迭代。

OpenJudge 提供**即用型评分器**，并支持生成**场景特定的评估标准（作为评分器）**，让这一流程更**简单**、更**专业**、更易于集成。它还可将评分结果转换为**奖励信号**，帮助你**微调**并优化应用。

---

## 📑 目录

- [核心特性](#-核心特性)
- [最新动态](#最新动态)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [集成](#-集成)
- [贡献](#-贡献)
- [社区](#-社区)
- [引用](#-引用)
---

## ✨ 核心特性

### 📦 系统化、质量保证的评分器库

访问 **50+ 生产就绪的评分器**，具有全面的分类体系，经过严格验证以确保可靠性能。

<table>
<tr>
<td width="33%" valign="top">

#### 🎯 通用

**关注点：** 语义质量、功能正确性、结构合规性

**核心评分器：**
- `Relevance` - 语义相关性评分
- `Similarity` - 文本相似度测量
- `Syntax Check` - 代码语法验证
- `JSON Match` - 结构合规性检查

</td>
<td width="33%" valign="top">

#### 🤖 智能体

**关注点：** 智能体生命周期、工具调用、记忆、计划可行性、轨迹质量

**核心评分器：**
- `Tool Selection` - 工具选择准确性
- `Memory` - 上下文保持能力
- `Plan` - 策略可行性
- `Trajectory` - 路径优化

</td>
<td width="33%" valign="top">

#### 🖼️ 多模态

**关注点：** 图文一致性、视觉生成质量、图像有用性

**核心评分器：**
- `Image Coherence` - 视觉-文本对齐
- `Text-to-Image` - 生成质量
- `Image Helpfulness` - 图像贡献度

</td>
</tr>
</table>

- 🌐 **多场景覆盖：** 广泛支持包括智能体、文本、代码、数学和多模态任务在内的多种领域。→ [探索支持的场景](https://agentscope-ai.github.io/OpenJudge/built_in_graders/overview/)
- 🔄 **全面的智能体评估：** 不仅评估最终结果，我们还评估整个生命周期——包括轨迹、记忆、反思和工具使用。→ [智能体生命周期评估](https://agentscope-ai.github.io/OpenJudge/built_in_graders/agent_graders/)
- ✅ **质量保证：** 每个评分器都配有基准数据集和 pytest 集成用于验证。→ [查看基准数据集](https://huggingface.co/datasets/agentscope-ai/OpenJudge)


### 🛠️ 灵活的评分器构建方法
选择适合您需求的构建方法：
* **自定义：** 需求明确但没有现成的评分器？如果您有明确的规则或逻辑，使用我们的 Python 接口或 Prompt 模板快速定义您自己的评分器。👉 [自定义评分器开发指南](https://agentscope-ai.github.io/OpenJudge/building_graders/create_custom_graders/)
* **零样本评估标准生成：** 不确定使用什么标准，也没有标注数据？只需提供任务描述和可选的示例查询，LLM 将自动为您生成评估标准。非常适合快速原型开发。👉 [零样本评估标准生成指南](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#simple-rubric-zero-shot-generation)
* **数据驱动的评估标准生成：** 需求模糊但有少量样例？使用 GraderGenerator 从您的标注数据中自动总结评估标准，并生成基于 LLM 的评分器。👉 [数据驱动评估标准生成指南](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#iterative-rubric-data-driven-generation)
* **训练评判模型：** 拥有大量数据且需要极致性能？使用我们的训练流程来训练专用的评判模型。适用于基于 Prompt 的评分无法满足的复杂场景。👉 [训练评判模型](https://agentscope-ai.github.io/OpenJudge/building_graders/training_judge_models/)


### 🔌 轻松集成

如果您正在使用主流可观测性平台（如 **LangSmith** 或 **Langfuse**），我们提供无缝集成方案，可增强平台的评测器和自动评测能力。我们也提供与训练框架（如 **VERL**）的集成方案，用于强化学习训练。👉 查看 [集成](#-集成) 了解详情

----
## 最新动态

- **2026-02-12** - 📚 **Reference Hallucination Arena** - 评估大语言模型学术引用幻觉的基准测试。 👉 [文档](./docs/validating_graders/ref_hallucination_arena.md)

- **2025-12-26** - 在 [PyPI](https://pypi.org/project/py-openjudge/) 上发布 OpenJudge v0.2.0 - **重大更新！** 此版本通过在奖励构建之上添加对多样化评估场景的强大支持，扩展了我们的核心能力。通过统一奖励和评估信号，OpenJudge v0.2.0 提供了一种更全面的方法来优化应用性能和卓越性。→ [迁移指南](#迁移指南v01x--v020)

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - 我们发布了一篇关于学习可泛化奖励标准以实现稳健建模的新论文。
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - 我们介绍了对齐评判反馈和提高强化学习稳定性的技术。
- **2025-07-09** - 在 [PyPI](https://pypi.org/project/rm-gallery/) 上发布 OpenJudge v0.1.0

---

## 📥 安装

```bash
pip install py-openjudge
```

> 💡 更多安装方法可在 [快速开始指南](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/) 中找到。

---

## 🚀 快速开始

### 简单示例

一个评估单条回复的简单示例：

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader

async def main():
    # 1️⃣ 创建模型客户端
    model = OpenAIChatModel(model="qwen3-32b")
    # 2️⃣ 初始化评分器
    grader = RelevanceGrader(model=model)
    # 3️⃣ 准备数据
    data = {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI that enables computers to learn from data.",
    }
    # 4️⃣ 评估
    result = await grader.aevaluate(**data)
    print(f"Score: {result.score}")   # Score: 4
    print(f"Reason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 使用内置评分器评估 LLM 应用

利用多个内置评分器对 LLM 应用进行全面评估：👉  [查看所有内置评分器](https://agentscope-ai.github.io/OpenJudge/built_in_graders/overview/)

> **业务场景：** 评估电商客服智能体处理订单咨询的表现。我们从 **相关性**、**幻觉**、**工具选择** 三个维度进行评估。

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common import RelevanceGrader, HallucinationGrader
from openjudge.graders.agent.tool.tool_selection import ToolSelectionGrader
from openjudge.runner import GradingRunner
from openjudge.runner.aggregator import WeightedSumAggregator
from openjudge.analyzer.statistical import DistributionAnalyzer

TOOL_DEFINITIONS = [
    {"name": "query_order", "description": "Query order status and logistics information", "parameters": {"order_id": "str"}},
    {"name": "query_logistics", "description": "Query detailed logistics tracking", "parameters": {"order_id": "str"}},
    {"name": "estimate_delivery", "description": "Estimate delivery time", "parameters": {"order_id": "str"}},
]
# 准备数据集
dataset = [{
    "query": "Where is my order ORD123456?",
    "response": "Your order ORD123456 has arrived at the Beijing distribution center and is expected to arrive tomorrow.",
    "context": "Order ORD123456: Arrived at Beijing distribution center, expected to arrive tomorrow.",
    "tool_definitions": TOOL_DEFINITIONS,
    "tool_calls": [{"name": "query_order", "arguments": {"order_id": "ORD123456"}}],
    # ... 更多测试样例
}]
async def main():
    # 1️⃣ 初始化判别模型
    model = OpenAIChatModel(model="qwen3-max")
    # 2️⃣ 配置多个评分器
    grader_configs = {
        "relevance": {"grader": RelevanceGrader(model=model), "mapper": {"query": "query", "response": "response"}},
        "hallucination": {"grader": HallucinationGrader(model=model), "mapper": {"query": "query", "response": "response", "context": "context"}},
        "tool_selection": {"grader": ToolSelectionGrader(model=model), "mapper": {"query": "query", "tool_definitions": "tool_definitions", "tool_calls": "tool_calls"}},
    }
    # 3️⃣ 设置聚合器计算综合分
    aggregator = WeightedSumAggregator(name="overall_score", weights={"relevance": 0.3, "hallucination": 0.4, "tool_selection": 0.3})
    # 4️⃣ 运行评估
    results = await GradingRunner(grader_configs=grader_configs, aggregators=[aggregator], max_concurrency=5).arun(dataset)
    # 5️⃣ 生成评估报告
    overall_stats = DistributionAnalyzer().analyze(dataset, results["overall_score"])
    print(f"{'Overall Score':<20} | {overall_stats.mean:>15.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 为你的场景构建自定义评分器

#### 零样本评估标准生成

无需标注数据，基于任务描述生成自定义评分器： 👉 [零样本生成评分准则教程](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#simple-rubric-zero-shot-generation)

**适用场景：** 没有标注数据但能清晰描述任务，想快速原型验证。

```python
import asyncio
from openjudge.generator.simple_rubric import SimpleRubricsGenerator, SimpleRubricsGeneratorConfig
from openjudge.models import OpenAIChatModel

async def main():
    # 1️⃣ 配置生成器
    config = SimpleRubricsGeneratorConfig(
        grader_name="customer_service_grader",
        model=OpenAIChatModel(model="qwen3-max"),
        task_description="E-commerce AI customer service primarily handles order inquiry tasks (such as logistics status and ETA) while focusing on managing customer emotions.",
        min_score=1,
        max_score=3,
    )
    # 2️⃣ 生成评分器
    generator = SimpleRubricsGenerator(config)
    grader = await generator.generate(dataset=[], sample_queries=[])
    # 3️⃣ 查看生成的评估标准
    print("Generated Rubrics:", grader.kwargs.get("rubrics"))
    # 4️⃣ 使用评分器
    result = await grader.aevaluate(
        query="My order is delayed, what should I do?",
        response="I understand your concern. Let me check your order status..."
    )
    print(f"\nScore: {result.score}/3\nReason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 数据驱动的评估标准生成

从标注样本中自动学习评估标准：👉 [数据驱动生成评估标准教程](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#iterative-rubric-data-driven-generation)

**适用场景：** 拥有标注数据，需求高准确度的生产级评分器，尤其当评估标准隐含在数据中。

```python
import asyncio
from openjudge.generator.iterative_rubric.generator import IterativeRubricsGenerator, IterativePointwiseRubricsGeneratorConfig
from openjudge.models import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# 准备标注数据集（简化示例，实际推荐 10+ 条）
labeled_dataset = [
    {"query": "My order hasn't arrived after 10 days, I want to complain!", "response": "I sincerely apologize for the delay. I completely understand your frustration! Your order was delayed due to weather conditions, but it has now resumed shipping and is expected to arrive tomorrow. I've marked it for priority delivery.", "label_score": 5},
    {"query": "Where is my package? I need it urgently!", "response": "I understand your urgency! Your package is currently out for delivery and is expected to arrive before 2 PM today. The delivery driver's contact number is 138xxxx.", "label_score": 5},
    {"query": "Why hasn't my order arrived yet? I've been waiting for days!", "response": "Your order is expected to arrive the day after tomorrow.", "label_score": 2},
    {"query": "The logistics hasn't updated in 3 days, is it lost?", "response": "Hello, your package is not lost. It's still in transit, please wait patiently.", "label_score": 3},
    # ... 更多标注样例
]

async def main():
    # 1️⃣ 配置生成器
    config = IterativePointwiseRubricsGeneratorConfig(
        grader_name="customer_service_grader_v2", model=OpenAIChatModel(model="qwen3-max"),
        min_score=1, max_score=5,
        enable_categorization=True, categories_number=5,  # 启用归类聚合，聚合为 5 个主题
    )
    # 2️⃣ 从标注数据生成评分器
    generator = IterativeRubricsGenerator(config)
    grader = await generator.generate(labeled_dataset)
    # 3️⃣ 查看学习到的评估标准
    print("\nLearned Rubrics from Labeled Data:\n",grader.kwargs.get("rubrics", "No rubrics generated"))
    # 4️⃣ 评估新样本
    test_cases = [
        {"query": "My order hasn't moved in 5 days, can you check? I'm a bit worried", "response": "I understand your concern! Let me check immediately: Your package is currently at XX distribution center. Due to recent high order volume, there's a slight delay, but it's expected to arrive the day after tomorrow. I'll proactively contact you if there are any issues."},
        {"query": "Why is this delivery so slow? I'm waiting to use it!", "response": "Checking, please wait."},
    ]
    print("\n" + "=" * 70, "\nEvaluation Results:\n", "=" * 70)
    for i, case in enumerate(test_cases):
        result = await grader.aevaluate(query=case["query"], response=case["response"])
        print(f"\n[Test {i+1}]\n  Query: {case['query']}\n  Response: {case['response']}\n  Score: {result.score}/5\n  Reason: {result.reason[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

> 📚 完整的快速开始内容可在 [快速开始指南](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/) 中查看。

---

## 🔗 集成

无缝连接 OpenJudge 与主流可观测性和训练平台：

| 类别 | 平台 | 状态 | 文档 |
|:---------|:---------|:------:|:--------------|
| **可观测性** | [LangSmith](https://smith.langchain.com/) | ✅ 可用 | 👉 [LangSmith 集成指南](https://agentscope-ai.github.io/OpenJudge/integrations/langsmith/) |
| | [Langfuse](https://langfuse.com/) | ✅ 可用 | 👉 [Langfuse 集成指南](https://agentscope-ai.github.io/OpenJudge/integrations/langfuse/) |
| | 其他框架 | 🔵 计划中 | — |
| **训练** | [verl](https://github.com/volcengine/verl) | ✅ 可用 | 👉 [VERL 集成指南](https://agentscope-ai.github.io/OpenJudge/integrations/verl/) |
| | [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) | 🔵 计划中 | — |

> 💬 有您希望我们优先支持的框架吗？[提交 Issue](https://github.com/agentscope-ai/OpenJudge/issues)！



---

## 🤝 贡献

我们欢迎您的贡献！我们希望让参与 OpenJudge 的贡献尽可能简单和透明。

> **🎨 添加新评分器** — 有领域特定的评估逻辑？与社区分享吧！
> **🐛 报告 Bug** — 发现问题？通过 [提交 issue](https://github.com/agentscope-ai/OpenJudge/issues) 帮助我们修复
> **📝 改进文档** — 更清晰的解释或更好的示例总是受欢迎的
> **💡 提议新功能** — 有新集成的想法？让我们讨论！

📖 查看完整的 [贡献指南](https://agentscope-ai.github.io/OpenJudge/community/contributing/) 了解编码标准和 PR 流程。

---

## 💬 社区

欢迎加入 OpenJudge 钉钉交流群，与我们一起讨论：

<div align="center">
<img src="./docs/images/dingtalk_qr_code.png" alt="钉钉群二维码" width="200">
</div>

---

## 迁移指南（v0.1.x → v0.2.0）
> OpenJudge 之前以旧包名 `rm-gallery`（v0.1.x）发布。从 v0.2.0 开始，它以 `py-openjudge` 发布，Python 导入命名空间为 `openjudge`。

**OpenJudge v0.2.0 与 v0.1.x 不向后兼容。**
如果您目前正在使用 v0.1.x，请选择以下路径之一：

- **继续使用 v0.1.x（旧版）**：继续使用旧包

```bash
pip install rm-gallery
```

我们在 [`v0.1.7-legacy` 分支](https://github.com/agentscope-ai/OpenJudge/tree/v0.1.7-legacy) 中保留了 **v0.1.7（最新的 v0.1.x 版本）** 的源代码。

- **迁移到 v0.2.0（推荐）**：按照上方的 **[安装](#-安装)** 章节操作，然后浏览 **[快速开始](#-快速开始)**（或完整的 [快速开始指南](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/)）来更新您的导入/用法。

如果您遇到迁移问题，请 [提交 issue](https://github.com/agentscope-ai/OpenJudge/issues) 并附上您的最小复现代码和当前版本。

---

## 📄 引用

如果您在研究中使用 OpenJudge，请引用：

```bibtex
@software{
  title  = {OpenJudge: A Unified Framework for Holistic Evaluation and Quality Rewards},
  author = {The OpenJudge Team},
  url    = {https://github.com/agentscope-ai/OpenJudge},
  month  = {07},
  year   = {2025}
}
```

---

<div align="center">

**由 OpenJudge 团队用 ❤️ 打造**

[⭐ 给我们 Star](https://github.com/agentscope-ai/OpenJudge) · [🐛 报告 Bug](https://github.com/agentscope-ai/OpenJudge/issues) · [💡 提议功能](https://github.com/agentscope-ai/OpenJudge/issues)

</div>

