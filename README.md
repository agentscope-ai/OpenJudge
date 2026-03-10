<div align="center">

<img src="./docs/images/logo.svg" alt="Open-Judge Logo" width="500">

<br/>

<h3>
  <em>Holistic Evaluation, Quality Rewards: Driving Application Excellence</em>
</h3>

<p>
  🌟 <em>If you find OpenJudge helpful, please give us a <b>Star</b>!</em> 🌟
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://pypi.org/project/py-openjudge/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-blue?logo=pypi)](https://pypi.org/project/py-openjudge/)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://agentscope-ai.github.io/OpenJudge/)
[![Website](https://img.shields.io/badge/website-openjudge.me-blue?logo=googlechrome&logoColor=white)](https://openjudge.me/)
[![Try Online](https://img.shields.io/badge/try%20online-free-brightgreen?logo=rocket&logoColor=white)](https://openjudge.me/app/)

[🌐 Website](https://openjudge.me/) | [🚀 Try Online](https://openjudge.me/app/) | [📖 Documentation](https://agentscope-ai.github.io/OpenJudge/) | [🤝 Contributing](https://agentscope-ai.github.io/OpenJudge/community/contributing/) | [中文](./README_zh.md)

</div>




OpenJudge is an **open-source evaluation framework** for **AI applications** (e.g., AI agents or chatbots) designed to **evaluate quality** and  drive **continuous application optimization**.

> In practice, application excellence depends on a trustworthy evaluation workflow: Collect test data → Define graders → Run evaluation at scale → Analyze weaknesses → Iterate quickly.

OpenJudge provides **ready-to-use graders** and supports generating **scenario-specific rubrics (as graders)**, making this workflow **simpler**, **more professional**, and **easy to integrate** into your workflow.
It can also convert grading results into **reward signals** to help you **fine-tune** and optimize your application.

> **🚀 Try it now!** Visit [openjudge.me/app](https://openjudge.me/app/) to use graders online — no installation required. Test built-in graders, build custom rubrics, and explore evaluation results directly in your browser.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [News](#news)
- [Online Playground](#-online-playground)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Integrations](#-integrations)
- [Contributing](#-contributing)
- [Community](#-community)
- [Citation](#-citation)

---

## News

- **2026-03-10** - 🛠️ **New Skills** - Claude authenticity verification, find skills combo, and more. 👉 [Browse Skills](https://openjudge.me/skills)

- **2026-02-12** - 📚 **Reference Hallucination Arena** - Benchmark for evaluating LLM academic reference hallucination. 👉 [Documentation](./docs/validating_graders/ref_hallucination_arena.md) | 📊 [Leaderboard](https://openjudge.me/leaderboard)

- **2026-01-27** - 🆕 **Paper Review** - Automatically review academic papers using LLM-powered evaluation. 👉 [Documentation](https://agentscope-ai.github.io/OpenJudge/applications/paper_review/)

- **2026-01-27** - 🖥️ **OpenJudge UI** - A Streamlit-based visual interface for grader testing and Auto Arena. 👉 [Try Online](https://openjudge.me/app/) | Run locally: `streamlit run ui/app.py`

---

## ✨ Key Features

### 📦 Systematic & Quality-Assured Grader Library

Access **50+ production-ready graders** featuring a comprehensive taxonomy, rigorously validated for reliable performance.

<table>
<tr>
<td width="33%" valign="top">

#### 🎯 General

**Focus:** Semantic quality, functional correctness, structural compliance

**Key Graders:**
- `Relevance` - Semantic relevance scoring
- `Similarity` - Text similarity measurement
- `Syntax Check` - Code syntax validation
- `JSON Match` - Structure compliance

</td>
<td width="33%" valign="top">

#### 🤖 Agent

**Focus:** Agent lifecycle, tool calling, memory, plan feasibility, trajectory quality

**Key Graders:**
- `Tool Selection` - Tool choice accuracy
- `Memory` - Context preservation
- `Plan` - Strategy feasibility
- `Trajectory` - Path optimization

</td>
<td width="33%" valign="top">

#### 🖼️ Multimodal

**Focus:** Image-text coherence, visual generation quality, image helpfulness

**Key Graders:**
- `Image Coherence` - Visual-text alignment
- `Text-to-Image` - Generation quality
- `Image Helpfulness` - Image contribution

</td>
</tr>
</table>

- 🌐 **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks. 👉  [Explore Supported Scenarios](https://agentscope-ai.github.io/OpenJudge/built_in_graders/overview/)
- 🔄 **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories, Memory, Reflection, and Tool Use. 👉  [Agent Lifecycle Evaluation](https://agentscope-ai.github.io/OpenJudge/built_in_graders/agent_graders/)
- ✅ **Quality Assurance:** Every grader comes with benchmark datasets and pytest integration for validation. 👉  [View Benchmark Datasets](https://huggingface.co/datasets/agentscope-ai/OpenJudge)


### 🛠️ Flexible Grader Building Methods
Choose the build method that fits your requirements:
* **Customization:** Clear requirements, but no existing grader? If you have explicit rules or logic, use our Python interfaces or Prompt templates to quickly define your own grader.  👉 [Custom Grader Development Guide](https://agentscope-ai.github.io/OpenJudge/building_graders/create_custom_graders/)
* **Zero-shot Rubrics Generation:** Not sure what criteria to use, and no labeled data yet? Just provide a task description and optional sample queries—the LLM will automatically generate evaluation rubrics for you. Ideal for rapid prototyping when you want to get started immediately. 👉 [Zero-shot Rubrics Generation Guide](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#simple-rubric-zero-shot-generation)
* **Data-driven Rubrics Generation:** Ambiguous requirements, but have few examples? Use the GraderGenerator to automatically
summarize evaluation Rubrics from your annotated data, and generate a llm-based grader. 👉 [Data-driven Rubrics Generation Guide](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#iterative-rubric-data-driven-generation)
* **Training Judge Models:** Massive data and need peak performance? Use our training pipeline to train a dedicated Judge Model. This is ideal for complex scenarios where prompt-based grading falls short.👉 [Train Judge Models](https://agentscope-ai.github.io/OpenJudge/building_graders/training_judge_models/)


### 🔌 Easy Integration

Using mainstream observability platforms like **LangSmith** or **Langfuse**? We offer seamless integration to enhance their evaluators and automated evaluation capabilities. We also provide integrations with training frameworks like **VERL** for RL training. 👉 See [Integrations](#-integrations) for details

### 🌐 Online Playground

Explore OpenJudge without writing a single line of code. Our online platform at [openjudge.me/app](https://openjudge.me/app/) lets you:
- **Test graders interactively** — select a built-in grader, input your data, and see results instantly
- **Build custom rubrics** — use the zero-shot generator to create graders from task descriptions
- **View leaderboards** — compare model performance across evaluation benchmarks at [openjudge.me/leaderboard](https://openjudge.me/leaderboard)

---

## 📥 Installation

> 💡 **Don't want to install anything?** [Try OpenJudge online](https://openjudge.me/app/) — use graders directly in your browser, no setup needed.

```bash
pip install py-openjudge
```

> 💡 More installation methods can be found in the [Quickstart Guide](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/#installation).

---

## 🚀 Quickstart
> 📚 Complete Quickstart can be found in the [Quickstart Guide](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/).

### Simple Example

A simple example to evaluate a single response:

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader

async def main():
    # 1️⃣ Create model client
    model = OpenAIChatModel(model="qwen3-32b")
    # 2️⃣ Initialize grader
    grader = RelevanceGrader(model=model)
    # 3️⃣ Prepare data
    data = {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI that enables computers to learn from data.",
    }
    # 4️⃣ Evaluate
    result = await grader.aevaluate(**data)
    print(f"Score: {result.score}")   # Score: 4
    print(f"Reason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Evaluate LLM Applications with Built-in Graders

Use multiple built-in graders to comprehensively evaluate your LLM application: 👉  [Explore All built-in graders](https://agentscope-ai.github.io/OpenJudge/built_in_graders/overview/)

> **Business Scenario:** Evaluating an e-commerce customer service agent that handles order inquiries. We assess the agent's performance across three dimensions: **relevance**, **hallucination**, and **tool selection**.



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
# Prepare your dataset
dataset = [{
    "query": "Where is my order ORD123456?",
    "response": "Your order ORD123456 has arrived at the Beijing distribution center and is expected to arrive tomorrow.",
    "context": "Order ORD123456: Arrived at Beijing distribution center, expected to arrive tomorrow.",
    "tool_definitions": TOOL_DEFINITIONS,
    "tool_calls": [{"name": "query_order", "arguments": {"order_id": "ORD123456"}}],
    # ... more test cases
}]
async def main():
    # 1️⃣ Initialize judge model
    model = OpenAIChatModel(model="qwen3-max")
    # 2️⃣ Configure multiple graders
    grader_configs = {
        "relevance": {"grader": RelevanceGrader(model=model), "mapper": {"query": "query", "response": "response"}},
        "hallucination": {"grader": HallucinationGrader(model=model), "mapper": {"query": "query", "response": "response", "context": "context"}},
        "tool_selection": {"grader": ToolSelectionGrader(model=model), "mapper": {"query": "query", "tool_definitions": "tool_definitions", "tool_calls": "tool_calls"}},
    }
    # 3️⃣ Set up aggregator for overall score
    aggregator = WeightedSumAggregator(name="overall_score", weights={"relevance": 0.3, "hallucination": 0.4, "tool_selection": 0.3})
    # 4️⃣ Run evaluation
    results = await GradingRunner(grader_configs=grader_configs, aggregators=[aggregator], max_concurrency=5).arun(dataset)
    # 5️⃣ Generate evaluation report
    overall_stats = DistributionAnalyzer().analyze(dataset, results["overall_score"])
    print(f"{'Overall Score':<20} | {overall_stats.mean:>15.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Build Custom Graders for Your Scenario

#### Zero-shot Rubric Generation

Generate a custom grader from task description without labeled data: 👉 [Zero-shot Rubrics Generation Guide](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#simple-rubric-zero-shot-generation)

**When to use:** Quick prototyping when you have no labeled data but can clearly describe your task.

```python
import asyncio
from openjudge.generator.simple_rubric import SimpleRubricsGenerator, SimpleRubricsGeneratorConfig
from openjudge.models import OpenAIChatModel

async def main():
    # 1️⃣ Configure generator
    config = SimpleRubricsGeneratorConfig(
        grader_name="customer_service_grader",
        model=OpenAIChatModel(model="qwen3-max"),
        task_description="E-commerce AI customer service primarily handles order inquiry tasks (such as logistics status and ETA) while focusing on managing customer emotions.",
        min_score=1,
        max_score=3,
    )
    # 2️⃣ Generate grader
    generator = SimpleRubricsGenerator(config)
    grader = await generator.generate(dataset=[], sample_queries=[])
    # 3️⃣ View generated rubrics
    print("Generated Rubrics:", grader.kwargs.get("rubrics"))
    # 4️⃣ Use the grader
    result = await grader.aevaluate(
        query="My order is delayed, what should I do?",
        response="I understand your concern. Let me check your order status..."
    )
    print(f"\nScore: {result.score}/3\nReason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Data-driven Rubric Generation

Learn evaluation criteria from labeled examples: 👉 [Data-driven Rubrics Generation Guide](https://agentscope-ai.github.io/OpenJudge/building_graders/generate_rubrics_as_graders/#iterative-rubric-data-driven-generation)

**When to use:** You have labeled data and need high-accuracy graders for production use, especially when evaluation criteria are implicit.

```python
import asyncio
from openjudge.generator.iterative_rubric.generator import IterativeRubricsGenerator, IterativePointwiseRubricsGeneratorConfig
from openjudge.models import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# Prepare labeled dataset (simplified example, recommend 10+ samples in practice)
labeled_dataset = [
    {"query": "My order hasn't arrived after 10 days, I want to complain!", "response": "I sincerely apologize for the delay. I completely understand your frustration! Your order was delayed due to weather conditions, but it has now resumed shipping and is expected to arrive tomorrow. I've marked it for priority delivery.", "label_score": 5},
    {"query": "Where is my package? I need it urgently!", "response": "I understand your urgency! Your package is currently out for delivery and is expected to arrive before 2 PM today. The delivery driver's contact number is 138xxxx.", "label_score": 5},
    {"query": "Why hasn't my order arrived yet? I've been waiting for days!", "response": "Your order is expected to arrive the day after tomorrow.", "label_score": 2},
    {"query": "The logistics hasn't updated in 3 days, is it lost?", "response": "Hello, your package is not lost. It's still in transit, please wait patiently.", "label_score": 3},
    # ... more labeled examples
]

async def main():
    # 1️⃣ Configure generator
    config = IterativePointwiseRubricsGeneratorConfig(
        grader_name="customer_service_grader_v2", model=OpenAIChatModel(model="qwen3-max"),
        min_score=1, max_score=5,
        enable_categorization=True, categories_number=5,  # Enable categorization, Aggregate into 5 themes
    )
    # 2️⃣ Generate grader from labeled data
    generator = IterativeRubricsGenerator(config)
    grader = await generator.generate(labeled_dataset)
    # 3️⃣ View learned rubrics
    print("\nLearned Rubrics from Labeled Data:\n",grader.kwargs.get("rubrics", "No rubrics generated"))
    # 4️⃣ Evaluate new samples
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



---

## 🔗 Integrations

Seamlessly connect OpenJudge with mainstream observability and training platforms:

| Category | Platform | Status | Documentation |
|:---------|:---------|:------:|:--------------|
| **Observability** | [LangSmith](https://smith.langchain.com/) | ✅ Available | 👉 [LangSmith Integration Guide](https://agentscope-ai.github.io/OpenJudge/integrations/langsmith/) |
| | [Langfuse](https://langfuse.com/) | ✅ Available | 👉 [Langfuse Integration Guide](https://agentscope-ai.github.io/OpenJudge/integrations/langfuse/) |
| | Other frameworks | 🔵 Planned | — |
| **Training** | [verl](https://github.com/volcengine/verl) | ✅ Available | 👉 [VERL Integration Guide](https://agentscope-ai.github.io/OpenJudge/integrations/verl/) |
| | [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) | 🔵 Planned | — |

> 💬 Have a framework you'd like us to prioritize? [Open an Issue](https://github.com/agentscope-ai/OpenJudge/issues)!

---

## 🤝 Contributing

We love your input! We want to make contributing to OpenJudge as easy and transparent as possible.

> **🎨 Adding New Graders** — Have domain-specific evaluation logic? Share it with the community!
> **🐛 Reporting Bugs** — Found a glitch? Help us fix it by [opening an issue](https://github.com/agentscope-ai/OpenJudge/issues)
> **📝 Improving Docs** — Clearer explanations or better examples are always welcome
> **💡 Proposing Features** — Have ideas for new integrations? Let's discuss!

📖 See full [Contributing Guidelines](https://agentscope-ai.github.io/OpenJudge/community/contributing/) for coding standards and PR process.

---

## 💬 Community

Join our DingTalk group to connect with the community:

<div align="center">
<img src="./docs/images/dingtalk_qr_code.png" alt="DingTalk QR Code" width="200">
</div>

---

## Migration Guide (v0.1.x → v0.2.0)
> OpenJudge was previously distributed as the legacy package `rm-gallery` (v0.1.x). Starting from v0.2.0, it is published as `py-openjudge` and the Python import namespace is `openjudge`.

**OpenJudge v0.2.0 is NOT backward compatible with v0.1.x.**
If you are currently using v0.1.x, choose one of the following paths:

- **Stay on v0.1.x (legacy)**: keep using the old package

```bash
pip install rm-gallery
```

We preserved the source code of **v0.1.7 (the latest v0.1.x release)** in the [`v0.1.7-legacy` branch](https://github.com/agentscope-ai/OpenJudge/tree/v0.1.7-legacy).

- **Migrate to v0.2.0 (recommended)**: follow the **[Installation](#-installation)** section above, then walk through **[Quickstart](#-quickstart)** (or the full [Quickstart Guide](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/)) to update your imports / usage.

If you run into migration issues, please [open an issue](https://github.com/agentscope-ai/OpenJudge/issues) with your minimal repro and current version.

---

## 📄 Citation

If you use OpenJudge in your research, please cite:

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

**Made with ❤️ by the OpenJudge Team**

[🌐 Website](https://openjudge.me/) · [🚀 Try Online](https://openjudge.me/app/) · [⭐ Star Us](https://github.com/agentscope-ai/OpenJudge) · [🐛 Report Bug](https://github.com/agentscope-ai/OpenJudge/issues) · [💡 Request Feature](https://github.com/agentscope-ai/OpenJudge/issues)

</div>