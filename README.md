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

[📖 Documentation](https://agentscope-ai.github.io/OpenJudge/) | [🤝 Contributing](https://agentscope-ai.github.io/OpenJudge/community/contributing/) | [中文](./README_zh.md)

</div>

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [News](#news)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Integrations](#-integrations)
- [Contributing](#-contributing)
- [Community](#-community)
- [Citation](#-citation)

OpenJudge is a unified framework designed to drive **LLM and Agent application excellence** through **Holistic Evaluation** and **Quality Rewards**.

> 💡 Evaluation and reward signals are the cornerstones of application excellence. **Holistic evaluation** enables the systematic analysis of shortcomings to drive rapid iteration, while **high-quality** rewards provide the essential foundation for advanced optimization and fine-tuning.

OpenJudge unifies evaluation metrics and reward signals into a single, standardized **Grader** interface, offering pre-built graders, flexible customization, and seamless framework integration.

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

Using mainstream observability platforms like **LangSmith** or **Langfuse**? We offer seamless integration to enhance their evaluators and automated evaluation capabilities. We're also building integrations with training frameworks like **verl**. 👉 See [Integrations](#-integrations) for details

----
## News

- **2025-12-26** - Released OpenJudge v0.2.0 on [PyPI](https://pypi.org/project/py-openjudge/) - **Major Update!** This release expands our core capabilities by adding robust support for diverse evaluation scenarios on top of reward construction. By unifying reward and evaluation signals, OpenJudge v0.2.0 provides a more holistic approach to optimizing application performance and excellence. → [migration-guide](#migration-guide-v01x--v020)

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - We released a new paper on learning generalizable reward criteria for robust modeling.
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - We introduced techniques to align judge feedback and improve RL stability.
- **2025-07-09** - Released OpenJudge v0.1.0 on [PyPI](https://pypi.org/project/rm-gallery/)

---

## 📥 Installation

```bash
pip install py-openjudge
```

> 💡 More installation methods can be found in the [Quickstart Guide](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/#installation).

---

## 🚀 Quickstart

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

    print(f"Score: {result.score}")   # Score: 5
    print(f"Reason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

> 📚 Complete Quickstart can be found in the [Quickstart Guide](https://agentscope-ai.github.io/OpenJudge/get_started/quickstart/).

---

## 🔗 Integrations

Seamlessly connect OpenJudge with mainstream observability and training platforms:

| Category | Platform | Status | Documentation |
|:---------|:---------|:------:|:--------------|
| **Observability** | [LangSmith](https://smith.langchain.com/) | ✅ Available | 👉 [LangSmith Integration Guide](https://agentscope-ai.github.io/OpenJudge/integrations/langsmith/) |
| | [Langfuse](https://langfuse.com/) | ✅ Available | 👉 [Langfuse Integration Guide](https://agentscope-ai.github.io/OpenJudge/integrations/langfuse/) |
| | Other frameworks | 🔵 Planned | — |
| **Training** | [verl](https://github.com/volcengine/verl) | 🟡 In Progress | — |
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

[⭐ Star Us](https://github.com/agentscope-ai/OpenJudge) · [🐛 Report Bug](https://github.com/agentscope-ai/OpenJudge/issues) · [💡 Request Feature](https://github.com/agentscope-ai/OpenJudge/issues)

</div>