 [**English**](./README.md) | 中文
<h2 align="center">RM-Gallery: 一站式奖励模型平台</h2>

[![](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/rm-gallery/)
[![](https://img.shields.io/badge/pypi-v0.1.1.0-blue?logo=pypi)](https://pypi.org/project/rm-gallery/)
[![](https://img.shields.io/badge/license-Apache--2.0-black)](./LICENSE)
[![](https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown)]()
[![](https://img.shields.io/badge/Docs-API_Reference-blue?logo=markdown)]()
[![](https://img.shields.io/badge/Contribute-Welcome-green)]()

----
## 📢 新闻
- **[2025-06-30]** 我们现在发布 RM Gallery v0.1.0，同时在 [PyPI](https://pypi.org/simple/rm-gallery/) 上可用！
----

## 🌟 为什么选择 RM-Gallery？

RM-Gallery 是一个用于训练、构建和应用奖励模型的一站式平台。它为在任务级别和原子级别实现奖励模型提供了全面的解决方案，具有高吞吐量和容错能力。

<p align="center">
 <img src="./docs/images/framework.png" alt="Framework" width="75%">
</p>

### 训练奖励模型
- **集成的奖励模型训练流程**：提供基于强化学习的推理奖励模型训练框架，兼容主流框架（如 verl），并提供将 RM-Gallery 集成到框架中的示例。

### 构建奖励模型
- **统一的奖励模型架构**：通过标准化接口灵活实现奖励模型，支持各种架构（基于模型/无模型）、奖励格式（标量/评价）和评分模式（逐点/列表式/成对）

- **全面的奖励模型库**：为各种任务（如数学、编程、偏好对齐、智能体）提供丰富的即用型奖励模型实例，包括任务级（RMComposition）和组件级（RewardModel）。用户可以直接应用 RMComposition 进行特定任务，或通过组件级 RewardModel 组装自定义 RMComposition。

- **原则-评价-评分范式**：采用基于原则+评价+评分的推理奖励模型范式，提供最佳实践，帮助用户利用有限的偏好数据生成原则。

### 应用奖励模型

- **多种使用场景**：涵盖多个奖励模型（RM）使用场景，并提供详细的最佳实践，包括基于奖励的训练（如后训练）、基于奖励的推理（如 Best-of-N、优化）

- **高性能奖励模型服务**：利用新的 API 平台提供高吞吐量、容错的奖励模型服务，提高反馈效率。

## 📥 安装
> RM Gallery 需要 **Python 3.10** 或更高版本。

### 从源代码安装

```bash
# 从 GitHub 拉取源代码
git clone https://github.com/modelscope/rm-gallery.git

# 以可编辑模式安装包
pip install -e .
```

### 从 PyPi 安装

```bash
pip install rm-gallery
```

## 🚀 快速开始
<strong> 🚀 🚀 一行代码构建奖励模型 </strong>

```python
#使用注册表模式初始化
rm = RewardRegistry.get("你的奖励模型注册名称")(name="demo_rm")
```
有关 RM-Gallery 的完整基本用法，请参阅 [快速开始](docs/quick_start.ipynb)。

## 📚 文档
- 教程：
    - 数据
        - [数据流水线](docs/tutorial/data/pipeline.ipynb)
        - [数据标注器](docs/tutorial/data/annotation.ipynb)
        - [数据加载器](docs/tutorial/data/load.ipynb)
        - [数据处理器](docs/tutorial/data/process.ipynb)
    - 训练奖励模型
        - [训练逐点或成对奖励模型](docs/tutorial/training_rm/training_rm.md)

    - 构建奖励模型
        - [概述](docs/tutorial/building_rm/overview.ipynb)
        - [即用型奖励模型](docs/tutorial/building_rm/ready2use_rewards.md)
        - [构建自定义奖励模型](docs/tutorial/building_rm/custom_reward.ipynb)
        - [自动原则](docs/tutorial/building_rm/autoprinciple.ipynb)
        - [基准实践](docs/tutorial/building_rm/benchmark_practices.ipynb)

    - 奖励模型服务
        - [高性能奖励模型服务](docs/tutorial/rm_serving/rm_server.md)

    - 奖励模型应用
        - [后训练](docs/tutorial/rm_application/post_training.ipynb)
        - [最优 N 选择](docs/tutorial/rm_application/best_of_n.ipynb)
        - [优化](docs/tutorial/rm_application/refinement.ipynb)

## 🤝 贡献

我们始终鼓励贡献！

我们强烈建议在提交拉取请求之前在此仓库中安装预提交钩子。
这些钩子是在每次进行 git 提交时执行的小型内务处理脚本，
它们将自动处理格式化和代码检查。
```shell
pip install -e .
pre-commit install
```

更多详情请参阅我们的[贡献指南](./docs/contribution.md)。

## 📝 引用

如果在论文中使用 RM-Gallery，请引用：

```
@software{
title = {RM-Gallery: A One-Stop Reward Model Platform},
author = {The RM-Gallery Team},
url = {https://github.com/modelscope/RM-Gallery},
month = {06},
year = {2025}
}
```