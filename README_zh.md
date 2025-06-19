<!-- # RM-Gallery：一站式奖励模型平台 -->
English | [**中文**](./README-ZH.md)
<h2 align="center">RM-Gallery：一站式奖励模型平台</h2>

[![](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/memoryscope/)
[![](https://img.shields.io/badge/pypi-v0.1.1.0-blue?logo=pypi)](https://pypi.org/project/memoryscope/)
[![](https://img.shields.io/badge/license-Apache--2.0-black)](./LICENSE)
[![](https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown)](https://modelscope.github.io/MemoryScope/en/index.html#welcome-to-memoryscope-tutorial)
[![](https://img.shields.io/badge/Docs-API_Reference-blue?logo=markdown)](https://modelscope.github.io/MemoryScope/en/docs/api.html)
[![](https://img.shields.io/badge/Contribute-Welcome-green)](https://modelscope.github.io/MemoryScope/en/docs/contribution.html)

----
## 📢 新闻
- **[2025-06-XX]** 我们发布了 RM Gallery vXX，同时已在 [PyPI](https://pypi.org/simple/rm-gallery/) 上线！
----

## 🌟 为什么选择 RM-Gallery？

RM-Gallery 是一个集训练、构建和部署奖励模型于一体的平台。它为任务级和组件级的奖励模型实现提供了全面的解决方案，具备高吞吐量和容错能力。

<p align="center">
 <img src="./docs/images/framework.png" alt="Framework" width="75%">
</p>

### 奖励模型训练
- **集成奖励模型训练流程**：提供基于RL的推理奖励模型训练框架，兼容主流框架（如 verl、OpenRLHF），并提供集成 RM-Gallery 的示例。

### 奖励模型构建
- **统一的奖励模型架构**：通过标准化接口灵活实现奖励模型，支持多种架构（基于模型/无模型）、奖励格式（标量/点评）、评分方式（点对/列表对/成对）

- **丰富的奖励模型库**：为多样化任务（如数学、编程、偏好对齐、智能体）提供现成可用的奖励模型实例，支持任务级（RMComposition）和组件级（RewardModel）两种方式。用户可直接应用任务级 RMComposition，或通过组件级 RewardModel 自定义组装。

- **原则-评论-评分范式**：采用"原则+评论+评分"推理奖励模型范式，提供最佳实践，帮助用户在偏好数据有限的情况下生成原则。

### 奖励模型部署

- **多场景应用**：覆盖多种奖励模型应用场景，提供详细最佳实践，包括奖励训练（如后训练）、奖励推理（如 Best-of-N）、奖励后处理（如自我纠错）。

- **高性能奖励模型服务**：基于新一代 API 平台，提供高吞吐、容错的奖励模型服务，提升反馈效率。



## 📥 安装
> RM Gallery 需要 **Python 3.10** 或更高版本。


### 源码安装

```bash
# 从 GitHub 拉取源码
git clone https://github.com/modelscope/rm-gallery.git

# 以可编辑模式安装
cd rm-gallery
pip install -e .
```

### PyPi 安装

```bash
pip install rm-gallery
```

## 🚀 快速开始

快速上手请参考 [快速开始](docs/quick_start.ipynb)。

## 📚 文档
- 教程：
    - 数据
        - [数据管道](docs/tutorial/data/pipeline_zh.ipynb)
        - [数据标注](docs/tutorial/data/annotation_zh.ipynb)
        - [数据加载](docs/tutorial/data/load_zh.ipynb)
        - [数据处理](docs/tutorial/data/process_zh.ipynb)
    - 奖励模型训练
        - [推理奖励模型训练](docs/tutorial/training_rm/pointwise_zh.ipynb)
    - 奖励模型构建
        - [现成奖励模型](docs/tutorial/building_rm/ready2use.ipynb)
        - [自定义奖励模型构建](docs/tutorial/building_rm/customization.ipynb)
        - [自动原则生成](docs/tutorial/building_rm/customization.ipynb)
    - 奖励模型服务
        - [高性能模型服务](docs/tutorial/deploy_rm_server/tutorial.md)
    - 奖励模型部署
        - [RL 训练](docs/tutorial/rm_serving/rm_server.md)
        - [推理时扩展](docs/tutorial/deploying_rm/inference_time_scaling.ipynb)



## 🤝 贡献

欢迎任何形式的贡献！

我们强烈建议在提交 Pull Request 前安装本仓库的 pre-commit 钩子。
这些钩子会在每次 git commit 时自动执行格式化和 lint 检查。
```shell
pip install -e .
pre-commit install
```

更多细节请参考[贡献指南](./docs/contribution.md)。

## 📝 引用

如果你在论文中使用了 RM-Gallery，请引用：

```
@software{
author = {GM-Gallery},
month = {06},
title = {GM-Gallery},
url = {},
year = {2025}
}
```