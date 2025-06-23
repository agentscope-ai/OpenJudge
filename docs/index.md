<!-- # RM-Gallery: A One-Stop Reward Model Platform -->
English | [**中文**](./README_ZH.md)
<h2 align="center">RM-Gallery: A One-Stop Reward Model Platform</h2>

[![](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/memoryscope/)
[![](https://img.shields.io/badge/pypi-v0.1.1.0-blue?logo=pypi)](https://pypi.org/project/memoryscope/)
[![](https://img.shields.io/badge/license-Apache--2.0-black)](./LICENSE)
[![](https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown)](https://modelscope.github.io/MemoryScope/en/index.html#welcome-to-memoryscope-tutorial)
[![](https://img.shields.io/badge/Docs-API_Reference-blue?logo=markdown)](https://modelscope.github.io/MemoryScope/en/./api.html)
[![](https://img.shields.io/badge/Contribute-Welcome-green)](https://modelscope.github.io/MemoryScope/en/./contribution.html)

----
## 📢 News
- **[2025-06-XX]** We release RM Gallery vXX now, which is also available in [PyPI](https://pypi.org/simple/rm-gallery/)!
----

## 🌟 Why RM-Gallery?

RM-Gallery is a one-stop platform for training, building and deploying reward models. It provides a comprehensive solution for implementing reward models at both task-level and component-level, with high-throughput and fault-tolerant capabilities.

<p align="center">
 <img src="././images/framework.png" alt="Framework" width="75%">
</p>

### Training RM
- **Integrated RM Training Pipeline**: Provides an RL-based framework for training reasoning reward models, compatible with popular frameworks (e.g., verl, OpenRLHF), and offers examples for integrating RM-Gallery into the framework.

### Building RM
- **Unified Reward Model Architecture**: Flexible implementation of reward models through standardized interfaces, supporting various architectures (model-based/free), reward formats (scalar/critique), and scoring patterns (pointwise/listwise/pairwise)

- **Comprehensive RM Gallery**: Provides a rich collection of ready-to-use Reward Model instances for diverse tasks (e.g., math, coding, preference alignment, agent) with both task-level(RMComposition) and component-level(RewardModel). Users can directly apply RMComposition for specific tasks or assemble custom RMComposition via component-level RewardModel.

- **Principle-Critic-Score Paradigm**: Adopts the Principle+Critic+Score-based reasoning Reward Model  paradigm, offering best practices to help users generate principles with limited preference data.

### Deploying RM

- **Multiple Usage Scenarios**: Covers multiple Reward Model (RM) usage scenarios with detailed best practices, including Training with Rewards (e.g., post-training), Inference with Rewards (e.g., Best-of-N), and Post-Inference with Rewards (e.g., self-correction).

- **High-Performance RM Serving**: Leverages the New API platform to deliver high-throughput, fault-tolerant reward model serving, enhancing feedback efficiency.



## 📥 Installation
> RM Gallery requires **Python 3.10** or higher.


### Install From source

```bash
# Pull the source code from GitHub
git clone https://github.com/modelscope/rm-gallery.git

# Install the package in editable mode
cd rm-gallery
pip install -e .
```

### Install From PyPi

```bash
pip install rm-gallery
```

## 🚀 Quick Start
<strong> 🚀 🚀 Build RM with one-line code </strong>

```python
#Initialize using the registry pattern
rm = RewardRegistry.get("Your RM's Registry Name")(name="demo_rm")
```
For complete basic usage of RM-Gallery, please refer to [Quick Start](./quick_start.ipynb).


## 📚 Documentation
- Tutorial:
    - data
        - [data pipeline](./tutorial/data/pipeline.ipynb)
        - [data annotator](./tutorial/data/annotation.ipynb)
        - [data loader](./tutorial/data/load.ipynb)
        - [data processor](./tutorial/data/process.ipynb)
    - training rm
        - [training a reasoning reward model](./tutorial/training_rm/pointwise.ipynb)
    - building rm
        - [ready-to-use RMs](./tutorial/building_rm/ready2use.ipynb)
        - [building a custom RM](./tutorial/building_rm/customization.ipynb)
        - [auto principle](./tutorial/building_rm/customization.ipynb)
    - rm serving
        - [High-Performance Model Serving](./tutorial/deploy_rm_server/tutorial.md)
    - deploying rm
        - [RL training](./tutorial/rm_serving/rm_server.md)
        - [Inference Time Scaling](./tutorial/deploying_rm/inference_time_scaling.ipynb)




## 🤝 Contribute

Contributions are always encouraged!

We highly recommend install pre-commit hooks in this repo before committing pull requests.
These hooks are small house-keeping scripts executed every time you make a git commit,
which will take care of the formatting and linting automatically.
```shell
pip install -e .
pre-commit install
```

Please refer to our [Contribution Guide](./contribution.md) for more details.

## 📝 Citation

Reference to cite if you use RM-Gallery in a paper:

```
@software{
author = {GM-Gallery},
month = {06},
title = {GM-Gallery},
url = {},
year = {2025}
}
```