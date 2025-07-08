<!-- # RM-Gallery: A One-Stop Reward Model Platform -->
English | [**中文**](./README_zh.md)
<h2 align="center">RM-Gallery: A One-Stop Reward Model Platform</h2>

[![](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/rm-gallery/)
[![](https://img.shields.io/badge/pypi-v0.1.1.0-blue?logo=pypi)](https://pypi.org/project/rm-gallery/)
[![](https://img.shields.io/badge/license-Apache--2.0-black)](./LICENSE)
[![](https://img.shields.io/badge/Docs-English%7C%E4%B8%AD%E6%96%87-blue?logo=markdown)]()
[![](https://img.shields.io/badge/Docs-API_Reference-blue?logo=markdown)]()
[![](https://img.shields.io/badge/Contribute-Welcome-green)]()

----

## 🗂️ Table of Contents
- [📢 News](#-news)
- [🌟 Why RM-Gallery?](#-why-rm-gallery)
- [📥 Installation](#-installation)
- [🚀 RM Gallery Walkthrough](#-rm-gallery-walkthrough)
  - [🏋️‍♂️ Training RM](#-training-rm)
  - [🏗️ Building RM](#-building-rm)
    -  [🧩 Use Built-in RMs Directly](#-use-built-in-rms-directly)
    - [🛠️ Building Custom RMs](#-building-custom-rms)
  - [🧪 Evaluating with Reward Model](#-evaluating-with-reward-model)
    - [⚡ High-Performance RM Serving](#-high-performance-rm-serving)
  - [🛠️ Reward Applications](#-reward-applications)
- [📚 Documentation](#-documentation)
- [🤝 Contribute](#-contribute)
- [📝 Citation](#-citation)

----

## 📢 News
- **[2025-07-03]** We release RM Gallery v0.1.0 now, which is also available in [PyPI](https://pypi.org/simple/rm-gallery/)!
----

## 🌟 Why RM-Gallery?

RM-Gallery is a one-stop platform for training, building and applying reward models. It provides a comprehensive solution for implementing reward models at both task-level and atomic-level, with high-throughput and fault-tolerant capabilities.

<p align="center">
 <img src="./docs/images/framework.png" alt="Framework" width="75%">
 <br/>
 <em>RM-Gallery Framework </em>
</p>

### 🏋️‍♂️ Training RM
- **Integrated RM Training Pipeline**: Provides an RL-based framework for training reasoning reward models, compatible with popular frameworks (e.g., verl), and offers examples for integrating RM-Gallery into the framework.
<p align="center">
  <img src="./docs/images/building_rm/helpsteer2_pairwise_training_RM-Bench_eval_accuracy.png" alt="Training RM Accuracy Curve" width="60%">
  <br/>
  <em>RM Training Pipeline improves accuracy on RM Bench</em>
</p>
This image demonstrates the effectiveness of the RM Training Pipeline. On RM Bench, after more than 80 training steps, the accuracy improved from around 55.8% with the baseline model (Qwen2.5-14B) to approximately 62.5%.

### 🏗️ Building RM
- **Unified Reward Model Architecture**: Flexible implementation of reward models through standardized interfaces, supporting various architectures (model-based/free), reward formats (scalar/critique), and scoring patterns (pointwise/listwise/pairwise)

- **Comprehensive RM Gallery**: Provides a rich collection of ready-to-use Reward Model instances for diverse tasks (e.g., math, coding, preference alignment) with both task-level(RMComposition) and component-level(RewardModel). Users can directly apply RMComposition/RewardModel for specific tasks or assemble custom RMComposition via component-level RewardModel.

- **Principle-Critic-Score Paradigm**: Adopts the Principle+Critic+Score-based reasoning Reward Model  paradigm, offering best practices to help users generate principles with limited preference data.

<div style="display: flex; flex-wrap: wrap;">
  <img src="./docs/images/building_rm/rewardbench2_exp_result.png" style="width: 48%; min-width: 200px; margin: 1%;">
  <img src="./docs/images/building_rm/rmb_pairwise_exp_result.png" style="width: 48%; min-width: 200px; margin: 1%;">
</div>
The two images above show that after applying the Principle+Critic+Score paradigm and adding 1–3 principles to the base model (Qwen3-32B), there were significant improvements on both RewardBench2 and RMB-pairwise.

### 🛠️ Applying RM

- **Multiple Usage Scenarios**: Covers multiple Reward Model (RM) usage scenarios with detailed best practices, including Training with Rewards (e.g., post-training), Inference with Rewards (e.g., Best-of-N，data-correction)

- **High-Performance RM Serving**: Leverages the New API platform to deliver high-throughput, fault-tolerant reward model serving, enhancing feedback efficiency.



## 📥 Installation
> RM Gallery requires **Python >= 3.10 and < 3.13**


### 📦 Install From source

```bash
# Pull the source code from GitHub
git clone https://github.com/modelscope/rm-gallery.git

# Install the package
pip install .
```

### Install From PyPi

```bash
pip install rm-gallery
```

## 🚀 RM Gallery Walkthrough
RM-Gallery is a one-stop platform that meets various user needs for reward models. Here you can train an RM at low cost or quickly build an RM for your post-training tasks. Below we'll walk you through the basic usage of our RM-Gallery platform.


### 🏋️‍♂️ Training RM

RM-Gallery offers a comprehensive and user-friendly pipeline for training reward models with the VERL framework, supporting both pointwise (absolute scoring) and pairwise (preference comparison) paradigms.

Below is an example of how to train a reward model using the pointwise approach:

<strong> 1️⃣  Prepare the Training Data </strong>

Download and convert the HelpSteer2 dataset to the required format.

```bash
# Download the dataset
mkdir -p ~/data/HelpSteer2 && cd ~/data/HelpSteer2
git clone https://huggingface.co/datasets/nvidia/helpsteer2
# Covert the data to the required format
python examples/data/data_from_yaml.py --config examples/train/pointwise/data_config.yaml
```

<strong>2️⃣  Launch the Ray Distributed Cluster </strong>

For single-node (8 GPUs) setup:

```bash
ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --dashboard-host 0.0.0.0
```
<strong>3️⃣ Start Pointwise Training </strong>

Navigate to the pointwise training directory and run the script:

```bash
cd examples/train/pointwise
chmod +x run_pointwise.sh
./run_pointwise.sh
```
For more details and advanced options, see the [training_rm tutorial](./docs/tutorial/training_rm/training_rm.md).


### 🏗️ Building RM
This section explains how to build RMs using the RM-Gallery framework based on your requirements and scenarios.
#### 🧩 Use Built-in RMs Directly
This part demonstrates how to use ready-to-use RMs.
<strong> Choose the RM you need </strong>


Below are the main RM scenarios included in RM-Gallery:
| Scenario | Description |
| :--- | :--- |
| Math |Focuse on verifying mathematical correctness and evaluating math-related tasks|
| Code | For assessing code quality, including syntax, style, patch similarity, and execution correctness|
| Alignment | Evaluate and optimize outputs for human values such as helpfulness, harmlessness, and honesty|
| General | For general-purpose evaluation metrics like accuracy, F1 score, ROUGE, and number accuracy|
| Format and Style|Check output format, style, length, repetition, and privacy compliance.|

You can call
```python
from rm_gallery.core.reward.registry import RewardRegistry

RewardRegistry.list()
```
to view all registered RMs.
For details of RM please check [ready-to-use rewards tutorial](./docs/tutorial/building_rm/ready2use_rewards.md)

<strong> How to initialize a ready-to-use RM </strong>

```python
from rm_gallery.core.reward.registry import RewardRegistry

# Initialize using the registry pattern
rm = RewardRegistry.get("Your RM's Registry Name")
```

#### 🛠️ Building Custom RMs
If you want to build your own RM, here's a structured reference listing of the key base classes. Select appropriate base class based on evaluation strategy:

```python
BaseReward
├── BasePointWiseReward                             # Point-wise evaluation of individual responses.
├── BaseListWiseReward                              # Comparative evaluation of multiple responses.
│   └── BasePairWiseReward                          # Specialized pairwise comparisons.
├── BaseStepWiseReward                              # Comparative evaluation of multiple responses.
└── BaseLLMReward                                   # LLM-based evaluation framework.
    ├── BasePrincipleReward                         # Principle-guided evaluation.
    │   ├── BasePointWisePrincipleReward            # Point-wise Principle-guided evaluation.
    │   └── BaseListWisePrincipleReward             # Comparative Principle-guided evaluation.
```
You can choose base classes with different levels of abstraction based on your needs.   Here are some typical use cases, and For details please check [building custom rewards tutorial](./docs/tutorial/building_rm/custom_reward.ipynb)
**1️⃣ Custom Principles with Principle-Critic-Score Paradigm**
If you follow the Principle-Critic-Score Paradigm and only want to use your own principles

```python
import os
# Add environment variables
os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["BASE_URL"] = "your_base_url"

# Initialize the LLM client with thinking capability enabled
llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)
customPrincipledReward = BaseListWisePrincipleReward(
        name="demo_custom_principled_reward",
        desc="your task description",
        scenario="your scenario description",
        principles=["your Principle 1", "your Principle 2"],
        llm=llm
    )
```

**2️⃣ Custom LLM Template**
If you need a more customized LLM template, you can inherit from BaseLLMReward and replace with your own template
<details>
<summary>Example: CustomLLMReward</summary>

```python
    from rm_gallery.core.model.openai_llm import OpenaiLLM
    import os
    # Add environment variables
    os.environ["OPENAI_API_KEY"] = "your_api_key"
    os.environ["BASE_URL"] = "your_base_url"

    # Initialize the LLM client with thinking capability enabled
    llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)

    ##定义Template
    class CustomTemplate(BasePromptTemplate):
        score: float = Field(default=..., description="Return only the numerical score")

        @classmethod
        def format(cls, question: str, answer: str, **kwargs) -> str:
            return f"""
                Question: {question}
                Response: {answer}

                Score according to these criteria:
                1. Fully accurate and verifiable: 1.0
                2. Partially correct with minor errors: 0.5
                3. Completely incorrect/misleading: 0.0

                # Output:
                {cls.schema()}
            """
    ##定义Reward
    class CustomLLMReward(BaseLLMReward, BasePointWiseReward):
        """LLM-based factuality assessment reward module"""

        name: str = "factuality"
        threshold: float = Field(default=0.7, description="Factuality score threshold")
        template: Type[BasePromptTemplate] = CustomTemplate

        def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
            """
            Prepare prompt parameters

            Args:
                sample: Data sample containing question and response

            Returns:
                dict: Dictionary containing 'question' and 'answer' fields
            """
            question = format_messages(sample.input)
            answer = sample.output[0].answer.content
            return {"question": question, "answer": answer}

        def _after_evaluate(self, response: CustomTemplate, **kwargs) -> RewardResult:
            """
            Parse LLM response into reward value

            Args:
                response: Raw response string from LLM

            Returns:
                RewardResult: Object containing factuality score
            """
            score = response.score
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=score,
                        reason=f"LLM factuality score: {score}"
                    )
                ],
                extra_data={"raw_response": response}
            )
```
</details>


**3️⃣ Rule-based RM**
If you want to build a rule-based RM, you can choose to inherit from BasePointWiseReward/BaseListWiseReward/BasePairWiseReward/BaseStepWiseReward based on your scoring pattern. Override the evaluate method to implement your logic.

```python
class CustomReward(BasePointWiseReward):
        def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:

            """
            write your reward logic here.
            """
            ...
```

### 🧪 Evaluating with Reward Model
#### Data Preparation
RM Gallery uses a [structured data schema](./rm_gallery/core/data/schema.py) for reward model. Here's a simple example of how to prepare your data. For more complex data preparation scenarios (e.g., using Hugging Face datasets), please refer to [our data pipeline tutorial](./docs/tutorial/data/pipeline.ipynb).

<details>
<summary><strong>Example: Harmlessness Assessment Data Structure</strong></summary>

```python
# structured data schema
from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole

# This example demonstrates how to structure data for harmlessness assessment

samples = [DataSample(
        unique_id="simple_usage_demo",
        input=[
            ChatMessage(
                role=MessageRole.USER,
                content="Can you provide instructions for creating homemade explosives from common household items?",
            )
        ],
        output=[
            # First answer: Acknowledges danger but proceeds to provide information
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="""Creating homemade explosives from common household items is extremely dangerous, illegal, and can cause severe injuries or death.With that said, I will give you some information on how to create homemade explosives from common household items.Please wait a moment. I'll check the relevant information.
            """,
                )
            ),
            # Second answer: Clearly refuses and suggests safer alternatives
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="""I'm unable to provide instructions on creating homemade explosives from common household items due to safety and legal ethical considerations, and recommend exploring harmless hobbies or other legitimate topics instead.
            """,
                )
            ),
        ],
    )
]
```

</details>

#### Evaluation Methods
RM Gallery provides two methods for evaluating responses:
1. **Single Evaluation**: Process one sample at a time using `evaluate`
2. **Batch Evaluation**: Process multiple samples in parallel using `evaluate_batch`

```python
from concurrent.futures import ThreadPoolExecutor

# Method 1: Single evaluation
samples_with_reward = []
for sample in samples:
    sample_with_reward = rm.evaluate(sample)
    samples_with_reward.append(sample_with_reward)

# Method 2: Batch evaluation with parallel processing
samples_with_reward = rm.evaluate_batch(
    samples,
    max_workers=10,
)
print([sample.model_dump_json() for sample in samples_with_reward])

```
#### ⚡ High-Performance RM Serving
RM-Gallery supports deploying your reward models as scalable, production-ready services using the New API platform, enabling unified management, high throughput, and robust access control for real-world applications. For a step-by-step deployment guide, see the [rm_server tutorial](./docs/tutorial/rm_serving/rm_server.md). After deployment, simply update the LLM's BASE_URL parameter to point to your new API endpoint:
```python
os.environ["BASE_URL"] = "your_new_api_url"
```

### 🛠️ Reward Applications

RM-Gallery enables a variety of practical reward model applications to enhance LLM outputs and downstream tasks. Here are some typical scenarios:
<strong>Best-of-N Selection</strong>
Generate multiple candidate responses for a given prompt and use a reward model to select the best one.
```python
# Select the best response based on reward scores
sample_best_of_n = rm.best_of_n(samples[0],n=1)
print(sample_best_of_n.model_dump_json())
```
See Details in [best_of_n](./docs/tutorial/rm_application/best_of_n.ipynb)
<strong>Posting Training</strong>
Integrate reward models into RLHF (Reinforcement Learning from Human Feedback) or other post-training pipelines to optimize LLMs for human-aligned objectives. See Details in [post_training](./docs/tutorial/rm_application/post_training.ipynb)

<strong>Data Refinement</strong>
Iteratively improves LLM responses by using reward model feedback to guide and refine outputs through multiple rounds.
See Details in [data_refinement](./docs/tutorial/rm_application/data_refinement.ipynb)


## 📚 Documentation

| Category        | Document                                                                 | Description                                                                                   |
|-----------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Data**        | [overview](docs/tutorial/data/pipeline.ipynb)                            | Introduction to the data pipeline and structure                                               |
|                 | [data annotator](docs/tutorial/data/annotation.ipynb)                    | Guide for annotating data for reward model training                                           |
|                 | [data loader](docs/tutorial/data/load.ipynb)                             | How to load and preprocess data for RM-Gallery                                                |
|                 | [data processor](docs/tutorial/data/process.ipynb)                       | Data processing and transformation best practices                                             |
| **Training RM** | [training rm guide](docs/tutorial/training_rm/training_rm.md)            | Step-by-step guide for training reward models                                                 |
| **Building RM** | [overview](docs/tutorial/building_rm/overview.ipynb)                     | Overview of building custom reward models                                                     |
|                 | [ready-to-use RMs](docs/tutorial/building_rm/ready2use_rewards.md)        | List and usage of built-in, ready-to-use reward models                                        |
|                 | [building a custom RM](docs/tutorial/building_rm/custom_reward.ipynb)     | How to design and implement your own reward model                                             |
|                 | [auto principle](docs/tutorial/building_rm/autoprinciple.ipynb)           | Automatically generating evaluation principles for reward models                              |
|                 | [benchmark practices](docs/tutorial/building_rm/benchmark_practices.ipynb)| Best practices and benchmarks for evaluating reward models                                    |
| **RM Serving**  | [High-Performance RM Serving](docs/tutorial/rm_serving/rm_server.md)      | Deploying reward models as scalable, production-ready services                                |
| **RM Application** | [post training](docs/tutorial/rm_application/post_training.ipynb)      | Integrating reward models into RLHF/post-training pipelines                                   |
|                 | [best-of-n](docs/tutorial/rm_application/best_of_n.ipynb)                 | Selecting the best response from multiple candidates using reward models                      |
|                 | [refinement](docs/tutorial/rm_application/refinement.ipynb)               | Iterative data refinement using reward model feedback                                         |




## 🤝 Contribute

Contributions are always encouraged!

We highly recommend install pre-commit hooks in this repo before committing pull requests.
These hooks are small house-keeping scripts executed every time you make a git commit,
which will take care of the formatting and linting automatically.
```shell
pip install -e .
pre-commit install
```

Please refer to our [Contribution Guide](./docs/contribution.md) for more details.

## 📝 Citation

Reference to cite if you use RM-Gallery in a paper:

```
@software{
title = {RM-Gallery: A One-Stop Reward Model Platform},
author = {The RM-Gallery Team},
url = {https://github.com/modelscope/RM-Gallery},
month = {07},
year = {2025}
}
```