# Integrating OpenJudge Reward System with VERL Training

> **TL;DR**: OpenJudge provides modular, multi-dimensional reward computation with async concurrency for VERL RL training. This guide shows you how to define complex reward functions using rule-based and LLM-as-Judge graders, achieving high-performance evaluation at scale.

---

## 🚀 Quick Decision: Is This For You?

**YES, if you:**
- Are using VERL for RL training
- Need multi-dimensional reward evaluation (not just single score)
- Want to combine rule-based + LLM-based graders
- Need async/concurrent evaluation for performance

**NO, if you:**
- Only need simple rule-based rewards (use VERL built-in)
---

## 📖 How to Read This Guide

**⏱️ New Users (30 min):**  
Section 1 → Section 2 (run example) → Section 4 (integration steps)

**🔧 Customization (1 hour):**  
Section 3 (architecture) → Section 5 (performance) → Section 6 (FAQ)

---

## 📍 Quick Navigation

| I want to... | Jump to... |
|--------------|----------|
| Understand the value | [Section 1: Why This Integration?](#1-why-this-integration) |
| Run working example | [Section 2: Quick Start](#2-quick-start-5-minutes) |
| Understand architecture | [Section 3: Architecture](#3-three-layer-architecture) |
| Integrate step-by-step | [Section 4: Integration Guide](#4-step-by-step-integration-guide) |
| Learn async performance | [Section 5: Performance Design](#5-high-performance-design) |
| Troubleshoot issues | [Section 6: FAQ](#6-faq) |


---

## 1. Why This Integration?

### The Challenge

Training an RL agent for complex tasks (like financial analysis) requires sophisticated reward functions:

❌ **Single-dimensional rewards** fail to capture task complexity  
❌ **Hand-crafted rules** are hard to maintain and extend  
❌ **LLM-as-Judge** becomes a performance bottleneck  
❌ **Multiple evaluation dimensions** are difficult to parallelize

### The Solution

**OpenJudge** + **VERL** = Modular, High-Performance Reward Computation

✅ **Modular Graders**: Compose rule-based + LLM-based evaluators  
✅ **Async Concurrency**: Evaluate multiple samples and dimensions in parallel  
✅ **Seamless Integration**: Drop-in replacement for VERL reward functions  
✅ **Production-Ready**: Battle-tested in large-scale RL training

### Architecture Overview

```
VERL Training → RewardManager → RewardFunction → Graders
(rollout)      (tokens→text)   (orchestration)   (evaluation)
```

**Three Layers:**
1. **Graders** - Individual evaluators (rule-based + LLM-based)
2. **RewardFunction** - Orchestration and aggregation logic  
3. **RewardManager** - VERL framework adapter

See [Section 3](#3-three-layer-architecture) for detailed architecture.

---

## 2. Quick Start (5 Minutes)

### Prerequisites

```bash
# Install dependencies
cd OpenJudge
pip install openjudge
```

### Minimal Example

Create a simple reward function with 3 graders (1 LLM-based + 2 rule-based):

```python
# my_reward_function.py
from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader
from openjudge.graders.agent.observation.observation_information_gain import (
    ObservationInformationGainGrader,
)
from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

from custom_scenario.reward.openjudge_reward_manager import (
    BaseOpenJudgeRewardFunction,
    OpenJudgeRewardManager,
    RewardResult,
    RewardSample,
)


class SimpleRewardFunction(BaseOpenJudgeRewardFunction):
    """Simple reward function with 3 graders."""
    
    def __init__(self, model_name="qwen3-max", max_concurrency=32):
        self.model = OpenAIChatModel(model=model_name, temperature=0.0)
        self.grader_configs = {
            # LLM-based: Comprehensive trajectory evaluation
            "trajectory": GraderConfig(
                grader=TrajectoryComprehensiveGrader(
                    model=self.model, 
                    language=LanguageEnum.EN
                ),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            # Rule-based: Detect action loops
            "action_loop": GraderConfig(
                grader=ActionLoopDetectionGrader(similarity_threshold=1.0),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            # Rule-based: Information gain
            "info_gain": GraderConfig(
                grader=ObservationInformationGainGrader(similarity_threshold=0.5),
                mapper=lambda data: {"messages": data["messages"]},
            ),
        }
        self.max_concurrency = max_concurrency
    
    async def compute_batch_scores(self, prompt_to_samples):
        """Compute scores using async GradingRunner."""
        # Convert to OpenJudge format
        datasets = []
        for prompt, samples in prompt_to_samples.items():
            dataset = [{"messages": s.messages} for s in samples]
            datasets.append(dataset)
        
        # Create runner (fresh instance to avoid event loop issues)
        runner = GradingRunner(
            grader_configs=self.grader_configs,
            max_concurrency=self.max_concurrency,
            show_progress=True,
        )
        
        # Run async evaluation
        runner_results = await runner.arun_multiple_datasets(datasets)
        
        # Parse results and create RewardResult objects
        results = []
        for dataset_idx, dataset_results in enumerate(runner_results):
            prompt = list(prompt_to_samples.keys())[dataset_idx]
            samples = prompt_to_samples[prompt]
            
            for sample_idx, sample in enumerate(samples):
                # Extract scores from each grader
                scores = {}
                for grader_name in self.grader_configs.keys():
                    grader_result = dataset_results[grader_name][sample_idx]
                    score = grader_result.score if hasattr(grader_result, "score") else 0.0
                    scores[f"{grader_name}_score"] = score
                
                # Simple average aggregation
                total_score = sum(scores.values()) / len(scores)
                
                results.append(RewardResult(
                    original_index=sample.original_index,
                    group_index=sample.group_index,
                    score=total_score,
                    reward_info=scores,
                ))
        
        return results


# Create default instance for VERL
compute_score_async = SimpleRewardFunction(
    model_name="qwen3-max",
    max_concurrency=32,
)
```

### Configure VERL Training Script

```bash
# train.sh
ray job submit --address="http://127.0.0.1:8265" \
    -- \
    python3 -m verl.trainer.main_ppo \
    custom_reward_function.path="my_reward_function.py" \
    custom_reward_function.name=compute_score_async \
    +reward_model.reward_kwargs.reward_manager_class_path=custom_scenario.reward.openjudge_reward_manager.OpenJudgeRewardManager \
    ...
```

**🎉 Congratulations!** You've integrated OpenJudge with VERL in 5 minutes.

---

## 3. Three-Layer Architecture

### Layer 1: Graders (Evaluators)

**Responsibility**: Evaluate a single dimension of quality.

**Types**:
1. **Rule-Based Graders**: Fast, deterministic, no API calls
2. **LLM-as-Judge Graders**: Flexible, semantic understanding

**Built-in Graders** (OpenJudge provides many out-of-the-box):

| Grader | Type | Purpose | Score Range |
|--------|------|---------|-------------|
| `ActionLoopDetectionGrader` | Rule | Penalize repetitive actions | 0.0-1.0 |
| `ObservationInformationGainGrader` | Rule | Reward novel information | 0.0-1.0 |
| `TrajectoryComprehensiveGrader` | LLM | Holistic quality assessment | 0.0-1.0 |

**Example: Rule-Based Grader**

```python
from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader

# Detects repeated actions in agent trajectories
grader = ActionLoopDetectionGrader(similarity_threshold=1.0)
result = await grader.aevaluate(messages=[...])
print(f"Score: {result.score}")  # 1.0 = no loops, 0.0 = many loops
```

**Example: LLM-as-Judge Grader**

```python
from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
from openjudge.models.openai_chat_model import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-max", temperature=0.0)
grader = TrajectoryComprehensiveGrader(model=model)
result = await grader.aevaluate(messages=[...])
print(f"Score: {result.score}")  # Comprehensive quality score
```

### Layer 2: RewardFunction (Business Logic)

**Responsibility**: Orchestrate multiple graders and aggregate scores.

**Key Components**:
1. **Grader Selection**: Choose which dimensions to evaluate
2. **GradingRunner Initialization**: Configure async concurrency
3. **Aggregation Strategy**: Combine scores (e.g., weighted average)

**Key Responsibilities:**
1. Define grader configurations (which dimensions to evaluate)
2. Create `GradingRunner` with concurrency control
3. Aggregate scores from multiple graders
4. Handle errors gracefully

**Core Pattern:**
```python
class MyRewardFunction(BaseOpenJudgeRewardFunction):
    async def compute_batch_scores(self, prompt_to_samples):
        # 1. Convert to OpenJudge format
        datasets = [...]
        
        # 2. Create runner (fresh instance per call - important!)
        runner = GradingRunner(
            grader_configs=self.grader_configs,
            max_concurrency=32,  # Controls concurrent API calls
        )
        
        # 3. Run async evaluation
        results = await runner.arun_multiple_datasets(datasets)
        
        # 4. Parse and aggregate
        return self._aggregate_scores(results)
```

See [Section 4.2](#-step-2-assemble-rewardfunction) for complete implementation.

### Layer 3: RewardManager (Framework Bridge)

**Responsibility**: Adapt VERL's DataProto to OpenJudge format.

**Key Operations** (`openjudge_reward_manager.py`):

1. **Token Decoding**: Convert VERL's token IDs to text
2. **Prompt Grouping**: Group responses by prompt (critical for listwise reward computation)
3. **Result Reconstruction**: Ensure results match input order
4. **Tensor Filling**: Populate VERL's reward_tensor

**Core Operations:**
```python
class OpenJudgeRewardManager:
    def __call__(self, data: DataProto):
        # 1. Decode tokens → text
        prompts, responses = self._decode_tokens(data)
        
        # 2. Create samples with metadata
        samples = self._create_samples(prompts, responses, data)
        
        # 3. Group by prompt (enables listwise reward computation)
        prompt_to_samples = self._group_by_prompt(samples)
        
        # 4. Call reward function (async evaluation)
        results = asyncio.run(self.compute_score(prompt_to_samples))
        
        # 5. Reconstruct order and fill tensor
        return self._fill_reward_tensor(results, data)
```

This layer is **already implemented** in `openjudge_reward_manager.py`.

### Why Prompt Grouping?

VERL generates N responses per prompt (`rollout.n=4`). Grouping enables:

1. **Listwise comparison** - Rank responses within the same prompt
2. **Relative scoring** - Normalize scores per group (useful for GRPO)

```python
# Without grouping: [p1_r1, p1_r2, p1_r3, p1_r4, p2_r1, ...]
# With grouping: {"p1": [r1, r2, r3, r4], "p2": [r1, r2, ...]}
```

---

## 4. Step-by-Step Integration Guide

### 🟢 Step 1: Choose or Develop Graders

#### Option A: Use Built-in Graders

OpenJudge provides many production-ready graders:

```python
# Rule-based graders (fast, no API calls)
from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader
from openjudge.graders.agent.observation.observation_information_gain import (
    ObservationInformationGainGrader,
)

# LLM-based graders (flexible, requires API)
from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
```

#### Option B: Create Custom Grader

**Example: Simple Rule-Based Grader**

```python
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
from typing import Any, Dict, List

class ResponseLengthGrader(BaseGrader):
    """Penalize responses that are too short or too long."""
    
    def __init__(self, min_length=50, max_length=500):
        super().__init__(
            name="response_length",
            mode=GraderMode.POINTWISE,
            description="Evaluate response length appropriateness",
        )
        self.min_length = min_length
        self.max_length = max_length
    
    async def aevaluate(self, messages: List[Dict[str, Any]]) -> GraderScore:
        # Extract final response
        final_message = messages[-1]
        response_text = final_message.get("content", "")
        length = len(response_text)
        
        # Scoring logic
        if length < self.min_length:
            score = length / self.min_length  # Penalize short responses
        elif length > self.max_length:
            score = max(0.0, 1.0 - (length - self.max_length) / self.max_length)
        else:
            score = 1.0  # Optimal length
        
        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Response length: {length} chars",
            metadata={"length": length},
        )
```

**Example: Custom LLM-as-Judge Grader**

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate
from openjudge.models.schema.oai.message import ChatMessage

# Define evaluation prompt
CUSTOM_PROMPT = """
Evaluate the following response based on:
1. Accuracy (0-10)
2. Completeness (0-10)
3. Clarity (0-10)

Query: {query}
Response: {response}

Output JSON:
{{"accuracy": <int>, "completeness": <int>, "clarity": <int>}}
"""

class CustomLLMGrader(LLMGrader):
    def __init__(self, model):
        template = PromptTemplate(
            messages={
                LanguageEnum.EN: [ChatMessage(role="user", content=CUSTOM_PROMPT)]
            }
        )
        super().__init__(
            name="custom_llm_grader",
            mode=GraderMode.POINTWISE,
            description="Custom LLM evaluation",
            model=model,
            template=template,
        )
```

### 🟢 Step 2: Assemble RewardFunction

Create a class that inherits from `BaseOpenJudgeRewardFunction`:

```python
from typing import Dict, List
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner
from custom_scenario.reward.openjudge_reward_manager import (
    BaseOpenJudgeRewardFunction,
    RewardResult,
    RewardSample,
)

class MyRewardFunction(BaseOpenJudgeRewardFunction):
    """Custom reward function with multiple graders."""
    
    def __init__(
        self,
        model_name: str = "qwen3-max",
        temperature: float = 0.0,
        max_concurrency: int = 32,
        grader_weights: Dict[str, float] = None,
    ):
        # Initialize LLM model for LLM-based graders
        self.model = OpenAIChatModel(
            model=model_name,
            temperature=temperature,
        )
        
        # Create grader configurations
        self.grader_configs = self.create_grader_configs(self.model)
        
        # Store parameters
        self.max_concurrency = max_concurrency
        self.grader_weights = grader_weights or {
            "trajectory": 0.4,
            "action_loop": 0.3,
            "info_gain": 0.3,
        }
        self.grader_names = list(self.grader_configs.keys())
    
    def create_grader_configs(self, model) -> Dict[str, GraderConfig]:
        """
        Define evaluation dimensions.
        
        Each GraderConfig contains:
        - grader: The grader instance
        - mapper: Function to extract required fields from data
        """
        return {
            # Dimension 1: Comprehensive trajectory quality
            "trajectory": GraderConfig(
                grader=TrajectoryComprehensiveGrader(model=model),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            
            # Dimension 2: Action loop detection (rule-based)
            "action_loop": GraderConfig(
                grader=ActionLoopDetectionGrader(similarity_threshold=1.0),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            
            # Dimension 3: Information gain (rule-based)
            "info_gain": GraderConfig(
                grader=ObservationInformationGainGrader(similarity_threshold=0.5),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            
            # Add more dimensions as needed...
        }
    
    async def compute_batch_scores(
        self,
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """
        Main evaluation logic.
        
        Args:
            prompt_to_samples: Dict mapping prompts to their response samples
                Example: {"What is AI?": [sample1, sample2, sample3, sample4]}
        
        Returns:
            List of RewardResult (order-independent, matched by original_index)
        """
        # Step 1: Convert RewardSample to OpenJudge format
        datasets = []
        for prompt, group_samples in prompt_to_samples.items():
            dataset = []
            for sample in group_samples:
                data_item = {
                    "messages": sample.messages,
                    # Add extra fields if needed by specific graders
                    "chat_date": sample.extra.get("chat_date") if sample.extra else None
                }
                dataset.append(data_item)
            datasets.append(dataset)
        
        # Step 2: Create GradingRunner (fresh instance per call)
        # This avoids asyncio event loop issues
        runner = GradingRunner(
            grader_configs=self.grader_configs,
            max_concurrency=self.max_concurrency,
            show_progress=True,
        )
        
        # Step 3: Run async evaluation
        # This is where the magic happens - concurrent execution!
        try:
            runner_results = await runner.arun_multiple_datasets(datasets)
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            # Return default scores on error
            return self._create_error_results(prompt_to_samples, error=str(e))
        
        # Step 4: Parse and aggregate results
        results = self._parse_and_aggregate(runner_results, prompt_to_samples)
        
        return results
    
    def _parse_and_aggregate(
        self,
        runner_results: List[Dict[str, List]],
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """Parse runner results and aggregate scores."""
        all_results = []
        prompt_list = list(prompt_to_samples.keys())
        
        for dataset_idx, dataset_results in enumerate(runner_results):
            prompt = prompt_list[dataset_idx]
            group_samples = prompt_to_samples[prompt]
            
            for sample_idx, sample in enumerate(group_samples):
                # Collect scores from all graders
                scores = {}
                for grader_name in self.grader_names:
                    grader_result = dataset_results[grader_name][sample_idx]
                    score = grader_result.score if hasattr(grader_result, "score") else 0.0
                    scores[f"{grader_name}_score"] = score
                
                # Aggregate using weighted average
                total_score = sum(
                    scores[f"{name}_score"] * self.grader_weights.get(name, 1.0)
                    for name in self.grader_names
                )
                
                # Normalize by sum of weights
                total_weight = sum(self.grader_weights.get(name, 1.0) for name in self.grader_names)
                final_score = total_score / total_weight if total_weight > 0 else 0.0
                
                # Create result
                result = RewardResult(
                    original_index=sample.original_index,
                    group_index=sample.group_index,
                    score=final_score,
                    reward_info=scores,
                    details={
                        grader_name: {
                            "score": dataset_results[grader_name][sample_idx].score,
                            "reason": getattr(dataset_results[grader_name][sample_idx], "reason", ""),
                        }
                        for grader_name in self.grader_names
                    },
                )
                all_results.append(result)
        
        return all_results
    
    def _create_error_results(self, prompt_to_samples, error):
        """Create default results on error."""
        results = []
        for samples in prompt_to_samples.values():
            for sample in samples:
                results.append(RewardResult(
                    original_index=sample.original_index,
                    group_index=sample.group_index,
                    score=0.0,
                    reward_info={f"{name}_score": 0.0 for name in self.grader_names},
                    details={"error": error},
                ))
        return results


# Create default instance for VERL integration
compute_score_async = MyRewardFunction(
    model_name="qwen3-max",
    temperature=0.0,
    max_concurrency=32,
    grader_weights={
        "trajectory": 0.4,
        "action_loop": 0.3,
        "info_gain": 0.3,
    },
)
```

### 🟢 Step 3: Configure RewardManager

The RewardManager is already implemented in `openjudge_reward_manager.py`. You just need to instantiate it:

```python
from custom_scenario.reward.openjudge_reward_manager import OpenJudgeRewardManager

# Create reward manager
reward_manager = OpenJudgeRewardManager(
    tokenizer=tokenizer,
    num_examine=5,  # Number of samples to print for debugging
    compute_score=compute_score_async,  # Your reward function instance
)

# The manager handles:
# 1. Token decoding (DataProto → text)
# 2. Prompt grouping (critical for listwise reward computation)
# 3. Calling your reward function
# 4. Reconstructing results (preserving order)
# 5. Filling reward_tensor
```

**Key Parameters**:
- `tokenizer`: VERL tokenizer for decoding token IDs
- `num_examine`: Number of samples to log (for debugging)
- `compute_score`: Your `BaseOpenJudgeRewardFunction` instance

### 🟢 Step 4: Integrate with VERL Training

You have two options depending on your VERL version:

#### Option A: Using Registry (Community Latest Version - Recommended)

**Step 4.1**: Register your reward manager

```python
# my_reward_manager.py
from verl.workers.reward_manager import register
from custom_scenario.reward.openjudge_reward_manager import OpenJudgeRewardManager

@register("openjudge")
class MyOpenJudgeRewardManager(OpenJudgeRewardManager):
    """Registered OpenJudge reward manager."""
    pass
```

**Step 4.2**: Configure in training script

```bash
# train.sh
python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager="openjudge" \
    custom_reward_function.path="my_reward_function.py" \
    custom_reward_function.name=compute_score_async \
    ...
```

**How it works**:
```python
# verl/workers/reward_manager/registry.py
REWARD_MANAGER_REGISTRY = {}

def register(name):
    def decorator(cls):
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls
    return decorator

def get_reward_manager_cls(name):
    return REWARD_MANAGER_REGISTRY[name]
```

#### Option B: Using Dynamic Import (Current Version)

**Step 4.1**: No code changes needed, just configure shell script

```bash

# Define paths
REWARD_FUN="${PATH_TO_DR}/reward/openjudge_reward_function.py"

# Run training with dynamic import
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="$CONFIG_PATH/runtime_env.yaml" \
    -- \
    python3 -m verl.trainer.main_ppo \
    custom_reward_function.path="$REWARD_FUN" \
    custom_reward_function.name=compute_score_async \
    +reward_model.reward_kwargs.reward_manager_class_path=custom_scenario.reward.openjudge_reward_manager.OpenJudgeRewardManager \
    ...
```

**How it works**:

The dynamic import logic is implemented in `verl/trainer/ppo/reward.py`:

```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """Load reward manager from class path."""
    
    # Check for custom class path in reward_kwargs
    reward_manager_class_path = reward_kwargs.pop("reward_manager_class_path", None)
    
    if reward_manager_class_path:
        # Split module path and class name
        if ":" in reward_manager_class_path:
            module_path, class_name = reward_manager_class_path.split(":", 1)
        else:
            parts = reward_manager_class_path.split(".")
            module_path = ".".join(parts[:-1])
            class_name = parts[-1]
        
        # Import the module and get the class
        module = importlib.import_module(module_path)
        reward_manager_cls = getattr(module, class_name)
    else:
        # Use built-in reward manager
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    
    # Get custom reward function
    compute_score = get_custom_reward_fn(config)
    
    # Instantiate reward manager
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        **reward_kwargs,
    )
```

**Configuration Flow**:

```
train.sh
  ↓ (sets parameters)
verl.trainer.main_ppo
  ↓ (calls)
verl.trainer.ppo.reward.load_reward_manager()
  ↓ (imports dynamically)
custom_scenario.reward.openjudge_reward_manager.OpenJudgeRewardManager
  ↓ (initialized with)
compute_score_async (from openjudge_reward_function.py)
```

**Step 4.2**: Custom trainer integration (optional)

If you need custom handling in the training loop, extend `RayPPOTrainer`:

```python
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

class MyRayPPOTrainer(RayPPOTrainer):
    """Custom trainer with OpenJudge reward handling."""
    
    def _compute_or_extract_reward(self, batch, reward_fn=None, return_dict=False):
        """
        Compute or extract rewards.
        
        Handles reward_extra_info["details"] properly.
        """
        if "rm_scores" in batch.batch.keys():
            # Use cached scores
            reward_tensor = batch.batch["rm_scores"]
            
            if return_dict:
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: batch.non_tensor_batch[key] 
                    for key in reward_extra_keys
                } if reward_extra_keys else {}
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                return reward_tensor, {}
        
        # Compute rewards using reward_fn
        if return_dict:
            result = reward_fn(batch, return_dict=True)
            return result
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            return reward_tensor, reward_extra_infos_dict
    
    def _dump_generations(self, messages, inputs, outputs, scores, reward_extra_infos_dict, dump_path, all_details=None):
        """
        Dump training samples with details.
        
        Extracts details from reward_extra_info["details"] if not provided separately.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        
        # Extract details from dict if not provided
        if all_details is None and "details" in reward_extra_infos_dict:
            all_details = reward_extra_infos_dict["details"]
            # Remove from dict to avoid duplication
            reward_extra_infos_dict = {
                k: v for k, v in reward_extra_infos_dict.items() 
                if k != "details"
            }
        
        # Prepare data
        n = len(inputs)
        base_data = {
            "messages": messages,
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        
        # Add reward_extra_info fields
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        
        # Add details if available
        if all_details is not None and len(all_details) == n:
            base_data["details"] = all_details
        
        # Write JSONL
        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))
        
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
```

## 5. High-Performance Design

### 5.1 Async Concurrency Mechanism

**The Challenge**: LLM-as-Judge graders are slow

**The Solution**: Massive async concurrency via `GradingRunner`

#### Architecture

```
┌────────────────────────────────────────────────────┐
│              GradingRunner                         │
│                                                    │
│  ┌─────────────────────────────────────────────┐   │
│  │  Semaphore (max_concurrency=32)             │   │
│  │                                             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │ Grader1 │  │ Grader2 │  │ Grader3 │      │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘      │   │
│  │       │            │            │           │   │
│  │       ▼            ▼            ▼           │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │    Async Task Pool                  │    │   │
│  │  │  [Task1, Task2, ..., Task_N]        │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│        OpenAI API (or other LLM providers)          │
│         [32 concurrent requests maximum]            │
└─────────────────────────────────────────────────────┘
```

#### Code Implementation

```python
# In GradingRunner (simplified)
class GradingRunner:
    def __init__(self, grader_configs, max_concurrency=32):
        self.grader_configs = grader_configs
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _evaluate_with_semaphore(self, grader, data_item):
        """Single evaluation with concurrency control."""
        async with self.semaphore:
            return await grader.aevaluate(**data_item)
    
    async def arun_multiple_datasets(self, datasets):
        """Evaluate multiple datasets concurrently."""
        all_tasks = []
        
        # Create tasks for all graders × all samples × all datasets
        for dataset in datasets:
            for data_item in dataset:
                for grader_name, grader_config in self.grader_configs.items():
                    task = self._evaluate_with_semaphore(
                        grader_config.grader,
                        grader_config.mapper(data_item)
                    )
                    all_tasks.append((grader_name, task))
        
        # Execute all tasks concurrently (respecting semaphore limit)
        results = await asyncio.gather(*[task for _, task in all_tasks])
        
        # Group results by dataset and grader
        return self._group_results(results, datasets)
```

#### Performance Example

| Approach | Execution | Time |
|----------|-----------|------|
| Sequential | 3 graders × 64 samples × 200ms | 38.4s |
| Async (32 concurrent) | 6 waves × 200ms | 1.2s |
| **Speedup** | | **32×** |

#### Critical: Event Loop Management

⚠️ **Always create fresh `GradingRunner` per training step:**

```python
# ✅ CORRECT
async def compute_batch_scores(self, prompt_to_samples):
    runner = GradingRunner(...)  # Fresh instance
    return await runner.arun_multiple_datasets(datasets)

# ❌ WRONG
self.runner = GradingRunner(...)  # In __init__
# Causes "attached to different loop" error in Ray/VERL
```

**Reason:** asyncio.Semaphore binds to event loop at creation. Ray may use different loops across training steps.

### 5.2 Prompt Grouping Optimization

**Implementation:** See [Section 3 - Why Prompt Grouping](#why-prompt-grouping) for details.

**Use Cases:**

1. **Listwise Scoring** - Give bonus to best response in group
   ```python
   scores[best_idx] += 0.1
   ```

2. **Relative Ranking** - Normalize scores per group (GRPO-compatible)
   ```python
   relative_scores = [s - mean(scores) for s in scores]
   ```

### 5.3 Batch Processing Strategy

#### DataProto Flow

```
┌─────────────────────────────────────────────┐
│    VERL DataProto (Training Batch)          │
│  ┌─────────────────────────────────────┐   │
│  │ batch["prompts"]: [B, L_prompt]     │   │  B = batch_size (e.g., 64)
│  │ batch["responses"]: [B, L_response] │   │  L = sequence length
│  │ batch["attention_mask"]: [B, L]     │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │ non_tensor_batch["messages"]: [B]   │   │  Lists of message dicts
│  │ non_tensor_batch["extra_info"]: [B] │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
              │
              ▼ (RewardManager.__call__)
┌─────────────────────────────────────────────┐
│         Token Decoding (Parallel)            │
│  prompts_str = tokenizer.batch_decode()     │
│  responses_str = tokenizer.batch_decode()   │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│      Create RewardSample Objects (B)        │
│  [RewardSample(i, prompt, response, ...)    │
│   for i in range(B)]                         │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│      Group by Prompt (G groups)              │
│  prompt_to_samples = {                       │
│    prompt1: [sample0, sample1, ...],         │
│    prompt2: [sample4, sample5, ...],         │
│  }                                           │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│    RewardFunction.compute_batch_scores()     │
│  → GradingRunner.arun_multiple_datasets()   │
│  → Async concurrent evaluation               │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│    Reconstruct Results (Order Matching)      │
│  result_map = {                              │
│    original_index: RewardResult              │
│  }                                           │
│  rewards = [result_map[i].score             │
│             for i in range(B)]               │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│       Fill reward_tensor [B, L]              │
│  reward_tensor[i, length-1] = score[i]      │
│  (Reward at last valid token position)      │
└─────────────────────────────────────────────┘
```

#### Result Reconstruction

**Challenge**: GradingRunner returns results grouped by dataset (prompt), but VERL needs results in original batch order.

**Solution**: Use `original_index` to reconstruct order

```python
def _reconstruct_results(self, result_items, data_length):
    """
    Convert List[RewardResult] to final output format.
    
    Args:
        result_items: Order-independent results from RewardFunction
        data_length: Original batch size
    
    Returns:
        (all_rewards, all_reward_infos) in original order
    """
    # Build index mapping
    result_map = {item.original_index: item for item in result_items}
    
    # Validate integrity
    assert len(result_map) == data_length, "Missing results!"
    assert set(result_map.keys()) == set(range(data_length)), "Index mismatch!"
    
    # Extract in original order
    all_rewards = [result_map[i].score for i in range(data_length)]
    
    # Collect all reward_info fields
    all_info_keys = set()
    for item in result_items:
        all_info_keys.update(item.reward_info.keys())
    
    # Fill missing keys with 0.0
    for item in result_items:
        for key in all_info_keys:
            if key not in item.reward_info:
                item.reward_info[key] = 0.0
    
    # Reconstruct reward_info dict
    all_reward_infos = {
        key: [result_map[i].reward_info[key] for i in range(data_length)]
        for key in all_info_keys
    }
    
    # Add details if available
    all_details = [result_map[i].details for i in range(data_length)]
    if any(d is not None for d in all_details):
        all_reward_infos["details"] = all_details
    
    return all_rewards, all_reward_infos
```

## 6. FAQ

### General Questions

**Q: Can I use OpenJudge without LLM-as-Judge graders?**  
A: Yes! Use only rule-based graders (ActionLoop, InformationGain) to avoid API costs.

**Q: Does this work with non-OpenAI LLM providers?**  
A: Yes, OpenJudge supports any OpenAI-compatible API (Azure, vLLM, etc.)

### Performance Questions

**Q: What's the recommended `max_concurrency` setting?**  
A: Start with 32. Adjust based on your API rate limits:
- 500 RPM → max_concurrency=32
- 3000 RPM → max_concurrency=64
- Self-hosted: Depends on GPU capacity

**Q: How much does async concurrency speed things up?**  
A: Typical speedup is 10-30×:
- Sequential: 3 graders × 64 samples × 200ms = 38.4s
- Async (32 concurrent): ~1.2s

**Q: What's the overhead of prompt grouping?**  
A: Negligible (<1ms per batch). The benefits far outweigh the cost.

### Troubleshooting

**Q: "attached to different event loop" error?**  
A: Create fresh `GradingRunner` per training step (see [Section 5.1](#critical-event-loop-management))

**Q: Getting zero scores?**  
A: Enable debug logging: `logger.add(sys.stderr, level="DEBUG")`

**Q: Results count mismatch?**  
A: Check `GraderConfig` mapper provides all required fields

**Q: Slow training?**  
A: (1) Increase `max_concurrency`, (2) Reduce redundant graders, (3) Enable `launch_reward_fn_async=True`

### Integration Questions

**Q: Can I use OpenJudge with non-VERL frameworks?**  
A: Yes! The core `RewardFunction` only depends on OpenJudge. You just need to:
1. Create your own manager to convert your framework's format
2. Call `reward_fn.compute_batch_scores(prompt_to_samples)`

**Q: Can I mix OpenJudge with other reward sources?**  
A: Yes! Combine rewards in your custom trainer:
```python
def _compute_or_extract_reward(self, batch, reward_fn):
    # OpenJudge rewards
    openjudge_rewards = reward_fn(batch)
    
    # Other rewards (e.g., task success)
    task_rewards = self.task_evaluator(batch)
    
    # Combine
    final_rewards = 0.7 * openjudge_rewards + 0.3 * task_rewards
    return final_rewards
```


## Appendix A: API Reference

### BaseOpenJudgeRewardFunction

```python
class BaseOpenJudgeRewardFunction:
    """Base class for OpenJudge reward functions."""
    
    async def compute_batch_scores(
        self,
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """
        Compute scores for grouped samples.
        
        Args:
            prompt_to_samples: Dict mapping prompts to samples
        
        Returns:
            List of RewardResult (order-independent)
        """
        raise NotImplementedError
```

### OpenJudgeRewardManager

```python
class OpenJudgeRewardManager:
    """VERL framework integration layer."""
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: BaseOpenJudgeRewardFunction,
        **kwargs
    ):
        """
        Initialize reward manager.
        
        Args:
            tokenizer: VERL tokenizer
            num_examine: Number of samples to log
            compute_score: Reward function instance
        """
        ...
    
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False
    ):
        """
        Compute rewards for DataProto batch.
        
        Args:
            data: VERL DataProto
            return_dict: Return dict with extra info
        
        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        ...
```

### RewardSample / RewardResult

```python
@dataclass
class RewardSample:
    """Sample to be scored."""
    original_index: int
    group_index: int
    prompt: str
    response: str
    messages: List[Dict]
    extra: Any

@dataclass
class RewardResult:
    """Scoring result."""
    original_index: int
    group_index: int
    score: float
    reward_info: Dict[str, float]
    details: Optional[Dict] = None
```

---

## Appendix B: Configuration Parameters

### GradingRunner Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grader_configs` | Dict | Required | Grader configurations |
| `max_concurrency` | int | 32 | Max concurrent API calls |
| `show_progress` | bool | True | Show progress bar |

### RewardFunction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "qwen3-max" | LLM model name |
| `temperature` | float | 0.0 | LLM temperature |
| `max_concurrency` | int | 32 | Async concurrency limit |
| `grader_weights` | Dict | None | Dimension weights (None = equal) |
| `language` | LanguageEnum | EN | Evaluation language |

### Shell Script Parameters

```bash
# Essential parameters
custom_reward_function.path="path/to/reward_function.py"
custom_reward_function.name=compute_score_async
+reward_model.reward_kwargs.reward_manager_class_path=path.to.OpenJudgeRewardManager
```

---

## Related Files

### Core Implementation
- `custom_scenario/reward/openjudge_reward_manager.py` - VERL framework adapter
- `custom_scenario/reward/openjudge_reward_function.py` - Business logic layer

### VERL Integration Points
- `verl/verl/trainer/ppo/reward.py` - Dynamic import logic
- `verl/verl/workers/reward_manager/registry.py` - Registry mechanism
