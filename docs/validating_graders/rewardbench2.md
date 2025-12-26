# Validate on RewardBench2

Validate your graders against RewardBench2, a comprehensive benchmark for evaluating response quality across multiple domains. RewardBench2 provides standardized test cases covering factuality, focus, safety, math, instruction following, and specialized domains.


## What is RewardBench2?

RewardBench2 [[1]](#references) is a benchmark dataset designed to evaluate reward models and LLM judges across diverse scenarios. It provides multi-domain coverage (factuality, focus, safety, math, precise instruction following) with expert-curated ground truth, tests for position and length bias, and offers a public leaderboard to compare your grader with state-of-the-art models.

The dataset includes 1,865 samples across six subsets:

| Subset | Samples | Evaluation Mode | Ground Truth |
|--------|---------|-----------------|--------------|
| **Factuality** | 475 | Four-way ranking | Best response among 4 candidates |
| **Focus** | 495 | Four-way ranking | Best response among 4 candidates |
| **Math** | 183 | Four-way ranking | Best response among 4 candidates |
| **Precise IF** | 160 | Four-way ranking | Best response among 4 candidates |
| **Safety** | 450 | Four-way ranking | Best response among 4 candidates |
| **Ties** | 102 | Absolute rating (1-10) | Multiple correct answers (1-26 per sample) |
| **Total** | **1,865** | - | - |

RewardBench2 uses two complementary evaluation approaches:

=== "Four-Way Comparison (Default)"

    **Example:**

    ```
    Query: "Explain quantum computing"

    Candidates:
    ├─ A: "Quantum computing leverages quantum mechanics..." ← Best
    ├─ B: "It's a type of advanced computing..."
    ├─ C: "Computers that use quantum physics..."
    └─ D: "I'm not sure about that."

    Task: Select the best response (A/B/C/D)
    ```

    **Description:**

    Candidates are randomly shuffled to prevent position bias. This mode tests comparative judgment ability with a binary outcome (correct/incorrect).

=== "Ties Absolute Rating"

    **Example:**

    ```
    Query: "Write a creative poem about nature"

    Candidates with Ground Truth:
    ├─ A: "The forest whispers..." → 9/10 ✓ Winner
    ├─ B: "Trees and flowers..." → 9/10 ✓ Winner (tie)
    ├─ C: "Nature is nice..." → 5/10
    └─ D: "Roses are red..." → 6/10

    Task: Rate each response (1-10), pick highest-rated
    ```

    **Description:**

    This mode allows multiple correct answers (ties) and tests absolute quality assessment, providing more nuanced evaluation than binary ranking.


## How to Validate on RewardBench2

Follow this three-step workflow to validate your grader:

<div class="workflow-single">
<div class="workflow-header">Validation Workflow</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Load Dataset</strong>

Load RewardBench2 from HuggingFace and optionally save locally for faster reuse.

???+ example "Show Code"
    ```python
    from datasets import load_dataset
    import pandas as pd

    # Load from HuggingFace
    dataset = load_dataset('allenai/reward-bench-2', split='test')

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(dataset)

    # Optional: Save locally for faster reuse
    df.to_parquet('rewardbench2_test.parquet')
    ```
</li>
<li><strong>Create Your Grader</strong>

Initialize a model and create a grader with custom prompts for evaluating responses.

???+ example "Show Code"
    ```python
    from openjudge.graders.llm_grader import LLMGrader
    from openjudge.models import OpenAIChatModel

    # Initialize model
    model = OpenAIChatModel(model="qwen3-32b")

    # Create grader with custom prompt
    grader = LLMGrader(
        name="rewardbench2_grader",
        model=model,
        system_prompt="You are an expert judge evaluating AI responses for quality, accuracy, and helpfulness.",
        response_format="Select the best response and output only: [[A]], [[B]], [[C]], or [[D]]"
    )
    ```
</li>
<li><strong>Run Validation</strong>

Execute evaluation on the dataset and analyze results to get accuracy metrics.

???+ example "Show Code"
    ```python
    from openjudge.runner import GradingRunner
    from openjudge.analyzer import ValidationAnalyzer

    # Setup runner
    runner = GradingRunner(grader_configs={"my_grader": grader})

    # Run evaluation
    results = await runner.arun(dataset)

    # Analyze results
    analyzer = ValidationAnalyzer()
    report = analyzer.analyze(
        dataset=dataset,
        grader_results=results["my_grader"]
    )

    # Print accuracy
    print(f"Overall Accuracy: {report.metadata['accuracy']:.2%}")
    print(f"Per-Subset Performance:\n{report.metadata['subset_accuracy']}")
    ```
</li>
</ol>
</div>
</div>

For rapid testing, use the cookbook script directly:

```bash
cd tutorials/cookbooks/grader_validation
python rewardbench2.py --data-path rewardbench2_test.parquet --model qwen3-32b
```

**Reference:** See complete implementation in `tutorials/cookbooks/grader_validation/rewardbench2.py` with dataset `allenai/reward-bench-2` on HuggingFace. Key classes include `RewardBench2Grader`, `RewardBench2Analyzer`, and `load_rewardbench2_data()`.



## Interpreting Results

The primary metric is overall accuracy across all subsets:

```
Overall Accuracy: 78.5%
Correct: 785/1000
```

**Interpretation:**
- **> 80%** — Excellent: Grader performs well across domains
- **70-80%** — Good: Reliable for most use cases
- **60-70%** — Fair: May need refinement for production use
- **< 60%** — Poor: Requires significant improvement

Beyond overall accuracy, examine per-subset performance to identify specific strengths and weaknesses:

```
Per-Subset Performance:
  Factuality      : 82.3% (391/475)
  Focus           : 78.8% (390/495)
  Math            : 65.0% (119/183)
  Precise IF      : 71.9% (115/160)
  Safety          : 88.4% (398/450)
  Ties            : 76.5% ( 78/102)
```

This breakdown reveals strengths in Safety (88.4%) and Factuality (82.3%), but lower Math accuracy (65.0%) suggests difficulty with mathematical reasoning. Review failed Math cases and consider adding domain-specific examples to your prompt.


## Error Analysis

Systematic error analysis helps identify patterns and guide improvements. Start by collecting failed cases and grouping them by subset:

```python
# Group errors by subset
errors_by_subset = {}
for sample, result in zip(validation_data, results):
    if result.score < 1.0:  # Incorrect prediction
        subset = sample["subset"]
        if subset not in errors_by_subset:
            errors_by_subset[subset] = []
        errors_by_subset[subset].append({
            "query": sample["query"],
            "predicted": result.metadata.get("predicted_letter"),
            "correct": result.metadata.get("correct_letter"),
            "reason": result.reason,
            "responses": sample["responses"]
        })
```

Next, review errors by subset to identify patterns:

```python
for subset, errors in errors_by_subset.items():
    print(f"\n{subset} Errors ({len(errors)}):")
    for error in errors[:3]:  # Show top 3
        print(f"  Query: {error['query'][:80]}...")
        print(f"  Predicted: {error['predicted']}, Correct: {error['correct']}")
        print(f"  Reason: {error['reason'][:100]}...")
```

**Common error types to look for:** Check if errors cluster on certain positions (position bias), compare lengths of predicted vs. correct responses (length bias), identify if errors concentrate in specific topics like advanced math (domain gaps), review grader reasoning for misunderstanding evaluation criteria (prompt issues), and detect if grader outputs aren't properly parsed (parsing failures).

Based on these patterns, take targeted action:

| Error Pattern | Root Cause | Solution |
|---------------|------------|----------|
| Favors position A/D | Position bias | Add anti-bias instructions, randomize order |
| Prefers longer responses | Length bias | Adjust prompt: "Evaluate quality, not length" |
| Weak on Math subset | Domain knowledge gap | Add few-shot math examples to prompt |
| Inconsistent format | Parsing issues | Use structured output format |
| Generic reasoning | Vague criteria | Provide explicit evaluation rubric |

To detect biases systematically, analyze prediction distributions to check for position bias:

```python
# Analyze prediction distribution
position_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
for result in results:
    predicted = result.metadata.get("predicted_letter")
    position_counts[predicted] += 1

# Should be ~25% each
for pos, count in position_counts.items():
    print(f"{pos}: {count/len(results):.1%}")
```

If biased (e.g., A: 40%, D: 15%), add anti-bias instructions or use structured output. Check length bias by comparing average lengths of predicted vs. correct responses—if grader consistently prefers longer/shorter responses, adjust your prompt.

Improve grader performance through effective prompt engineering. **For Four-Way Comparison:** Add explicit anti-bias instructions ("Ignore response length and position. Focus solely on quality."), emphasize evaluation criteria (helpfulness, accuracy, clarity, completeness), use structured output format (`[[A]]`, `[[B]]`, `[[C]]`, or `[[D]]`), and include few-shot examples if consistency is low. **For Ties Rating:** Provide a calibrated scale (1-3: Poor, 4-5: Below avg, 6-7: Good, 8-9: Excellent, 10: Outstanding), request reasoning before rating ("Explain your evaluation, then rate 1-10"), and ensure numeric rating appears on the last line for reliable parsing.

For advanced optimization, consider these techniques: Add domain-specific samples with identical format to create custom subsets. Run multiple graders on the same dataset to compare performance or build ensembles. Use iterative refinement by reviewing errors, updating prompts, and re-validating until you reach target accuracy. Split data into folds for cross-validation to test robustness across different samples.


## Next Steps

- [Refine Data Quality](../applications/data_refinement.md) — Improve model outputs using grader feedback
- [Pairwise Evaluation](../applications/select_rank.md) — Compare and rank multiple model outputs
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Refine your grader implementation


## References

[1] Malik, S., Pyatkin, V., Land, S., Morrison, J., Smith, N. A., Hajishirzi, H., & Lambert, N. (2025). RewardBench 2: Advancing Reward Model Evaluation. *arXiv preprint arXiv:2506.01937*. [https://arxiv.org/abs/2506.01937](https://arxiv.org/abs/2506.01937)

