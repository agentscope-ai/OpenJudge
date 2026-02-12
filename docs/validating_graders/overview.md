# Validating Graders

Ensure your graders make accurate, reliable judgments by validating them against datasets with known ground truth—just as you would test any critical system before production.

Graders are evaluation systems that judge AI outputs, but they need evaluation themselves. Without validation, you risk deploying evaluators that introduce systematic errors, favor certain response types, or fail on your specific use cases. Validation quantifies grader accuracy, identifies biases, and builds confidence that your evaluation pipeline measures what it should.

**Core Workflow:** Run your grader on test cases with known correct answers → Compare predictions against ground truth → Analyze accuracy and error patterns → Refine and re-validate until quality thresholds are met.




## How to Validate a Grader?

Validation compares your grader's judgments against known ground truth. Run your grader on a validation dataset, compare predictions with ground truth labels, compute accuracy metrics, and generate a validation report to measure evaluation quality.

### Approach 1: Benchmark Validation

Validate against public benchmarks with standardized ground truth. This approach works well when you need reproducible results, want to compare with published baselines, or need quick validation for general-purpose graders without collecting custom data. We currently support **[RewardBench2](rewardbench2.md)** for multi-domain response quality evaluation and **[Reference Hallucination Arena](ref_hallucination_arena.md)** for evaluating LLM reference recommendation accuracy, with MT-Bench and AlpacaEval coming soon.

**Example:**

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models import OpenAIChatModel
from openjudge.runner import GradingRunner

# Create grader
model = OpenAIChatModel(model="qwen3-32b")
grader = LLMGrader(name="my_grader", model=model)

# Run on benchmark
runner = GradingRunner(grader_configs={"my_grader": grader})
results = await runner.arun(rewardbench2_dataset)

# Analyze accuracy
from openjudge.analyzer import ValidationAnalyzer
analyzer = ValidationAnalyzer()
report = analyzer.analyze(dataset=rewardbench2_dataset, grader_results=results["my_grader"])

print(f"Accuracy: {report.metadata['accuracy']:.2%}")
```

### Approach 2: Custom Validation

Build validation pipelines tailored to your domain and evaluation criteria. Use this approach for domain-specific tasks (legal, medical, finance), proprietary test sets, or when you need full control over validation methodology.

**Validation Metrics:**

Choose metrics based on your evaluation task:

=== "Ranking Metrics"

    For graders that rank or select best responses:

    | Metric | When to Use | Interpretation |
    |--------|-------------|----------------|
    | **Accuracy** | Binary classification (correct/incorrect) | % of times grader selects correct answer |
    | **Top-K Accuracy** | Multiple acceptable answers | % of times correct answer is in top K predictions |
    | **Mean Reciprocal Rank (MRR)** | Ranking quality | Average reciprocal rank of correct answer |
    | **Kendall's Tau** | Full ranking correlation | Agreement between predicted and true rankings |

=== "Scoring Metrics"

    For graders that output continuous scores:

    | Metric | When to Use | Interpretation |
    |--------|-------------|----------------|
    | **Pearson Correlation** | Linear relationship | How well scores correlate with ground truth |
    | **Spearman Correlation** | Ranking correlation | Agreement in relative ordering |
    | **Mean Absolute Error (MAE)** | Score accuracy | Average distance from ground truth scores |
    | **F1 Score** | Binary threshold (pass/fail) | Balance between precision and recall |

**See cookbook for complete examples:** `tutorials/cookbooks/grader_validation/`


## Best Practices

**Data Quality:** Use diverse test cases with 100+ samples for reliable accuracy estimates. Ensure balanced categories (20+ samples each), clear ground truth from high-agreement annotations, and hold-out sets that never overlap with training data.

**Validation Design:** Design validation to mirror real-world use cases. Randomize answer positions to prevent position bias, run multiple times with different seeds for robustness, and review failed cases to identify systematic issues.

**Result Interpretation:** Remember that 70% accuracy may be excellent or poor depending on task difficulty. Compare against baselines and existing graders, analyze per-category performance to uncover hidden weaknesses, and use confidence intervals for small validation sets.

> **Tip:** For advanced validation, try k-fold cross-validation to test robustness, adversarial testing with challenging cases, and confidence calibration to verify if grader confidence correlates with actual accuracy.



## Next Steps

- [RewardBench2 Validation](rewardbench2.md) — Validate on a multi-domain benchmark
- [Reference Hallucination Arena](ref_hallucination_arena.md) — Evaluate LLM reference recommendation accuracy
- [Grader Analysis](../running_graders/grader_analysis.md) — Generate comprehensive validation reports
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build custom validation pipelines

