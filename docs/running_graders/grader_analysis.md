
# Grader Results Analysis
After successfully [running grading tasks](run_tasks.md) on your dataset, the next crucial step is to analyze the grader results to understand how well your AI models or agents are performing. Grader results analysis helps you gain insights from the evaluation results and generate comprehensive reports about model or agent performance.

## Why Grader Results Analysis Matters
Grader results analysis is essential because it transforms raw evaluation scores into actionable insights about your AI models or agents. While running graders provides individual scores or rankings, analysis provides critical context to understand what these results mean for your model's real-world performance.

Through the analysis, you can:
- Identify model strengths and weaknesses across different evaluation dimensions
- Measure overall performance consistency and reliability
- Detect potential biases or blind spots in model behavior
- Generate data-driven insights for model improvement

More importantly, the analysis enables iterative optimization of the graders themselves. By analyzing the results, you can:
- Refine grader criteria based on observed performance patterns
- Adjust evaluation thresholds or parameters for better discrimination
- Compare different graders' effectiveness and select the most appropriate one for your use case


## How to Do Grader Results Analysis
Grader results analysis approaches can be broadly categorized into two types based on data availability: statistical analysis without ground truth and comparative analysis with ground truth. Each approach offers unique insights into your model's performance characteristics.

### Statistical Analysis Without Ground Truth
In many cases, you won't have reference labels to compare against. In these scenarios, statistical analysis helps you understand your model's behavior patterns. This approach focuses on examining the intrinsic properties of the scores produced by your graders, revealing patterns that might indicate strengths or weaknesses in model performance.


#### Statistical Analysis Example
Here's an example to analyzing your model's performance distribution:

```python
from openjudge.analyzer.statistical.distribution_analyzer import DistributionAnalyzer
from openjudge.runner.grading_runner import GradingRunner
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel

# Prepare dataset
dataset = [
    {"query": "What is AI?", "response": "AI is artificial intelligence."},
    {"query": "What is ML?", "response": "ML is machine learning."},
    {"query": "What is DL?", "response": "DL is deep learning."}
]

# Configure grader
grader_configs = {
    "correctness": {
        "grader": CorrectnessGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {"query": "query", "response": "response"}
    }
}

# Run graders on the dataset (as described in run_tasks.md)
runner = GradingRunner(grader_configs=grader_configs)
results = await runner.arun(dataset)

# Analyze score distribution to understand model performance
analyzer = DistributionAnalyzer()
report = analyzer.analyze(dataset, results["correctness"])

print(f"Mean score: {report.mean}")
print(f"Standard deviation: {report.stdev}")
print(f"Score range: {report.min_score} to {report.max_score}")
```

This distribution analysis helps you understand how your model performs across different inputs. If all scores cluster closely together, it might indicate that your model has limited variability in its responses. On the other hand, if scores are spread widely, it might indicate varying performance on different inputs, which could reveal where your model excels or struggles.

#### Built-in Statistical Analysis
OpenJudge provides several built-in statistical analysis for examining model performance without ground truth:

| Analysis Name | Functionality |
|---------------|---------------|
| [DistributionAnalyzer](../../openjudge/analyzer/statistical/distribution_analyzer.py#L18-L60) | Examines the distribution of scores across the dataset, including mean, standard deviation, min, and max values to understand the range and variability of model performance |
| [ConsistencyAnalyzer](../../openjudge/analyzer/validation/consistency_analyzer.py#L23-L84) | Evaluates how consistently your model performs when presented with similar inputs or when the same input is evaluated multiple times |


### Comparative Analysis With Ground Truth
When you have reference labels, you can perform more comprehensive analysis by comparing model performance against known standards. This approach provides direct measurements of how well your model aligns with ground truth or expert judgment and enables calculation of standard performance metrics like precision, recall, and F1 scores.

Comparative analysis is particularly powerful because it gives you concrete measures of how well your model's outputs align with human judgment or other authoritative sources of quality assessment.

#### Comparative Analysis Example
Here's an example to comparing your model's performance with ground truth labels:

```python
from openjudge.analyzer.validation.accuracy_analyzer import AccuracyAnalyzer
from openjudge.runner.grading_runner import GradingRunner
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel

# Dataset with ground truth labels for comparison
dataset = [
    {"query": "What is AI?", "response": "AI is artificial intelligence.", "correct_label": 1},
    {"query": "What is ML?", "response": "ML is machine learning.", "correct_label": 1},
    {"query": "What is DL?", "response": "Wrong answer", "correct_label": 0}
]

# Configure and run grader
grader_configs = {
    "correctness": {
        "grader": CorrectnessGrader(model=OpenAIChatModel("qwen3-32b")),
        "mapper": {"query": "query", "response": "response"}
    }
}

runner = GradingRunner(grader_configs=grader_configs)
results = await runner.arun(dataset)

# Analyze accuracy
analyzer = AccuracyAnalyzer()
accuracy_report = analyzer.analyze(
    dataset=dataset,
    grader_results=results["correctness"],
    label_path="correct_label"  # Path to ground truth in your data
)

print(f"Overall accuracy: {accuracy_report.accuracy}")
```

This comparative analysis tells you the percentage of times your model's evaluation matched the ground truth, providing a baseline performance measure. While accuracy alone doesn't tell the whole story, it serves as a foundational metric that helps you understand the general alignment of your model with reference standards.

#### Built-in Comparative Analysis
OpenJudge provides several built-in comparative analysis for examining model performance with ground truth:

| Analysis Name | Functionality |
|---------------|---------------|
| [AccuracyAnalyzer](../../openjudge/analyzer/validation/accuracy_analyzer.py#L15-L60) | Measures the accuracy of your model's evaluation when ground truth labels are available for comparison |
| [F1ScoreAnalyzer](../../openjudge/analyzer/validation/f1_score_analyzer.py#L17-L74) | Calculates F1 scores balancing precision and recall for comprehensive evaluation, particularly useful for imbalanced datasets |
| [FalsePositiveAnalyzer](../../openjudge/analyzer/validation/false_positive_analyzer.py#L15-L50) | Identifies instances where the model incorrectly identifies positive cases, helping to understand over-estimation patterns |
| [FalseNegativeAnalyzer](../../openjudge/analyzer/validation/false_negative_analyzer.py#L15-L50) | Identifies instances where the model fails to detect actual positive cases, helping to understand under-estimation patterns |
| [PrecisionAnalyzer](../../openjudge/analyzer/validation/precision_analyzer.py#L15-L56) | Calculates precision of the model's positive predictions compared to actual positive cases |
| [RecallAnalyzer](../../openjudge/analyzer/validation/recall_analyzer.py#L15-L56) | Calculates recall of the model's ability to identify all actual positive cases |
| [CorrelationAnalyzer](../../openjudge/analyzer/validation/correlation_analyzer.py#L16-L60) | Evaluates the correlation between different metrics or evaluation criteria to understand relationships in model performance |

## Next Steps
After analyzing your grader results, you can confidently [refine data quality](../applications/data_refinement.md) based on the insights gained from your analysis or [create custom graders](../building_graders/create_custom_graders.md) to address specific evaluation needs identified during analysis.