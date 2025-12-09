# :chart_with_upwards_trend: Generate Validation Reports

Evaluation reports are essential for understanding how well your graders perform and identifying areas for improvement. When you're working with RM-Gallery, you'll want to generate reports that give you actionable insights into both the behavior of your graders and their accuracy compared to ground truth data.

## Understanding Evaluation Reports

When validating graders, you'll work with two main categories of reports:

1. **Statistical Reports**: These help you understand the intrinsic properties of your graders, such as how they distribute scores across your dataset.
2. **Validation Reports**: These compare your grader results against known ground truth to measure accuracy and other performance metrics.

You'll find statistical reports particularly useful when you're first setting up graders and want to understand their baseline behavior. Validation reports become crucial when you need to assess the quality of your evaluations or compare different grader approaches.

## :bar_chart: Analyzing Grader Behavior with Statistical Reports

Statistical reports give you insight into how your graders behave across your dataset without needing ground truth labels. They're excellent for understanding the range and distribution of scores your graders produce.

### Distribution Analysis

One of the most fundamental statistical reports is the distribution analysis, which reveals key metrics about how scores are spread across your data:

```python
from rm_gallery.core.analyzer.statistical.distribution_analyzer import DistributionAnalyzer
from rm_gallery.core.runner.grading_runner import GradingRunner

# After running your graders
runner = GradingRunner(grader_configs=grader_configs)
results = await runner.arun(dataset)

# Generate distribution insights
analyzer = DistributionAnalyzer()
report = analyzer.analyze(dataset, results["grader_name"])

print(f"Mean score: {report.mean}")
print(f"Standard deviation: {report.stdev}")
print(f"Score range: {report.min_score} to {report.max_score}")
```

This type of analysis helps you understand whether your grader provides meaningful differentiation between samples. For example, if all scores cluster around the same value, your grader might not be sensitive enough to distinguish between high and low quality outputs.

## :white_check_mark: Measuring Accuracy with Validation Reports

While statistical reports tell you about your grader's behavior, validation reports measure its performance against known standards. These are indispensable when you need to validate new graders or compare different approaches.

### Accuracy Analysis

For classification tasks, accuracy analysis is often the starting point:

```python
from rm_gallery.core.analyzer.validation.accuracy_analyzer import AccuracyAnalyzer

# Assuming your dataset contains ground truth labels
analyzer = AccuracyAnalyzer()
accuracy_report = analyzer.analyze(
    dataset=dataset,
    grader_results=results["your_grader_name"],
    label_path="correct_label"
)

print(f"Overall accuracy: {accuracy_report.accuracy}")
```

This straightforward metric tells you what percentage of the time your grader makes correct predictions, giving you a clear baseline for performance.

### Beyond Simple Accuracy: F1 Scores

For more nuanced understanding, especially with imbalanced datasets, F1 scores provide a better picture:

```python
from rm_gallery.core.analyzer.validation.f1_score_analyzer import F1ScoreAnalyzer

analyzer = F1ScoreAnalyzer(prediction_threshold=0.5)
f1_report = analyzer.analyze(
    dataset=dataset,
    grader_results=results["grader_name"],
    label_path="label_set"
)

print(f"F1 Score: {f1_report.f1_score}")
print(f"Precision: {f1_report.precision}")
print(f"Recall: {f1_report.recall}")
```

This approach is particularly valuable when false positives and false negatives have different costs in your application.

## :microscope: Deep Dive into Specific Metrics

Sometimes you need to examine specific aspects of your grader's performance in isolation.

### Error Analysis

Understanding the types of errors your grader makes can guide improvements:

```python
from rm_gallery.core.analyzer.validation.false_positive_analyzer import FalsePositiveAnalyzer
from rm_gallery.core.analyzer.validation.false_negative_analyzer import FalseNegativeAnalyzer

# Analyze false positive rate
fp_analyzer = FalsePositiveAnalyzer()
fp_report = fp_analyzer.analyze(dataset, results["grader_name"], label_path="label")

# Analyze false negative rate
fn_analyzer = FalseNegativeAnalyzer()
fn_report = fn_analyzer.analyze(dataset, results["grader_name"], label_path="label")

print(f"False positive rate: {fp_report.false_positive_rate}")
print(f"False negative rate: {fn_report.false_negative_rate}")
```

### Consistency Checks

For LLM-based graders, consistency is crucial to ensure reliable evaluations:

```python
from rm_gallery.core.analyzer.validation.consistency_analyzer import ConsistencyAnalyzer

analyzer = ConsistencyAnalyzer()
consistency_report = analyzer.analyze(dataset, results["grader_name"])
print(f"Consistency score: {consistency_report.consistency}")
```

High consistency scores indicate your grader produces stable results when evaluating similar inputs.

## :rocket: Choosing the Right Approach for Your Needs

Different situations call for different reporting strategies:

- **Initial Setup**: Start with distribution analysis to understand baseline behavior
- **Performance Validation**: Use accuracy and F1 reports when you have ground truth data
- **Error Investigation**: Employ specific metric analyzers to diagnose problems
- **Quality Assurance**: Regularly check consistency for LLM-based graders

Consider combining multiple report types for a comprehensive view:

```python
# Create a comprehensive evaluation dashboard
comprehensive_report = {
    "behavior": {
        "distribution": distribution_analyzer.analyze(dataset, results["grader_name"]),
        "consistency": consistency_analyzer.analyze(dataset, results["grader_name"])
    },
    "performance": {
        "accuracy": accuracy_analyzer.analyze(dataset, results["grader_name"], label_path="label"),
        "f1_metrics": f1_analyzer.analyze(dataset, results["grader_name"], label_path="label")
    }
}
```

## :tada: Making Reports Actionable

The ultimate goal of validation reports is to inform decisions about your evaluation setup. When reviewing reports, ask yourself:

1. Are the scores meaningfully distributed across the range?
2. Is accuracy sufficient for your use case?
3. Are there specific error patterns you can address?
4. Is the grader behaving consistently?

Armed with these insights, you can iterate on your grader design, adjust parameters, or even create entirely new evaluation approaches.

With validation reports in hand, you're ready to move on to [running grading tasks at scale](run_grading_tasks.md) or [creating custom graders](../building_graders/create_custom_graders.md) to address any shortcomings you've identified.