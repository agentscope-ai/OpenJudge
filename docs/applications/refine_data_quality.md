# :rocket: Refine Data Quality: End-to-End Data Optimization Examples

Data quality is crucial for training effective AI models and conducting reliable evaluations. With RM-Gallery, you can iteratively refine data quality through automated feedback loops. This guide walks you through practical examples of data refinement processes.

## Why Data Refinement Matters

When training AI models or evaluating their performance, the quality of your data directly impacts results. Poor quality data can lead to models that produce inaccurate, biased, or unreliable outputs. Data refinement helps you:

- Clean and improve existing datasets
- Generate higher quality synthetic data
- Create feedback loops for continuous improvement
- Ensure your training data meets specific quality standards

## How Data Refinement Works in RM-Gallery

Data refinement in RM-Gallery uses **graders** to evaluate data quality and provide feedback for improvement. The process typically follows these steps:

1. Generate initial responses or data samples
2. Evaluate quality using appropriate graders
3. Provide feedback based on evaluation results
4. Generate refined versions based on feedback
5. Repeat the process for multiple iterations

Let's see how this works in practice.

## :arrows_counterclockwise: Basic Refinement Workflow

The core concept behind data refinement is iterative improvement guided by automated feedback. Here's a step-by-step example:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner

# Initialize components
model = OpenAIChatModel(model="gpt-4", api_key="your-api-key")
grader = LLMGrader(
    model=model,
    name="quality_grader",
    template="""
    Evaluate the quality of the following response to the given query.

    Query: {query}
    Response: {response}

    Consider factors like accuracy, completeness, clarity, and helpfulness.
    Provide a score from 0.0 to 1.0 and detailed feedback.

    Score: {score}
    Feedback: {reason}
    """
)

# Sample data that needs refinement
sample = {
    "query": "Explain quantum computing in simple terms",
    "response": "It's about computers that are really fast."
}

# Evaluate initial quality
runner = GradingRunner({"quality": grader})
result = await runner.arun([sample])

# Use feedback to guide refinement
feedback = result[0]["quality"]["reason"]
print(f"Feedback for improvement: {feedback}")

# In a real scenario, you'd use this feedback to generate improved versions
```

This workflow lets you systematically improve data quality by leveraging automated evaluation feedback.

## Advanced Refinement with LLMRefinement

For more sophisticated refinement workflows, RM-Gallery provides the [LLMRefinement](../../tutorials/cookbooks/data_refinement/refinement.py) class that automates the iterative refinement process.

### Setting Up the Refinement Pipeline

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from tutorials.cookbooks.data_refinement.refinement import LLMRefinement

# Initialize the language model
model = OpenAIChatModel(model="gpt-4", api_key="your-api-key")

# Create a listwise grader for ranking responses
grader = LLMGrader(
    model=model,
    name="response_ranker",
    mode="listwise",
    template="""
    You are an expert evaluator comparing multiple responses to a query.

    Query: {query}
    Responses:
    {responses}

    Rank these responses from best (1) to worst, and provide detailed feedback
    explaining your ranking decisions.

    Return your response in JSON format:
    {{
        "rank": [1, 2, 3],
        "reason": "Explanation of ranking"
    }}
    """
)

# Initialize the refiner
refiner = LLMRefinement(
    grader=grader,
    model=model,
    max_iterations=3
)
```

### Performing Iterative Refinement

```python
# Sample data for refinement
sample = {
    "history": [
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
}

# Perform refinement
refined_sample = refiner.refine(sample)

# The refined_sample contains all intermediate steps and the final output
```

## :mag: Data Quality Refinement for Training Datasets

When refining data for training purposes, you typically want to improve multiple aspects of quality:

### Accuracy Refinement

```python
from rm_gallery.core.graders.common.accuracy import AccuracyGrader
from rm_gallery.core.graders.common.hallucination import HallucinationGrader

# Dataset that needs accuracy improvements
dataset = [
    {
        "query": "Who was the first president of the United States?",
        "response": "George Washington was the first president of the United States."
    },
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Berlin."  # Incorrect
    }
]

# Accuracy grader
accuracy_grader = AccuracyGrader()
hallucination_grader = HallucinationGrader()

# Evaluate dataset
runner = GradingRunner({
    "accuracy": accuracy_grader,
    "hallucination": hallucination_grader
})

results = await runner.arun(dataset)

# Identify problematic samples
problematic_samples = []
for i, (accuracy_result, hallucination_result) in enumerate(
    zip(results["accuracy"], results["hallucination"])
):
    if accuracy_result.score < 0.5 or hallucination_result.score > 0.5:
        problematic_samples.append((i, dataset[i]))
```

### Completeness Refinement

```python
from rm_gallery.core.graders.llm_grader import LLMGrader

# Completeness grader
completeness_grader = LLMGrader(
    model=model,
    name="completeness",
    template="""
    Evaluate the completeness of the following response.

    Query: {query}
    Response: {response}

    Consider whether the response fully addresses all aspects of the query,
    provides sufficient detail, and doesn't leave out important information.

    Score from 0.0 (incomplete) to 1.0 (complete):
    {{
        "score": {score},
        "reason": "Explanation of completeness assessment"
    }}
    """
)

# Evaluate completeness
runner = GradingRunner({"completeness": completeness_grader})
completeness_results = await runner.arun(dataset)

# Find incomplete responses
incomplete_samples = []
for i, result in enumerate(completeness_results["completeness"]):
    if result.score < 0.7:  # Threshold for completeness
        incomplete_samples.append((i, dataset[i], result.reason))
```

## :wrench: Automated Data Cleaning Pipeline

You can build an automated pipeline to clean and refine entire datasets:

```python
import asyncio
from typing import List, Dict

class DataRefinementPipeline:
    def __init__(self, graders: Dict, model, refinement_threshold: float = 0.7):
        self.graders = graders
        self.model = model
        self.refinement_threshold = refinement_threshold
        self.runner = GradingRunner(graders)

    async def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """Evaluate the entire dataset with all graders."""
        return await self.runner.arun(dataset)

    def identify_issues(self, results: Dict, dataset: List[Dict]) -> List[Dict]:
        """Identify samples with quality issues."""
        issues = []
        for i, sample in enumerate(dataset):
            sample_issues = {}
            for grader_name, grader_results in results.items():
                result = grader_results[i]
                if hasattr(result, 'score') and result.score < self.refinement_threshold:
                    sample_issues[grader_name] = {
                        'score': result.score,
                        'reason': result.reason
                    }

            if sample_issues:
                issues.append({
                    'index': i,
                    'sample': sample,
                    'issues': sample_issues
                })

        return issues

    async def refine_sample(self, sample: Dict, issues: Dict) -> Dict:
        """Refine a single sample based on identified issues."""
        # Create a refinement prompt based on issues
        issue_descriptions = []
        for grader_name, issue in issues.items():
            issue_descriptions.append(f"- {grader_name}: {issue['reason']}")

        refinement_prompt = f"""
        Original query: {sample.get('query', '')}
        Original response: {sample.get('response', '')}

        Issues identified:
        {' '.join(issue_descriptions)}

        Please provide an improved response that addresses these issues:
        """

        # Generate refined response
        messages = [{"role": "user", "content": refinement_prompt}]
        response = await self.model.achat(messages=messages)

        refined_sample = sample.copy()
        refined_sample['original_response'] = sample.get('response', '')
        refined_sample['response'] = response.content
        refined_sample['refinement_notes'] = issues

        return refined_sample

    async def refine_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Refine an entire dataset."""
        # Evaluate dataset
        results = await self.evaluate_dataset(dataset)

        # Identify issues
        issues = self.identify_issues(results, dataset)

        # Create refined dataset
        refined_dataset = dataset.copy()

        # Refine problematic samples
        for issue in issues:
            index = issue['index']
            refined_sample = await self.refine_sample(
                issue['sample'],
                issue['issues']
            )
            refined_dataset[index] = refined_sample

        return refined_dataset

# Usage example
graders = {
    "accuracy": AccuracyGrader(),
    "helpfulness": HelpfulnessGrader(model=model),
    "completeness": completeness_grader
}

pipeline = DataRefinementPipeline(graders, model, refinement_threshold=0.7)
refined_dataset = await pipeline.refine_dataset(dataset)
```

## :hammer_and_pick: Domain-Specific Data Refinement

Different domains may require specialized refinement approaches:

### Code Data Refinement

```python
from rm_gallery.core.graders.code.syntax_checker import SyntaxCheckerGrader
from rm_gallery.core.graders.code.code_execution import CodeExecutionGrader

code_graders = {
    "syntax": SyntaxCheckerGrader(),
    "execution": CodeExecutionGrader()
}

code_dataset = [
    {
        "query": "Write a Python function to calculate factorial",
        "response": """
        def factorial(n):
            if n = 1:  # Syntax error
                return 1
            else:
                return n * factorial(n-1)
        """
    }
]

code_pipeline = DataRefinementPipeline(code_graders, model)
refined_code_dataset = await code_pipeline.refine_dataset(code_dataset)
```

### Mathematical Problem Solving Refinement

```python
from rm_gallery.core.graders.math.answer_verification import AnswerVerificationGrader

math_graders = {
    "answer_correctness": AnswerVerificationGrader(),
    "solution_process": LLMGrader(
        model=model,
        name="solution_process",
        template="Evaluate the correctness and clarity of the mathematical solution process..."
    )
}

math_dataset = [
    {
        "query": "Solve for x: 2x + 5 = 15",
        "response": "x = 10"  # Incorrect process, only shows answer
    }
]

math_pipeline = DataRefinementPipeline(math_graders, model)
refined_math_dataset = await math_pipeline.refine_dataset(math_dataset)
```

## :bulb: Best Practices for Data Refinement

1. **Start Simple**: Begin with basic graders and gradually add complexity
2. **Combine Multiple Graders**: Use different types of graders for comprehensive evaluation
3. **Set Appropriate Thresholds**: Balance quality with quantity in your refinement process
4. **Preserve Original Data**: Keep original samples alongside refined versions for comparison
5. **Iterative Improvement**: Run multiple rounds of refinement for better results
6. **Manual Review**: Incorporate human review for critical applications
7. **Track Improvements**: Monitor how refinement affects downstream model performance

## :tada: Conclusion

Data refinement is a powerful technique for improving dataset quality in RM-Gallery. By leveraging graders for automated evaluation and feedback, you can systematically improve the accuracy, completeness, and overall quality of your training and evaluation data.

The iterative refinement process enables continuous improvement and can significantly impact the performance of models trained on the refined data. As you implement these techniques, remember to:

- Start with simple approaches and gradually increase complexity
- Always validate that refinements actually improve data quality
- Combine automated feedback with human oversight for critical applications
- Track metrics to measure the impact of your refinement efforts

After refining your data quality, consider exploring:

- :rocket: [Building custom graders](../building_graders/create_custom_graders.md) for domain-specific refinement criteria
- :chart_with_upwards_trend: [Validating graders](../validating_graders/validation_workflow.md) to ensure your refinement process is effective
- :muscle: [Training reward models](../building_graders/train_a_grader/) using your refined data
- :gear: [Running comprehensive evaluations](../running_graders/run_grading_tasks.md) with your improved datasets