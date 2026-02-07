# Documentation Style Guide

This guide demonstrates the various styling features available in our documentation. Use this as a reference when writing docs.


## Tabbed Code Blocks

Use tabbed code blocks to show the same concept in multiple languages. This is especially useful for API examples.

### Basic Example

=== "Python"

    ```python
    from langsmith import traceable, wrappers
    from openai import OpenAI

    # Optionally wrap the OpenAI client to trace all model calls.
    oai_client = wrappers.wrap_openai(OpenAI())

    # Optionally add the 'traceable' decorator to trace the inputs/outputs of this function.
    @traceable
    def toxicity_classifier(inputs: dict) -> dict:
        instructions = (
            "Please review the user query below and determine if it contains any form of toxic content "
            "such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, "
            "and 'Not toxic' if it doesn't."
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": inputs["text"]},
        ]
        result = oai_client.chat.completions.create(
            messages=messages, model="gpt-4o-mini", temperature=0
        )
        return {"class": result.choices[0].message.content}
    ```

=== "TypeScript"

    ```typescript
    import { traceable, wrappers } from "langsmith";
    import { OpenAI } from "openai";

    // Optionally wrap the OpenAI client to trace all model calls.
    const oaiClient = wrappers.wrapOpenAI(new OpenAI());

    // Optionally add the 'traceable' decorator to trace the inputs/outputs of this function.
    const toxicityClassifier = traceable(
      async (inputs: { text: string }): Promise<{ class: string }> => {
        const instructions =
          "Please review the user query below and determine if it contains any form of toxic content " +
          "such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, " +
          "and 'Not toxic' if it doesn't.";

        const result = await oaiClient.chat.completions.create({
          messages: [
            { role: "system", content: instructions },
            { role: "user", content: inputs.text },
          ],
          model: "gpt-4o-mini",
          temperature: 0,
        });

        return { class: result.choices[0].message.content ?? "" };
      },
      { name: "toxicityClassifier" }
    );
    ```

### Syntax Guide

To create tabbed code blocks, use the following syntax:

```markdown
=== "Tab Name 1"

    ```python
    # Your Python code here (indented 4 spaces)
    print("Hello, World!")
    ```

=== "Tab Name 2"

    ```javascript
    // Your JavaScript code here (indented 4 spaces)
    console.log("Hello, World!");
    ```
```

!!! warning "Important"
    The content under each tab **must be indented by 4 spaces**. This is required for the tabs to work correctly.


## Admonitions (Call-out Boxes)

Admonitions are useful for highlighting important information.

### Note

!!! note
    This is a note. Use it to provide additional context or information.

### Tip

!!! tip
    This is a tip. Use it to share best practices or shortcuts.

### Warning

!!! warning
    This is a warning. Use it to alert users about potential issues.

### Danger

!!! danger
    This is a danger alert. Use it for critical warnings.

### Info

!!! info
    This is an info box. Use it for general information.

### Example

!!! example
    This is an example block. Use it to illustrate concepts.

### Syntax

```markdown
!!! note "Custom Title"
    Your content here. Make sure to indent with 4 spaces.

!!! warning
    Warning without a custom title.
```


## Collapsible Sections

Use collapsible sections to hide detailed content that not all readers need.

??? note "Click to expand"
    This content is hidden by default. Click the header to reveal it.

    You can include any content here, including code blocks:

    ```python
    print("Hidden code!")
    ```

???+ tip "Expanded by default"
    Use `???+` to create a collapsible section that starts expanded.

### Syntax

```markdown
??? note "Collapsed by default"
    Hidden content here.

???+ tip "Expanded by default"
    Visible content here.
```


## Code Blocks

### Basic Code Block

```python
def hello_world():
    print("Hello, World!")
```

### Code Block with Line Numbers

The configuration already enables line numbers via `anchor_linenums: true`.

### Code Block with Title

```python title="example.py"
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

### Inline Code

Use backticks for inline code: `print("Hello")` or `const x = 1`.


## Tables

| Grader Type | Description | Use Case |
|-------------|-------------|----------|
| `TextSimilarity` | Compares text similarity | Answer matching |
| `LLMGrader` | Uses LLM for evaluation | Complex assessments |
| `FunctionGrader` | Custom function-based | Specific logic |


## Math Equations

Inline math: \( E = mc^2 \)

Block math:

\[
f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi) e^{2\pi i \xi x} d\xi
\]


## Links and References

- [External Link](https://example.com)
- [Internal Link to Overview](../built_in_graders/overview.md)
- Footnotes[^1]

[^1]: This is a footnote. Use them sparingly.


## Lists

### Unordered List

- Item 1
- Item 2
    - Nested item 2.1
    - Nested item 2.2
- Item 3

### Ordered List

1. First step
2. Second step
3. Third step

### Task List

- [x] Completed task
- [ ] Pending task
- [ ] Another pending task


## Workflow Steps

Use workflow components to display step-by-step processes with numbered badges and visual indicators.

### Basic Workflow (Single Tab)

<div class="workflow-single">
<div class="workflow-header">Evaluation flow</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Create a dataset</strong>

Create a [dataset](../built_in_graders/overview.md) with [examples](../built_in_graders/overview.md) from manually curated test cases, historical production traces, or synthetic data generation.</li>
<li><strong>Define evaluators</strong>

Create [evaluators](../built_in_graders/overview.md) to score performance:

- [Human](../built_in_graders/overview.md) review
- [Code](../built_in_graders/overview.md) rules
- [LLM-as-judge](../built_in_graders/overview.md)
- [Pairwise](../built_in_graders/overview.md) comparison</li>
<li><strong>Run an experiment</strong>

Execute your application on the dataset to create an [experiment](../built_in_graders/overview.md). Configure [repetitions, concurrency, and caching](../built_in_graders/overview.md) to optimize runs.</li>
<li><strong>Analyze results</strong>

Compare experiments for [benchmarking](../built_in_graders/overview.md), [unit tests](../built_in_graders/overview.md), [regression tests](../built_in_graders/overview.md), or [backtesting](../built_in_graders/overview.md).</li>
</ol>
</div>
</div>

### Tabbed Workflow

Use tabs when you have multiple related workflows to display:

=== "Offline evaluation flow"

    <div class="workflow">
    <ol class="workflow-steps">
    <li><strong>Create a dataset</strong>

    Create a [dataset](../built_in_graders/overview.md) with [examples](../built_in_graders/overview.md) from manually curated test cases, historical production traces, or synthetic data generation.</li>
    <li><strong>Define evaluators</strong>

    Create [evaluators](../built_in_graders/overview.md) to score performance:

    - [Human](../built_in_graders/overview.md) review
    - [Code](../built_in_graders/overview.md) rules
    - [LLM-as-judge](../built_in_graders/overview.md)
    - [Pairwise](../built_in_graders/overview.md) comparison</li>
    <li><strong>Run an experiment</strong>

    Execute your application on the dataset to create an [experiment](../built_in_graders/overview.md). Configure [repetitions, concurrency, and caching](../built_in_graders/overview.md) to optimize runs.</li>
    <li><strong>Analyze results</strong>

    Compare experiments for [benchmarking](../built_in_graders/overview.md), [unit tests](../built_in_graders/overview.md), [regression tests](../built_in_graders/overview.md), or [backtesting](../built_in_graders/overview.md).</li>
    </ol>
    </div>

=== "Online evaluation flow"

    <div class="workflow">
    <ol class="workflow-steps">
    <li><strong>Configure tracing</strong>

    Set up [tracing](../built_in_graders/overview.md) to capture production data in real-time.</li>
    <li><strong>Add online evaluators</strong>

    Attach [evaluators](../built_in_graders/overview.md) that run automatically on production traces:

    - [Latency](../built_in_graders/overview.md) monitoring
    - [Error rate](../built_in_graders/overview.md) tracking
    - [Quality](../built_in_graders/overview.md) scoring</li>
    <li><strong>Monitor dashboards</strong>

    View real-time [dashboards](../built_in_graders/overview.md) to track model performance and detect regressions.</li>
    <li><strong>Set up alerts</strong>

    Configure [alerts](../built_in_graders/overview.md) to notify you when metrics fall below thresholds.</li>
    </ol>
    </div>

### Workflow Syntax Guide

**Single workflow (no tabs):**

```markdown
<div class="workflow-single">
<div class="workflow-header">Workflow Title</div>

<div class="workflow">
<ol>
<li><strong>Step Title</strong>

Step description with [links](url) and details.</li>
<li><strong>Another Step</strong>

Description with nested list:

- Item 1
- Item 2</li>
</ol>
</div>
</div>
```

**Tabbed workflows:**

```markdown
=== "Tab 1 Name"

    <div class="workflow">
    <ol class="workflow-steps">
    <li><strong>Step Title</strong>

    Description here.</li>
    </ol>
    </div>

=== "Tab 2 Name"

    <div class="workflow">
    <ol class="workflow-steps">
    <li><strong>Step Title</strong>

    Description here.</li>
    </ol>
    </div>
```

!!! tip "Formatting Tips"
    - Each `<li>` should start with `<strong>Title</strong>` followed by a blank line
    - Descriptions can include links, nested lists, and formatted text
    - Keep step titles concise but descriptive
    - The workflow container must use `<div class="workflow">` wrapper


## Combining Elements

Here's an example combining multiple elements:

=== "Installation"

    !!! tip "Prerequisites"
        Make sure you have Python 3.8+ installed.

    ```bash
    pip install py-openjudge
    ```

=== "Verification"

    ```python
    import openjudge
    print(openjudge.__version__)
    ```

    !!! success "Expected Output"
        You should see the version number printed.


## Typography Elements

### Blockquotes

Use blockquotes for important quotes or highlighted text:

> This is a simple blockquote. It's great for highlighting important information or quoting external sources.

With citation:

> The best way to predict the future is to invent it.
>
> — Alan Kay

### Keyboard Shortcuts

Display keyboard shortcuts using the `<kbd>` tag:

Press <kbd>Ctrl</kbd> + <kbd>C</kbd> to copy, and <kbd>Ctrl</kbd> + <kbd>V</kbd> to paste.

On Mac, use <kbd>⌘</kbd> + <kbd>S</kbd> to save.

### Highlighted Text

Use `<mark>` to highlight important text:

The most <mark>critical configuration</mark> is the API key setting.

### Abbreviations

Define abbreviations that show tooltips on hover:

The HTML specification is maintained by the W3C.

*[HTML]: HyperText Markup Language
*[W3C]: World Wide Web Consortium


## Enhanced Admonitions

### Admonitions with Rich Content

Admonitions can contain complex nested content:

!!! note "Using Code in Admonitions"
    You can include inline code like `grader.evaluate()` and code blocks:

    ```python
    from openjudge import LLMGrader

    grader = LLMGrader(model="gpt-4")
    result = grader.evaluate(response, reference)
    ```

    Links also work: see the [Graders Overview](../built_in_graders/overview.md).

!!! tip "Lists in Admonitions"
    Admonitions support all list types:

    **Unordered:**
    - First item
    - Second item with `code`
    - Third item with [link](https://example.com)

    **Ordered:**
    1. Step one
    2. Step two
    3. Step three

!!! warning "Nested Admonitions"
    You can even nest admonitions for complex scenarios:

    !!! danger "Critical Warning"
        This is a nested danger alert inside a warning.

### Custom Titled Admonitions

!!! info "API Rate Limits"
    The default rate limit is **100 requests per minute**. Contact support to increase limits.

!!! example "Complete Example"
    Here's a full working example:

    ```python
    from openjudge import TextSimilarityGrader

    grader = TextSimilarityGrader(threshold=0.8)

    result = grader.evaluate(
        response="The capital of France is Paris.",
        reference="Paris is the capital of France."
    )

    print(f"Score: {result.score}")  # Score: 0.95
    print(f"Pass: {result.passed}")  # Pass: True
    ```


## Links Styling

### Standard Links

- [Internal documentation link](../built_in_graders/overview.md)
- [External link to GitHub](https://github.com/agentscope-ai/OpenJudge)

### Links with Code

Check the [`LLMGrader`](../built_in_graders/overview.md) class for more details.

Use the [`evaluate()`](../built_in_graders/overview.md) method to run grading.


## Horizontal Rules

Standard divider (three dashes):


Use dividers to separate major sections of your documentation.


## Enhanced Tables

### Standard Table

| Feature | Status | Description |
|---------|--------|-------------|
| `LLMGrader` | ✅ Stable | LLM-Based evaluation |
| `TextSimilarity` | ✅ Stable | Text comparison |
| `FunctionGrader` | 🔶 Beta | Custom functions |

### Table with Code and Links

| Grader | Import | Documentation |
|--------|--------|---------------|
| `LLMGrader` | `from openjudge import LLMGrader` | [View docs](../built_in_graders/overview.md) |
| `TextSimilarityGrader` | `from openjudge import TextSimilarityGrader` | [View docs](../built_in_graders/overview.md) |
| `FunctionGrader` | `from openjudge import FunctionGrader` | [View docs](../built_in_graders/overview.md) |


## Footnotes

Here's a sentence with a footnote[^1], and another one[^2].

[^1]: This is the first footnote with additional details.
[^2]: This is the second footnote. You can include `code` and [links](https://example.com) in footnotes.


## Next Steps

- [Contribute to OpenJudge](contributing.md) — Learn how to contribute code and documentation
- [Quick Start](../get_started/quickstart.md) — Get started with OpenJudge
- [Built-in Graders](../built_in_graders/overview.md) — Explore available graders

