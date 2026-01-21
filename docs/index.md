# OpenJudge

## Why OpenJudge?
OpenJudge is an **open-source evaluation framework** for **AI applications** (e.g., AI agents or chatbots) designed to **evaluate quality** and  drive **continuous application optimization**.

<div class="callout-tip" markdown>
<img src="https://unpkg.com/lucide-static@latest/icons/lightbulb.svg" class="callout-icon"> In practice, application excellence depends on a trustworthy evaluation workflow: Collect test data → Define graders → Run evaluation at scale → Analyze weaknesses → Iterate quickly.
</div>

OpenJudge provides **ready-to-use graders** and supports generating **scenario-specific rubrics (as graders)**, making this workflow **simpler**, **more professional**, and **easy to integrate** into your workflow.

It can also convert grading results into **reward signals** to help you **fine-tune** and optimize your application.

### Key Features

<div class="key-features" markdown>

+ **Systematic & Quality-Assured Grader Library**: Access 50+ production-ready graders featuring a comprehensive taxonomy, rigorously validated for reliable performance.
    - **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks via specialized graders. <a href="built_in_graders/overview/" class="feature-link">Explore Supported Scenarios<span class="link-arrow">→</span></a>
    - **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories and specific components (Memory, Reflection, Tool Use). <a href="built_in_graders/agent_graders/" class="feature-link">Agent Lifecycle Evaluation <span class="link-arrow">→</span></a>
    - **Quality Assurance:** Built for reliability. Every grader comes with benchmark datasets and pytest integration for immediate quality validation. <a href="https://huggingface.co/datasets/agentscope-ai/OpenJudge" class="feature-link" target="_blank"> View Benchmark Datasets<span class="link-arrow">→</span></a>

+ **Flexible Grader Building**: Choose the build method that fits your requirements:
    - **Customization:** Clear requirements, but no existing grader? If you have explicit rules or logic, use our Python interfaces or Prompt templates to quickly define your own grader. <a href="building_graders/create_custom_graders/" class="feature-link">Custom Grader Development Guide <span class="link-arrow">→</span></a>
    - **Zero-shot Rubrics Generation:** Not sure what criteria to use, and no labeled data yet? Just provide a task description and optional sample queries—the LLM will automatically generate evaluation rubrics for you. Ideal for rapid prototyping. <a href="building_graders/generate_rubrics_as_graders/#simple-rubric-zero-shot-generation" class="feature-link">Zero-shot Rubrics Generation Guide <span class="link-arrow">→</span></a>
    - **Data-driven Rubrics Generation:** Ambiguous requirements, but have few examples? Use the GraderGenerator to automatically summarize evaluation Rubrics from your annotated data, and generate a llm-based grader. <a href="building_graders/generate_rubrics_as_graders/#iterative-rubric-data-driven-generation" class="feature-link">Data-driven Rubrics Generation Guide <span class="link-arrow">→</span></a>
    - **Training Judge Models:** Massive data and need peak performance? Use our training pipeline to train a dedicated Judge Model. This is ideal for complex scenarios where prompt-based grading falls short. <a href="building_graders/training_judge_models/" class="feature-link">Train Judge Models <span class="link-arrow">→</span></a>

+ **Easy Integration**: Using mainstream observability platforms like **LangSmith** or **Langfuse**? We offer seamless integration to enhance their evaluators and automated evaluation capabilities. We also provide integrations with training frameworks like **VERL** for RL training.

</div>



## Quick Tutorials

<div class="card-grid">

  <a href="applications/zero_shot_evaluation/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/zap.svg" class="card-icon card-icon-general">
      <h3>Zero-Shot Evaluation</h3>
    </div>
    <p class="card-desc">
      <b>Compare models without test data:</b> Generate queries, collect responses, and rank via pairwise evaluation.
    </p>
  </a>

  <a href="get_started/evaluate_ai_agents/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/bot.svg" class="card-icon card-icon-agent">
      <h3>Evaluate An AI Agent</h3>
    </div>
    <p class="card-desc">
      <b>Agent lifecycle evaluation:</b> Assess response, trajectory, tool usage, planning, memory, and reflection.
    </p>
  </a>

  <a href="get_started/build_reward/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/brain-circuit.svg" class="card-icon card-icon-tool">
      <h3>Build Rewards for Training</h3>
    </div>
    <p class="card-desc">
      <b>Quality reward signals:</b> Aggregate graders with custom weighting for model alignment.
    </p>
  </a>



</div>


## More Tutorials

### Built-in Graders

<div class="card-grid">

  <a href="built_in_graders/agent_graders/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/bot.svg" class="card-icon card-icon-agent">
      <h3>Agent</h3>
    </div>
    <p class="card-desc">
      Agent graders for evaluating various aspects of AI agent behavior. These graders assess action selection, tool usage, memory management, planning, reflection, and overall trajectory quality.
    </p>
  </a>

  <a href="built_in_graders/general/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/globe.svg" class="card-icon card-icon-general">
      <h3>General Tasks</h3>
    </div>
    <p class="card-desc">
      Assess fundamental capabilities such as instruction following, text quality, safety guardrails, and format.
    </p>
  </a>

  <a href="built_in_graders/multimodal/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/image.svg" class="card-icon card-icon-multimodal">
      <h3>Multimodal</h3>
    </div>
    <p class="card-desc">
      Vision-language graders for evaluating AI responses involving images. These graders assess image-text coherence, image helpfulness, and text-to-image generation quality.
    </p>
  </a>

  <a href="built_in_graders/code_math/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/calculator.svg" class="card-icon card-icon-math">
      <h3>Math & Code</h3>
    </div>
    <p class="card-desc">
      Specialized graders for evaluating code generation and mathematical problem-solving capabilities. These graders assess syntax correctness, execution results, code style, and mathematical expression accuracy.
    </p>
  </a>

  <a href="built_in_graders/text/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/text.svg" class="card-icon card-icon-text">
      <h3>Text</h3>
    </div>
    <p class="card-desc">
      Algorithm-based graders for text similarity and matching. Fast, deterministic, and zero-cost evaluation using BLEU, ROUGE, F1, regex, and 15+ similarity algorithms.
    </p>
  </a>

  <a href="built_in_graders/format/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/braces.svg" class="card-icon card-icon-format">
      <h3>Format</h3>
    </div>
    <p class="card-desc">
      Format validation graders for structured outputs. Validate JSON syntax, check length constraints, detect repetition, and verify reasoning tags for chain-of-thought.
    </p>
  </a>

</div>


### Build Graders

<div class="card-grid">

  <a href="building_graders/create_custom_graders/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/wrench.svg" class="card-icon card-icon-tool">
      <h3>Customization</h3>
    </div>
    <p class="card-desc">
      <b>Clear requirements, but no existing grader?</b> If you have explicit rules or logic, use our Python interfaces or Prompt templates to quickly define your own grader.
    </p>
  </a>

  <a href="building_graders/generate_rubrics_as_graders/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/sparkles.svg" class="card-icon card-icon-data">
      <h3>Generate Rubrics</h3>
    </div>
    <p class="card-desc">
      <b>Auto-generate evaluation criteria.</b> Use Zero-Shot generation from task descriptions, or Data-Driven generation to learn rubrics from labeled preference data.
    </p>
  </a>

  <a href="building_graders/training_judge_models/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/scale.svg" class="card-icon card-icon-integration">
      <h3>Train Judge Models</h3>
    </div>
    <p class="card-desc">
      <b>Massive data and need peak performance?</b> Train dedicated judge models using SFT, Bradley-Terry, or GRPO. Supports both scalar rewards and generative evaluation with reasoning.
    </p>
  </a>

</div>

### Integrations

<div class="card-grid">

  <a href="integrations/langsmith/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/telescope.svg" class="card-icon card-icon-integration">
      <h3>LangSmith</h3>
    </div>
    <p class="card-desc">
      Build external evaluation pipelines for LangSmith. Wrap OpenJudge graders as LangSmith evaluators and run batch evaluations with GradingRunner.
    </p>
  </a>

  <a href="integrations/langfuse/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/activity.svg" class="card-icon card-icon-data">
      <h3>Langfuse</h3>
    </div>
    <p class="card-desc">
      Fetch traces from Langfuse, evaluate with OpenJudge graders, and push scores back. Supports batch processing and score aggregation.
    </p>
  </a>

  <a href="integrations/verl/" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/dumbbell.svg" class="card-icon card-icon-tool">
      <h3>VERL</h3>
    </div>
    <p class="card-desc">
      Integrate OpenJudge graders as reward functions for VERL RL training. Supports batch processing and async evaluation at scale.
    </p>
  </a>

</div>


### Applications

<div class="card-grid">

  <a href="applications/data_refinement/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/gem.svg" class="card-icon card-icon-data">
      <h3>Data Refinement</h3>
    </div>
    <p class="card-desc">
      Automate the curation of high-quality datasets. Use Graders to filter, rank, and synthesize training data for Supervised Fine-Tuning (SFT).
    </p>
  </a>

  <a href="applications/select_rank/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/scale.svg" class="card-icon card-icon-general">
      <h3>Pairwise Evaluation</h3>
    </div>
    <p class="card-desc">
      Compare and rank multiple model outputs using LLM-based pairwise comparisons. Compute win rates, generate win matrices, and identify the best-performing models.
    </p>
  </a>

</div>


### Running Graders

<div class="card-grid">

  <a href="running_graders/run_tasks/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/play.svg" class="card-icon card-icon-tool">
      <h3>Run Grading Tasks</h3>
    </div>
    <p class="card-desc">
      Orchestrate evaluations at scale with GradingRunner. Configure data mapping, control concurrency, and aggregate results from multiple graders into unified scores.
    </p>
  </a>

  <a href="running_graders/grader_analysis/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/bar-chart-2.svg" class="card-icon card-icon-data">
      <h3>Analyze Grader Results</h3>
    </div>
    <p class="card-desc">
      Transform raw scores into actionable insights. Examine score distributions, measure consistency, and compare performance against ground truth labels.
    </p>
  </a>

</div>


### Validating Graders

<div class="card-grid">

  <a href="validating_graders/overview/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/shield-check.svg" class="card-icon card-icon-general">
      <h3>Validation Overview</h3>
    </div>
    <p class="card-desc">
      Ensure your graders make accurate judgments. Learn validation workflows, best practices, and metrics for measuring grader quality.
    </p>
  </a>

  <a href="validating_graders/rewardbench2/" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/trophy.svg" class="card-icon card-icon-agent">
      <h3>RewardBench2</h3>
    </div>
    <p class="card-desc">
      Validate against the RewardBench2 benchmark for multi-domain response quality evaluation with standardized ground truth.
    </p>
  </a>

</div>