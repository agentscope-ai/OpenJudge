# Building Custom Graders

Extend OpenJudge beyond built-in evaluators by creating custom graders or training reward models. Build domain-specific evaluation logic that seamlessly integrates with OpenJudge's evaluation pipeline.


## Why Build Custom Graders?

While OpenJudge provides 50+ pre-built graders, custom graders enable you to evaluate industry-specific criteria (legal, medical, financial), implement proprietary scoring logic, and train models that learn from your preference data. They also help optimize costs by replacing expensive API judges with self-hosted models while maintaining consistent evaluation standards across applications.


## Building Approaches

OpenJudge supports three paths for creating custom graders, each optimized for different scenarios.


| Approach | Time to Deploy | Data Required | Best For | Cost Profile |
|----------|---------------|---------------|----------|--------------|
| **Create Custom Graders** | Minutes | None | Quick prototyping, domain-specific logic | Pay-per-query (API) or free (code-based) |
| **Generate from Data** | 1-4 hours | 50-500 examples | Iterative refinement, transparent rubrics | Medium setup + pay-per-query |
| **Train Reward Models** | 1-3 days | 1K-100K pairs | High-volume production (>1M queries/month) | High upfront, 10x lower per-query |

Use this decision tree to choose the right approach based on your data availability and requirements:

```
                         START
                           │
                           ▼
               ┌─────────────────────┐
               │ Have evaluation     │
               │ data with labels?   │
               └──────┬───────┬──────┘
                      │       │
                  YES │       │ NO
                      │       │
                      ▼       ▼
           ┌──────────────┐  ┌──────────────────┐
           │ Want to      │  │ Need evaluation  │
           │ train model? │  │ now?             │
           └────┬────┬────┘  └────┬─────────┬───┘
                │    │            │         │
            YES │    │ NO     YES │         │ NO
                │    │            │         │
                ▼    ▼            ▼         ▼
          ┌──────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
          │Train │ │Generator │ │Custom  │ │ Define   │
          │Model │ │ (Rubric) │ │Graders │ │ criteria │
          └──────┘ └──────────┘ └────────┘ └──────────┘
                │         │           │            │
                └─────────┴───────────┴────────────┘
                              │
                              ▼
                ┌───────────────────────────────┐
                │  Use in evaluation pipeline   │
                │  (GradingRunner, batch eval)  │
                └───────────────────────────────┘
```

**Choose based on your situation:**

- **Have labeled data + need automation?** → Train a reward model
- **Have data + need fast iteration?** → Generate rubrics from data
- **No data + need immediate results?** → Create custom graders

### Approach 1: Create Custom Graders

Define evaluation logic using LLM judges or code-based functions with no training required. LLM-based graders use models like `qwen3-32b` with custom prompts for domain-specific criteria. Code-based graders implement deterministic logic—checking response length, keyword presence, format validation, or compliance requirements.

**Learn more:** [Create Custom Graders →](create_custom_graders.md) | [Built-in Graders →](../built_in_graders/overview.md)


### Approach 2: Generate Graders from Data

Automatically analyze evaluation data to create structured scoring rubrics. Provide 50-500 labeled examples, and the generator extracts patterns to build interpretable criteria. Generated graders produce explicit rubrics that explain scoring decisions, ideal for scenarios requiring transparency and rapid refinement.

**Learn more:** [Generate Graders from Data →](generate_graders_from_data.md)


### Approach 3: Train Reward Models

Train neural networks on preference data to learn evaluation criteria automatically. Supports Bradley-Terry (preference pairs), Generative Pointwise (absolute scores), and Generative Pairwise (comparison decisions). Requires 1K-100K examples and 1-3 days but delivers highly consistent evaluation at 10x lower per-query cost—ideal for high-volume scenarios exceeding 1M queries per month.

**Learn more:** [Train with GRPO →](training_grpo.md) | [Bradley-Terry Training](https://github.com/modelscope/OpenJudge/tree/main/cookbooks/training_judge_model/bradley-terry) | [SFT Training](https://github.com/modelscope/OpenJudge/tree/main/cookbooks/training_judge_model/sft)



## Next Steps

- [Create Custom Graders](create_custom_graders.md) — Build graders using LLM or code-based logic
- [Generate Graders from Data](generate_graders_from_data.md) — Auto-generate rubrics from labeled data
- [Train with GRPO](training_grpo.md) — Train generative judge models with reinforcement learning
- [Built-in Graders](../built_in_graders/overview.md) — Explore pre-built graders to customize
- [Run Grading Tasks](../running_graders/run_tasks.md) — Deploy graders at scale with batch workflows

