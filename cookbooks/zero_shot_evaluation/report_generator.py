# -*- coding: utf-8 -*-
"""Report generator for zero-shot evaluation results."""

import asyncio
from typing import List

from cookbooks.zero_shot_evaluation.schema import (
    ComparisonDetail,
    OpenAIEndpoint,
    TaskConfig,
)
from cookbooks.zero_shot_evaluation.zero_shot_pipeline import EvaluationResult
from openjudge.models.openai_chat_model import OpenAIChatModel

# Constants for report generation
_NUM_WINNING_EXAMPLES_FOR_RANKING = 2
_NUM_LOSING_EXAMPLES_FOR_RANKING = 1
_NUM_SAMPLE_REASONS_PER_MODEL = 3


class ReportGenerator:
    """Generate evaluation report with parallel LLM calls."""

    def __init__(
        self,
        judge_endpoint: OpenAIEndpoint,
        language: str = "zh",
        include_examples: int = 3,
    ):
        self.language = language
        self.include_examples = include_examples
        extra_params = judge_endpoint.extra_params or {}
        self.model = OpenAIChatModel(
            model=judge_endpoint.model,
            api_key=judge_endpoint.api_key,
            base_url=judge_endpoint.base_url,
            temperature=extra_params.get("temperature", 0.3),
        )

    async def generate(
        self,
        task_config: TaskConfig,
        rubrics: List[str],
        result: EvaluationResult,
        details: List[ComparisonDetail],
    ) -> str:
        """Generate complete report with parallel section generation."""
        # Prepare context
        ctx = self._prepare_context(task_config, rubrics, result, details)

        # Generate sections in parallel
        sections = await asyncio.gather(
            self._gen_summary(ctx),
            self._gen_ranking_explanation(ctx),
            self._gen_model_analysis(ctx),
            self._gen_examples(ctx),
        )

        # Assemble report
        lang_title = "评估报告" if self.language == "zh" else "Evaluation Report"
        header = f"# {lang_title}\n\n"
        return header + "\n\n---\n\n".join(s for s in sections if s)

    def _prepare_context(
        self,
        task_config: TaskConfig,
        rubrics: List[str],
        result: EvaluationResult,
        details: List[ComparisonDetail],
    ) -> dict:
        """Prepare shared context for all sections."""
        # Filter to only original order (remove swapped duplicates)
        original_details = [d for d in details if d.order == "original"]

        # Format rankings
        rankings_text = "\n".join(f"{i+1}. {name}: {rate:.1%}" for i, (name, rate) in enumerate(result.rankings))

        # Format rubrics
        rubrics_text = "\n".join(f"- {r}" for r in rubrics)

        # Group details by model pair for examples
        model_examples = {}
        for d in original_details:
            key = tuple(sorted([d.model_a, d.model_b]))
            if key not in model_examples:
                model_examples[key] = []
            model_examples[key].append(d)

        # Select representative examples (prefer those with detailed reasons)
        selected_examples = []
        for pair_details in model_examples.values():
            sorted_details = sorted(pair_details, key=lambda x: len(x.reason), reverse=True)
            selected_examples.extend(sorted_details[: self.include_examples])

        return {
            "task_description": task_config.description,
            "scenario": task_config.scenario or "",
            "rubrics": rubrics_text,
            "rankings": rankings_text,
            "win_matrix": result.win_matrix,
            "total_queries": result.total_queries,
            "total_comparisons": result.total_comparisons,
            "best_model": result.best_pipeline,
            "model_names": [name for name, _ in result.rankings],
            "examples": selected_examples[: self.include_examples * 3],
            "all_details": original_details,  # Use deduplicated details
        }

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with given prompt."""
        lang_instruction = "Output in Chinese (中文)." if self.language == "zh" else "Output in English."
        messages = [
            {"role": "system", "content": f"You are an expert AI evaluation analyst. {lang_instruction}"},
            {"role": "user", "content": prompt},
        ]
        response = await self.model.achat(messages=messages)
        return response.content or ""

    async def _gen_summary(self, ctx: dict) -> str:
        """Generate executive summary."""
        prompt = f"""Generate a concise executive summary for an AI model evaluation.

Task: {ctx['task_description']}
Scenario: {ctx['scenario']}

Evaluation Statistics:
- Total test queries: {ctx['total_queries']}
- Total pairwise comparisons: {ctx['total_comparisons']}

Final Rankings:
{ctx['rankings']}

Best performing model: {ctx['best_model']}

Requirements:
- Write 150-200 words
- Include: evaluation purpose, methodology summary, key findings, winner
- Use professional tone"""

        content = await self._call_llm(prompt)
        title = "## 执行摘要" if self.language == "zh" else "## Executive Summary"
        return f"{title}\n\n{content}"

    async def _gen_ranking_explanation(self, ctx: dict) -> str:
        """Generate ranking explanation with evidence."""
        # Find key examples showing why top model won/lost
        best = ctx["best_model"]

        # Best model wins: either (model_a=best and winner=model_a) or (model_b=best and winner=model_b)
        winning_examples = [
            d
            for d in ctx["all_details"]
            if (d.model_a == best and d.winner == "model_a") or (d.model_b == best and d.winner == "model_b")
        ][:_NUM_WINNING_EXAMPLES_FOR_RANKING]

        # Best model loses: either (model_a=best and winner=model_b) or (model_b=best and winner=model_a)
        losing_examples = [
            d
            for d in ctx["all_details"]
            if (d.model_a == best and d.winner == "model_b") or (d.model_b == best and d.winner == "model_a")
        ][:_NUM_LOSING_EXAMPLES_FOR_RANKING]

        examples_text = ""
        for i, ex in enumerate(winning_examples + losing_examples, 1):
            actual_winner = ex.model_a if ex.winner == "model_a" else ex.model_b
            examples_text += f"""
Example {i}:
- Query: {ex.query[:200]}...
- Winner: {actual_winner}
- Reason: {ex.reason}
"""

        prompt = f"""Explain why the models are ranked this way based on the evaluation.

Rankings:
{ctx['rankings']}

Evaluation Criteria:
{ctx['rubrics']}

Win Matrix (row beats column with this rate):
{self._format_win_matrix(ctx['win_matrix'])}

Key Examples:
{examples_text}

Requirements:
- Explain why {ctx['best_model']} ranks first
- Highlight key differences between top models
- Reference specific evidence from examples
- Be objective and balanced"""

        content = await self._call_llm(prompt)
        title = "## 排名解释" if self.language == "zh" else "## Ranking Explanation"
        return f"{title}\n\n{content}"

    async def _gen_model_analysis(self, ctx: dict) -> str:
        """Generate per-model analysis."""
        # Collect stats for each model
        model_stats = {name: {"wins": 0, "losses": 0, "reasons": []} for name in ctx["model_names"]}

        for d in ctx["all_details"]:
            winner = d.model_a if d.winner == "model_a" else d.model_b
            loser = d.model_b if d.winner == "model_a" else d.model_a
            model_stats[winner]["wins"] += 1
            model_stats[loser]["losses"] += 1
            if d.reason:
                model_stats[winner]["reasons"].append(f"[Win] {d.reason[:150]}")
                model_stats[loser]["reasons"].append(f"[Loss] {d.reason[:150]}")

        stats_text = ""
        for name in ctx["model_names"]:
            stats = model_stats[name]
            sample_reasons = stats["reasons"][:_NUM_SAMPLE_REASONS_PER_MODEL]
            reasons_text = "\n".join("  * " + r for r in sample_reasons)
            stats_text += f"""
Model: {name}
- Wins: {stats['wins']}, Losses: {stats['losses']}
- Sample evaluation reasons:
{reasons_text}
"""

        prompt = f"""Analyze each model's performance in this evaluation.

Task: {ctx['task_description']}

Evaluation Criteria:
{ctx['rubrics']}

Model Statistics:
{stats_text}

Requirements:
For each model, provide:
1. Overall assessment (2-3 sentences)
2. Key strengths (with evidence)
3. Key weaknesses (with evidence)
4. Improvement suggestions"""

        content = await self._call_llm(prompt)
        title = "## 模型分析" if self.language == "zh" else "## Model Analysis"
        return f"{title}\n\n{content}"

    async def _gen_examples(self, ctx: dict) -> str:
        """Generate showcase examples."""
        examples = ctx["examples"][: self.include_examples]
        if not examples:
            return ""

        examples_text = ""
        for i, ex in enumerate(examples, 1):
            examples_text += f"""
### Case {i}

**Query:** {ex.query}

**{ex.model_a}:**
{ex.response_a[:500]}{'...' if len(ex.response_a) > 500 else ''}

**{ex.model_b}:**
{ex.response_b[:500]}{'...' if len(ex.response_b) > 500 else ''}

**Winner:** {ex.model_a if ex.winner == 'model_a' else ex.model_b}

**Evaluation Reason:** {ex.reason}
"""

        title = "## 典型案例" if self.language == "zh" else "## Representative Cases"
        return f"{title}\n{examples_text}"

    def _format_win_matrix(self, win_matrix: dict) -> str:
        """Format win matrix for display."""
        lines = []
        for model_a, opponents in win_matrix.items():
            for model_b, rate in opponents.items():
                lines.append(f"  {model_a} vs {model_b}: {rate:.1%}")
        return "\n".join(lines)
