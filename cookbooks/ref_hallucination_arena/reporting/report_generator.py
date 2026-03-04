# -*- coding: utf-8 -*-
"""Markdown report generator for Reference Hallucination Arena."""

from datetime import datetime
from typing import Dict, List, Optional

from cookbooks.ref_hallucination_arena.schema import (
    ArenaResult,
    ModelVerificationResult,
    RefArenaConfig,
    VerificationStatus,
)


class RefReportGenerator:
    """Generate a Markdown evaluation report based on verification results.

    Reports are purely based on objective verification metrics (no LLM judge).
    """

    def __init__(self, language: str = "zh", include_examples: int = 3):
        self.lang = language
        self.include_examples = include_examples

    def generate(
        self,
        config: RefArenaConfig,
        result: ArenaResult,
        all_verification_details: Optional[Dict[str, List[ModelVerificationResult]]] = None,
    ) -> str:
        """Generate the full Markdown report.

        Args:
            config: Arena configuration.
            result: Final arena result with rankings and scores.
            all_verification_details: Optional per-model verification details for examples.

        Returns:
            Markdown string.
        """
        sections = [
            self._header(config),
            self._summary(result),
            self._accuracy_table(result),
            self._rankings_table(result),
            self._discipline_breakdown(result),
            self._source_breakdown(result),
        ]
        if all_verification_details:
            sections.append(self._examples_section(result, all_verification_details))
        sections.append(self._footer())

        return "\n\n".join(sections)

    # ---- Sections ----

    def _header(self, config: RefArenaConfig) -> str:
        if self.lang == "zh":
            return (
                "# 文献推荐幻觉评估报告\n\n"
                f"**任务描述**: {config.task.description}\n\n"
                f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"**评估模型**: {', '.join(config.target_endpoints.keys())}\n\n"
                f"**数据集**: {config.dataset.path}"
            )
        return (
            "# Reference Hallucination Arena Report\n\n"
            f"**Task**: {config.task.description}\n\n"
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"**Models**: {', '.join(config.target_endpoints.keys())}\n\n"
            f"**Dataset**: {config.dataset.path}"
        )

    def _summary(self, result: ArenaResult) -> str:
        best = result.rankings[0] if result.rankings else ("N/A", 0.0)
        worst = result.rankings[-1] if result.rankings else ("N/A", 0.0)

        if self.lang == "zh":
            return (
                "## 总览\n\n"
                f"| 指标 | 值 |\n|---|---|\n"
                f"| 评估查询数 | {result.total_queries} |\n"
                f"| 推荐文献总数 | {result.total_references} |\n"
                f"| 验证通过总数 | {result.total_verified} |\n"
                f"| 总体整体准确率 | {result.overall_accuracy:.1%} |\n"
                f"| 最佳模型 | {best[0]} ({best[1]:.1%}) |\n"
                f"| 最差模型 | {worst[0]} ({worst[1]:.1%}) |"
            )
        return (
            "## Summary\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Total Queries | {result.total_queries} |\n"
            f"| Total References | {result.total_references} |\n"
            f"| Total Verified | {result.total_verified} |\n"
            f"| Overall Accuracy | {result.overall_accuracy:.1%} |\n"
            f"| Best Model | {best[0]} ({best[1]:.1%}) |\n"
            f"| Worst Model | {worst[0]} ({worst[1]:.1%}) |"
        )

    def _accuracy_table(self, result: ArenaResult) -> str:
        """Per-field accuracy breakdown table."""
        if self.lang == "zh":
            title = "## 多维度准确率分析"
            header = "| 模型 | 标题准确率 | 作者准确率 | 年份准确率 | DOI准确率 | **整体准确率** | 推荐总数 |"
            sep = "|---|---|---|---|---|---|---|"
        else:
            title = "## Per-Field Accuracy Breakdown"
            header = "| Model | Title Acc. | Author Acc. | Year Acc. | DOI Acc. | **Overall Acc.** | Total Refs |"
            sep = "|---|---|---|---|---|---|---|"

        rows = [title, "", header, sep]
        for name, _ in result.rankings:
            ms = result.model_scores[name]
            rows.append(
                f"| {name} | {ms.title_accuracy:.1%} | {ms.author_accuracy:.1%} | "
                f"{ms.year_accuracy:.1%} | {ms.doi_accuracy:.1%} | "
                f"**{ms.overall_accuracy:.1%}** | {ms.total_refs} |"
            )

        # Explanation
        if self.lang == "zh":
            rows.append("")
            rows.append(
                "> **说明**: 各维度准确率表示该字段与真实论文完全匹配的比例。"
                "整体准确率要求标题、作者、年份全部正确才算通过，是最严格的指标。"
            )
        else:
            rows.append("")
            rows.append(
                "> **Note**: Each field accuracy shows the percentage of references "
                "where that field exactly matches a real paper. Overall accuracy "
                "requires ALL fields to match and is the strictest metric."
            )
        return "\n".join(rows)

    def _rankings_table(self, result: ArenaResult) -> str:
        title = "## 模型排名" if self.lang == "zh" else "## Model Rankings"

        # Check if any model has year constraint data
        has_yc = any(ms.year_constrained_refs > 0 for ms in result.model_scores.values())

        if self.lang == "zh":
            header = "| 排名 | 模型 | 整体准确率 | 文献完整度 |"
            if has_yc:
                header += " 时间合规率 |"
            header += " 推荐总数 |"
            sep = "|---|---|---|---|"
            if has_yc:
                sep += "---|"
            sep += "---|"
        else:
            header = "| Rank | Model | Overall Accuracy | Completeness |"
            if has_yc:
                header += " Year Compliance |"
            header += " Total Refs |"
            sep = "|---|---|---|---|"
            if has_yc:
                sep += "---|"
            sep += "---|"

        rows = [title, "", header, sep]
        for rank, (name, _) in enumerate(result.rankings, 1):
            ms = result.model_scores[name]
            row = f"| {rank} | {name} | **{ms.overall_accuracy:.1%}** | " f"{ms.completeness:.1%} |"
            if has_yc:
                if ms.year_constrained_refs > 0:
                    row += f" {ms.year_compliance_rate:.1%} |"
                else:
                    row += " - |"
            row += f" {ms.total_refs} |"
            rows.append(row)
        return "\n".join(rows)

    def _discipline_breakdown(self, result: ArenaResult) -> str:
        title = "## 学科维度分析" if self.lang == "zh" else "## Discipline Breakdown"

        # Collect all disciplines
        all_disciplines = set()
        for ms in result.model_scores.values():
            all_disciplines.update(ms.discipline_scores.keys())

        if not all_disciplines:
            return f"{title}\n\n暂无学科维度数据。" if self.lang == "zh" else f"{title}\n\nNo discipline data."

        parts = [title]
        for disc in sorted(all_disciplines):
            disc_title = f"\n### {disc}\n"
            if self.lang == "zh":
                header = "| 模型 | 整体准确率 | 文献数 |"
            else:
                header = "| Model | Overall Accuracy | Refs |"
            sep = "|---|---|---|"

            rows = [disc_title, header, sep]
            for name, _ in result.rankings:
                ms = result.model_scores[name]
                ds = ms.discipline_scores.get(disc)
                if ds:
                    rows.append(f"| {name} | {ds.verification_rate:.1%} | {ds.total_refs} |")
            parts.append("\n".join(rows))

        return "\n".join(parts)

    def _source_breakdown(self, result: ArenaResult) -> str:
        title = "## 验证来源分布" if self.lang == "zh" else "## Verification Source Distribution"

        if self.lang == "zh":
            header = "| 模型 | Crossref | PubMed | arXiv | DBLP |"
        else:
            header = "| Model | Crossref | PubMed | arXiv | DBLP |"
        sep = "|---|---|---|---|---|"

        rows = [title, "", header, sep]
        for name, _ in result.rankings:
            ms = result.model_scores[name]
            s = ms.verified_by_source
            rows.append(
                f"| {name} | {s.get('crossref', 0)} | {s.get('pubmed', 0)} | "
                f"{s.get('arxiv', 0)} | {s.get('dblp', 0)} |"
            )
        return "\n".join(rows)

    def _examples_section(
        self,
        result: ArenaResult,
        all_details: Dict[str, List[ModelVerificationResult]],
    ) -> str:
        title = "## 典型案例" if self.lang == "zh" else "## Representative Examples"
        parts = [title]

        # Pick from best model's results
        if not result.rankings:
            return title

        best_model = result.rankings[0][0]
        details = all_details.get(best_model, [])

        shown = 0
        for mvr in details:
            if shown >= self.include_examples:
                break

            # Find one verified and one hallucination example
            verified_ex = None
            halluc_ex = None
            for vr in mvr.results:
                if vr.status == VerificationStatus.VERIFIED and not verified_ex:
                    verified_ex = vr
                elif vr.status == VerificationStatus.NOT_FOUND and not halluc_ex:
                    halluc_ex = vr

            if verified_ex or halluc_ex:
                if self.lang == "zh":
                    parts.append(f"\n### 查询: {mvr.query[:80]}...")
                else:
                    parts.append(f"\n### Query: {mvr.query[:80]}...")

                if verified_ex:
                    ref = verified_ex.reference
                    label = "验证通过" if self.lang == "zh" else "Verified"
                    parts.append(
                        f"\n**{label}**: {ref.title}\n"
                        f"- {verified_ex.message}\n"
                        f"- Source: {verified_ex.source}, Confidence: {verified_ex.confidence:.2f}"
                    )

                if halluc_ex:
                    ref = halluc_ex.reference
                    label = "幻觉" if self.lang == "zh" else "Hallucination"
                    parts.append(f"\n**{label}**: {ref.title}\n" f"- {halluc_ex.message}")

                shown += 1

        return "\n".join(parts)

    def _footer(self) -> str:
        if self.lang == "zh":
            return (
                "---\n\n"
                "*本报告基于 Crossref、PubMed、arXiv、DBLP 的客观验证结果生成。"
                "所有指标均为「越高越好」：整体准确率要求标题、作者、年份全部与真实论文完全一致。*"
            )
        return (
            "---\n\n"
            "*This report is based on objective verification against Crossref, PubMed, arXiv, and DBLP. "
            'All metrics are "higher is better": overall accuracy requires title, author, and year '
            "to exactly match a real paper.*"
        )
