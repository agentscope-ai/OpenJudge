# -*- coding: utf-8 -*-
"""Chart generator for Reference Hallucination Arena.

Generates:
  1. Verification rate bar chart (ranking)
  2. Hallucination rate comparison chart
  3. Discipline-level radar/grouped bar chart
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from cookbooks.ref_hallucination_arena.schema import ArenaResult, ChartConfig


def _setup_cjk_font():
    """Try to configure CJK font for matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk_fonts = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
    ]
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for font in cjk_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams.get("font.sans-serif", [])
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


class RefChartGenerator:
    """Generate charts for arena evaluation results."""

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()

    def generate_verification_chart(
        self,
        result: ArenaResult,
        output_dir: str,
    ) -> Optional[str]:
        """Generate verification rate ranking bar chart.

        Args:
            result: Arena evaluation result.
            output_dir: Directory to save chart.

        Returns:
            Path to saved chart, or None on failure.
        """
        try:
            _setup_cjk_font()
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available, skipping chart generation")
            return None

        rankings = result.rankings
        if not rankings:
            return None

        names = [n for n, _ in rankings]

        # Collect per-field accuracy metrics for each model
        metrics = {
            "Title": [result.model_scores[n].title_accuracy * 100 for n in names],
            "Author": [result.model_scores[n].author_accuracy * 100 for n in names],
            "Year": [result.model_scores[n].year_accuracy * 100 for n in names],
            "DOI": [result.model_scores[n].doi_accuracy * 100 for n in names],
            "Overall": [result.model_scores[n].overall_accuracy * 100 for n in names],
        }
        colors_map = {
            "Title": "#42A5F5",
            "Author": "#66BB6A",
            "Year": "#FFA726",
            "DOI": "#AB47BC",
            "Overall": "#EF5350",
        }
        metric_keys = list(metrics.keys())
        n_metrics = len(metric_keys)

        fig, ax = plt.subplots(figsize=(max(8, len(names) * n_metrics * 0.4 + 2), 6), dpi=self.config.dpi)
        x_pos = np.arange(len(names))
        width = 0.8 / n_metrics

        for i, key in enumerate(metric_keys):
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(
                x_pos + offset,
                metrics[key],
                width,
                label=key,
                color=colors_map[key],
                alpha=1.0 if key == "Overall" else 0.75,
                edgecolor="black" if key == "Overall" else "none",
                linewidth=1.2 if key == "Overall" else 0,
            )
            if self.config.show_values and key == "Overall":
                for bar in bars:
                    h = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 1,
                        f"{h:.1f}%",
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8, loc="upper right")

        title = self.config.title or "Reference Accuracy Breakdown"
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        output_path = Path(output_dir) / f"verification_chart.{self.config.format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Verification chart saved to {output_path}")
        return str(output_path)

    def generate_discipline_chart(
        self,
        result: ArenaResult,
        output_dir: str,
    ) -> Optional[str]:
        """Generate grouped bar chart showing verification rate by discipline.

        Args:
            result: Arena evaluation result.
            output_dir: Directory to save chart.

        Returns:
            Path to saved chart, or None.
        """
        try:
            _setup_cjk_font()
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        # Collect all disciplines
        all_disciplines = set()
        for ms in result.model_scores.values():
            all_disciplines.update(ms.discipline_scores.keys())
        all_disciplines = sorted(all_disciplines)

        if not all_disciplines:
            return None

        model_names = [n for n, _ in result.rankings]
        n_models = len(model_names)
        n_disc = len(all_disciplines)

        fig, ax = plt.subplots(figsize=(max(8, n_disc * n_models * 0.4), 6), dpi=self.config.dpi)

        x = np.arange(n_disc)
        width = 0.8 / n_models
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, name in enumerate(model_names):
            ms = result.model_scores[name]
            rates = [ms.discipline_scores.get(d, None) for d in all_disciplines]
            values = [(ds.verification_rate * 100 if ds else 0.0) for ds in rates]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=name, color=colors[i], alpha=0.85)

            if self.config.show_values:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.5,
                            f"{h:.0f}%",
                            ha="center",
                            fontsize=7,
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(all_disciplines, rotation=20, ha="right")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Overall Accuracy by Discipline", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()

        output_path = Path(output_dir) / f"discipline_chart.{self.config.format}"
        fig.savefig(str(output_path), dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Discipline chart saved to {output_path}")
        return str(output_path)
