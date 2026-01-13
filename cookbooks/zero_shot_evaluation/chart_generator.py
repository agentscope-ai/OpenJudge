# -*- coding: utf-8 -*-
"""Chart generator for zero-shot evaluation results.

This module provides visualization capabilities for evaluation results,
generating beautiful bar charts to display model win rates.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from cookbooks.zero_shot_evaluation.schema import ChartConfig


class WinRateChartGenerator:
    """Generator for win rate comparison charts.

    Creates visually appealing bar charts showing model rankings
    based on pairwise evaluation results.

    Attributes:
        config: Chart configuration options

    Example:
        >>> generator = WinRateChartGenerator(config)
        >>> path = generator.generate(
        ...     rankings=[("GPT-4", 0.73), ("Claude", 0.65)],
        ...     output_dir="./results",
        ...     task_description="Translation evaluation",
        ... )
    """

    # Color palette - inspired by modern data visualization
    ACCENT_COLOR = "#FF6B35"  # Vibrant orange for best model
    ACCENT_HATCH = "///"  # Diagonal stripes pattern
    BAR_COLORS = [
        "#4A4A4A",  # Dark gray
        "#6B6B6B",  # Medium gray
        "#8C8C8C",  # Light gray
        "#ADADAD",  # Lighter gray
        "#CECECE",  # Very light gray
    ]

    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize chart generator.

        Args:
            config: Chart configuration. Uses defaults if not provided.
        """
        self.config = config or ChartConfig()

    def _configure_cjk_font(self, plt, font_manager) -> Optional[str]:
        """Configure matplotlib to support CJK (Chinese/Japanese/Korean) characters.

        Attempts to find and use a system font that supports CJK characters.
        Falls back gracefully if no suitable font is found.

        Returns:
            Font name if found, None otherwise
        """
        # Common CJK fonts on different platforms (simplified Chinese priority)
        cjk_fonts = [
            # macOS - Simplified Chinese (verified available)
            "Hiragino Sans GB",
            "Songti SC",
            "Kaiti SC",
            "Heiti SC",
            "Lantinghei SC",
            "PingFang SC",
            "STFangsong",
            # Windows
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            # Linux
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Droid Sans Fallback",
            # Generic
            "Arial Unicode MS",
        ]

        # Get available fonts
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}

        # Find the first available CJK font
        for font_name in cjk_fonts:
            if font_name in available_fonts:
                plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get("font.sans-serif", [])
                plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display
                logger.debug(f"Using CJK font: {font_name}")
                return font_name

        # No CJK font found, log warning
        logger.warning(
            "No CJK font found. Chinese characters may not display correctly. "
            "Consider installing a CJK font like 'Noto Sans CJK SC'."
        )
        return None

    def generate(
        self,
        rankings: List[Tuple[str, float]],
        output_dir: str,
        task_description: Optional[str] = None,
        total_queries: int = 0,
        total_comparisons: int = 0,
    ) -> Optional[Path]:
        """Generate win rate bar chart.

        Args:
            rankings: List of (model_name, win_rate) tuples, sorted by win rate
            output_dir: Directory to save the chart
            task_description: Task description for subtitle
            total_queries: Number of queries evaluated
            total_comparisons: Number of pairwise comparisons

        Returns:
            Path to saved chart file, or None if generation failed
        """
        if not rankings:
            logger.warning("No rankings data to visualize")
            return None

        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib import font_manager
        except ImportError:
            logger.warning("matplotlib not installed. Install with: pip install matplotlib")
            return None

        # Extract config values (defaults are centralized in ChartConfig schema)
        figsize = self.config.figsize
        dpi = self.config.dpi
        fmt = self.config.format
        show_values = self.config.show_values
        highlight_best = self.config.highlight_best
        custom_title = self.config.title

        # Prepare data (already sorted high to low)
        model_names = [r[0] for r in rankings]
        win_rates = [r[1] * 100 for r in rankings]  # Convert to percentage
        n_models = len(model_names)

        # Setup figure with modern styling (MUST be before font config)
        plt.style.use("seaborn-v0_8-whitegrid")

        # Configure font for CJK (Chinese/Japanese/Korean) support
        # This MUST be after plt.style.use() as style resets font settings
        self._configure_cjk_font(plt, font_manager)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Create bar positions
        x_pos = np.arange(n_models)
        bar_width = 0.6

        # Determine colors for each bar
        colors = []
        edge_colors = []

        for i in range(n_models):
            if i == 0 and highlight_best:
                # Best model gets accent color
                colors.append(self.ACCENT_COLOR)
                edge_colors.append(self.ACCENT_COLOR)
            else:
                # Other models get grayscale
                color_idx = min(i - 1, len(self.BAR_COLORS) - 1) if highlight_best else min(i, len(self.BAR_COLORS) - 1)
                colors.append(self.BAR_COLORS[color_idx])
                edge_colors.append(self.BAR_COLORS[color_idx])

        # Draw bars
        bars = ax.bar(
            x_pos,
            win_rates,
            width=bar_width,
            color=colors,
            edgecolor=edge_colors,
            linewidth=1.5,
            zorder=3,
        )

        # Add hatch pattern to best model
        if highlight_best and n_models > 0:
            bars[0].set_hatch(self.ACCENT_HATCH)
            bars[0].set_edgecolor("white")

        # Add value labels on top of bars
        if show_values:
            for i, (bar, rate) in enumerate(zip(bars, win_rates)):
                height = bar.get_height()
                ax.annotate(
                    f"{rate:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color="#333333",
                )

        # Customize axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=11, fontweight="medium")
        ax.set_ylabel("Win Rate (%)", fontsize=12, fontweight="medium", labelpad=10)
        ax.set_ylim(0, max(10, min(100, max(win_rates) * 1.15)))  # Add headroom for labels

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

        # Customize grid
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#DDDDDD", zorder=0)
        ax.xaxis.grid(False)

        # Title
        title = custom_title or "Model Win Rate Comparison"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#333333")

        # Subtitle with evaluation info
        subtitle_parts = []
        if task_description:
            # Truncate long descriptions
            desc = task_description[:60] + "..." if len(task_description) > 60 else task_description
            subtitle_parts.append(f"Task: {desc}")
        if total_queries > 0:
            subtitle_parts.append(f"Queries: {total_queries}")
        if total_comparisons > 0:
            subtitle_parts.append(f"Comparisons: {total_comparisons}")

        if subtitle_parts:
            subtitle = "  |  ".join(subtitle_parts)
            ax.text(
                0.5,
                1.02,
                subtitle,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                color="#666666",
                style="italic",
            )

        # Create legend
        legend_elements = []
        if highlight_best and n_models > 0:
            best_patch = mpatches.Patch(
                facecolor=self.ACCENT_COLOR,
                edgecolor="white",
                hatch=self.ACCENT_HATCH,
                label=f"Best: {model_names[0]}",
            )
            legend_elements.append(best_patch)

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                frameon=True,
                framealpha=0.9,
                fontsize=10,
            )

        # Tight layout
        plt.tight_layout()

        # Save chart
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        chart_file = output_path / f"win_rate_chart.{fmt}"

        plt.savefig(
            chart_file,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        logger.info(f"Win rate chart saved to {chart_file}")
        return chart_file
