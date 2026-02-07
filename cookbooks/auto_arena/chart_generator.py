# -*- coding: utf-8 -*-
"""Chart generator for Auto Arena evaluation results.

This module provides visualization capabilities for evaluation results,
generating beautiful bar charts to display model/agent win rates.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from cookbooks.auto_arena.schema import ChartConfig


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

    def _get_figsize(self) -> Tuple[float, float]:
        """Get figure size based on orientation.

        Returns:
            Figure size (width, height) in inches
        """
        orientation = getattr(self.config, "orientation", "horizontal")
        if orientation == "vertical":
            # 3:4 ratio (portrait mode), 1080x1440 pixels at ~120 DPI = 9x12 inches
            return (9, 12)
        return self.config.figsize

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
        orientation = getattr(self.config, "orientation", "horizontal")
        figsize = self._get_figsize()
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
        y_pos = np.arange(n_models)
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

        if orientation == "vertical":
            # Vertical (portrait) mode: use horizontal bars
            # Use reversed y positions so y=max is at top, but keep data in original order
            # (rankings are already sorted high to low, so first item should be at top)
            reversed_y_pos = y_pos[::-1]

            bars = ax.barh(
                reversed_y_pos,
                win_rates,
                height=bar_width,
                color=colors,
                edgecolor=edge_colors,
                linewidth=1.5,
                zorder=3,
            )

            # Add hatch pattern to best model (first bar, at top)
            if highlight_best and n_models > 0:
                bars[0].set_hatch(self.ACCENT_HATCH)
                bars[0].set_edgecolor("white")

            # Add value labels at end of bars
            if show_values:
                for bar, rate in zip(bars, win_rates):
                    width = bar.get_width()
                    ax.annotate(
                        f"{rate:.1f}%",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="#333333",
                    )

            # Customize axes for horizontal bars
            ax.set_yticks(reversed_y_pos)
            ax.set_yticklabels(model_names, fontsize=11, fontweight="medium")
            ax.set_xlabel("Win Rate (%)", fontsize=12, fontweight="medium", labelpad=10)
            ax.set_xlim(0, max(10, min(100, max(win_rates) * 1.15)))

            # Customize grid
            ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#DDDDDD", zorder=0)
            ax.yaxis.grid(False)

            # Legend location for vertical
            legend_loc = "lower right"
        else:
            # Horizontal (landscape) mode: use vertical bars
            bars = ax.bar(
                y_pos,
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
                for bar, rate in zip(bars, win_rates):
                    height = bar.get_height()
                    ax.annotate(
                        f"{rate:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                        color="#333333",
                    )

            # Customize axes for vertical bars
            ax.set_xticks(y_pos)
            ax.set_xticklabels(model_names, fontsize=11, fontweight="medium")
            ax.set_ylabel("Win Rate (%)", fontsize=12, fontweight="medium", labelpad=10)
            ax.set_ylim(0, max(10, min(100, max(win_rates) * 1.15)))

            # Customize grid
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#DDDDDD", zorder=0)
            ax.xaxis.grid(False)

            # Legend location for horizontal
            legend_loc = "upper right"

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

        # Title
        title = custom_title or "Model Win Rate Comparison"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#333333")

        # Subtitle with task description only
        subtitle_parts = []
        if task_description:
            # Truncate long descriptions (shorter for vertical mode)
            max_len = 50 if orientation == "vertical" else 70
            desc = task_description[:max_len] + "..." if len(task_description) > max_len else task_description
            subtitle_parts.append(f"Task: {desc}")

        if subtitle_parts:
            # Use newlines for vertical mode, pipes for horizontal
            if orientation == "vertical":
                subtitle = "\n".join(subtitle_parts)
                # Place subtitle above title with more padding for vertical
                ax.text(
                    0.5,
                    1.08,
                    subtitle,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#666666",
                    style="italic",
                    linespacing=1.5,
                )
            else:
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
                loc=legend_loc,
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

    def generate_matrix(
        self,
        win_matrix: dict,
        model_order: List[str],
        output_dir: str,
        task_description: Optional[str] = None,
        total_queries: int = 0,
        total_comparisons: int = 0,
    ) -> Optional[Path]:
        """Generate win rate matrix heatmap.

        Args:
            win_matrix: Dict[str, Dict[str, float]] where win_matrix[A][B] = A beats B rate
            model_order: List of model names in display order (usually by ranking)
            output_dir: Directory to save the chart
            task_description: Task description for subtitle
            total_queries: Number of queries evaluated
            total_comparisons: Number of pairwise comparisons

        Returns:
            Path to saved chart file, or None if generation failed
        """
        if not win_matrix or len(model_order) < 2:
            logger.warning("Need at least 2 models to generate matrix")
            return None

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib import font_manager
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            logger.warning("matplotlib not installed. Install with: pip install matplotlib")
            return None

        # Extract config values
        dpi = self.config.dpi
        fmt = self.config.format
        show_values = self.config.show_values

        n_models = len(model_order)

        # Build matrix data (row = opponent, col = model, value = model beats opponent rate)
        # Transposed view: matrix_data[opponent][model] = model beats opponent
        matrix_data = np.zeros((n_models, n_models))
        for i, opponent in enumerate(model_order):
            for j, model in enumerate(model_order):
                if opponent == model:
                    matrix_data[i, j] = np.nan  # Diagonal
                else:
                    # model beats opponent rate
                    matrix_data[i, j] = win_matrix.get(model, {}).get(opponent, 0.5) * 100

        # Figure size - square aspect for matrix
        cell_size = 1.2
        fig_width = max(8, n_models * cell_size + 3)
        fig_height = max(6, n_models * cell_size + 2)
        figsize = (fig_width, fig_height)

        # Setup figure with modern styling
        plt.style.use("seaborn-v0_8-whitegrid")
        self._configure_cjk_font(plt, font_manager)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Custom colormap: light gray -> white (50%) -> orange with better contrast
        colors = [
            (0.0, "#E0E0E0"),  # 0% - light gray (loss)
            (0.3, "#F0F0F0"),  # 30% - very light gray
            (0.5, "#FFFFFF"),  # 50% - white (tie)
            (0.6, "#FFE8DC"),  # 60% - very light orange
            (0.7, "#FFCDB2"),  # 70% - light orange
            (0.85, "#FF8C5A"),  # 85% - medium orange
            (1.0, "#E85A20"),  # 100% - deep orange (win)
        ]
        cmap_colors = [c[1] for c in colors]
        cmap_positions = [c[0] for c in colors]
        cmap = LinearSegmentedColormap.from_list("win_rate", list(zip(cmap_positions, cmap_colors)))
        cmap.set_bad(color="#F0F0F0")  # Color for diagonal (NaN)

        # Create heatmap
        im = ax.imshow(matrix_data, cmap=cmap, aspect="equal", vmin=0, vmax=100)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label("Win Rate (%)", fontsize=11, fontweight="medium", labelpad=10)
        cbar.ax.tick_params(labelsize=9)

        # Set ticks and labels
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(model_order, fontsize=10, fontweight="medium")
        ax.set_yticklabels(model_order, fontsize=10, fontweight="medium")

        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add cell values with black bold text
        if show_values:
            for i in range(n_models):
                for j in range(n_models):
                    if i == j:
                        text = "—"
                        color = "#666666"
                    else:
                        value = matrix_data[i, j]
                        text = f"{value:.0f}%"
                        color = "#000000"  # Black for all values
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=13,
                        fontweight="bold",
                        color=color,
                    )

        # Add grid lines
        ax.set_xticks(np.arange(n_models + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_models + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#DDDDDD", linestyle="-", linewidth=1)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Subtitle with task description only (placed above title)
        subtitle_parts = []
        if task_description:
            max_len = 60
            desc = task_description[:max_len] + "..." if len(task_description) > max_len else task_description
            subtitle_parts.append(f"Task: {desc}")

        # Title with extra padding if subtitle exists
        title = "Model Win Rate Matrix"
        title_pad = 35 if subtitle_parts else 20
        ax.set_title(title, fontsize=16, fontweight="bold", pad=title_pad, color="#333333")

        # Place subtitle above title
        if subtitle_parts:
            subtitle = "  |  ".join(subtitle_parts)
            ax.text(
                0.5,
                1.08,
                subtitle,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#666666",
                style="italic",
            )

        # Axis labels
        ax.set_xlabel("Model (Column)", fontsize=11, fontweight="medium", labelpad=10)
        ax.set_ylabel("Opponent (Row)", fontsize=11, fontweight="medium", labelpad=10)

        # Add interpretation note at bottom
        note = "Column model vs Row opponent: higher % = column model wins more often"
        fig.text(
            0.5,
            0.02,
            note,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#888888",
            style="italic",
        )

        # Tight layout
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save chart
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        chart_file = output_path / f"win_rate_matrix.{fmt}"

        plt.savefig(
            chart_file,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        logger.info(f"Win rate matrix saved to {chart_file}")
        return chart_file
