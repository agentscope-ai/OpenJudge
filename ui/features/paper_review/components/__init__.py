# -*- coding: utf-8 -*-
"""UI components for Paper Review feature."""

from features.paper_review.components.batch_panel import (
    render_batch_progress,
    render_batch_results,
)
from features.paper_review.components.history_panel import (
    render_history_detail,
    render_history_list,
    render_history_stats,
)
from features.paper_review.components.progress_panel import (
    render_compact_progress,
    render_progress_panel,
)
from features.paper_review.components.result_panel import render_result_panel

__all__ = [
    # Progress
    "render_progress_panel",
    "render_compact_progress",
    # Result
    "render_result_panel",
    # Batch
    "render_batch_progress",
    "render_batch_results",
    # History
    "render_history_list",
    "render_history_detail",
    "render_history_stats",
]
