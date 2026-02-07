# -*- coding: utf-8 -*-
"""UI components for Auto Arena feature."""

from features.auto_arena.components.config_panel import render_config_panel
from features.auto_arena.components.history_panel import render_history_panel
from features.auto_arena.components.progress_panel import render_progress_panel
from features.auto_arena.components.report_viewer import render_report_viewer
from features.auto_arena.components.result_panel import render_result_panel
from features.auto_arena.components.sidebar import render_arena_sidebar

__all__ = [
    "render_config_panel",
    "render_history_panel",
    "render_progress_panel",
    "render_report_viewer",
    "render_result_panel",
    "render_arena_sidebar",
]
