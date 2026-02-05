# -*- coding: utf-8 -*-
"""Auto Arena feature implementation for OpenJudge Studio."""

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from core.base_feature import BaseFeature
from features.auto_arena.components.config_panel import render_config_panel
from features.auto_arena.components.history_panel import render_history_panel
from features.auto_arena.components.progress_panel import render_progress_panel
from features.auto_arena.components.report_viewer import render_report_viewer
from features.auto_arena.components.result_panel import render_result_panel
from features.auto_arena.components.sidebar import render_arena_sidebar
from features.auto_arena.services.history_manager import HistoryManager
from features.auto_arena.services.pipeline_runner import (
    PipelineProgress,
    PipelineRunner,
    PipelineStage,
)
from shared.i18n import t
from shared.utils.helpers import run_async


class AutoArenaFeature(BaseFeature):
    """Auto Arena feature.

    Provides UI for running automated model/agent evaluations that automatically
    evaluate AI models or agents without labeled data. The pipeline:
    1. Generates test queries based on task description
    2. Collects responses from multiple target models/agents
    3. Generates evaluation rubrics
    4. Runs pairwise comparisons
    5. Analyzes and ranks models/agents
    """

    feature_id = "auto_arena"
    feature_name = "Auto Arena"  # Default/fallback name
    feature_icon = "🏟️"
    feature_description = "Automated model/agent evaluation arena"  # Default/fallback
    order = 2

    @property
    def display_label(self) -> str:
        """Get the display label for navigation with i18n support."""
        return f"{self.feature_icon} {t('arena.name')}"

    def render_header(self) -> None:
        """Render the feature header with i18n support."""
        st.markdown(
            f"""<div style="margin-bottom: 1rem;">
                <h1 style="
                    font-size: 2rem;
                    font-weight: 700;
                    color: #F1F5F9;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                ">
                    <span>{self.feature_icon}</span>
                    <span>{t('arena.name')}</span>
                </h1>
                <p style="color: #94A3B8; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                    {t('arena.description')}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Session state keys
    STATE_PROGRESS = "arena_progress"
    STATE_RESULT = "arena_result"
    STATE_OUTPUT_DIR = "arena_output_dir"
    STATE_VIEWING_TASK = "arena_viewing_task"

    def render_sidebar(self) -> dict[str, Any]:
        """Render the Auto Arena sidebar configuration.

        Returns:
            Dictionary containing all sidebar configuration values
        """
        return render_arena_sidebar()

    def render_main_content(self, sidebar_config: dict[str, Any]) -> None:
        """Render the main content area for Auto Arena evaluation.

        Args:
            sidebar_config: Configuration from the sidebar
        """
        # Initialize session state
        self._init_session_state()

        # Inject CSS for help button positioned in tabs row
        st.markdown(
            """
            <style>
            /* Position help button in the tabs row */
            .help-in-tabs {
                position: absolute;
                right: 0;
                top: 50%;
                transform: translateY(-50%);
            }
            /* Make tabs container relative for positioning */
            .stTabs [data-baseweb="tab-list"] {
                position: relative;
            }
            /* Style for inline help button */
            .inline-help-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: transparent;
                border: 1px solid #64748B;
                border-radius: 6px;
                padding: 0.25rem 0.75rem;
                color: #94A3B8;
                font-size: 0.85rem;
                cursor: pointer;
                transition: all 0.2s;
                margin-left: 1rem;
            }
            .inline-help-btn:hover {
                background: rgba(99, 102, 241, 0.1);
                border-color: #6366F1;
                color: #6366F1;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Tab navigation with help as a third tab
        tab_new, tab_history, tab_help = st.tabs(
            [
                f"🆕 {t('arena.tabs.new')}",
                f"📜 {t('arena.tabs.history')}",
                f"❓ {t('arena.tabs.help')}",
            ]
        )

        with tab_new:
            self._render_new_evaluation_view(sidebar_config)

        with tab_history:
            self._render_history_view()

        with tab_help:
            self._render_quick_guide()

    def _render_new_evaluation_view(self, sidebar_config: dict[str, Any]) -> None:
        """Render the new evaluation configuration view."""
        # Two-column layout
        col_config, col_preview = st.columns([1, 1], gap="large")

        with col_config:
            config = render_config_panel(sidebar_config)

            # Start button
            st.markdown('<div class="custom-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

            # Validation
            is_valid, validation_msg = self._validate_config(config)

            if not is_valid:
                st.error(validation_msg)

            start_clicked = st.button(
                f"🚀 {t('arena.config.start')}",
                type="primary",
                use_container_width=True,
                disabled=not is_valid,
            )

        with col_preview:
            # Create a placeholder for progress/result display
            progress_placeholder = st.empty()

            # Show previous result or empty state
            result = st.session_state.get(self.STATE_RESULT)
            output_dir = st.session_state.get(self.STATE_OUTPUT_DIR)
            progress = st.session_state.get(self.STATE_PROGRESS)

            if result:
                with progress_placeholder.container():
                    render_result_panel(result=result, output_dir=output_dir)
            elif progress and progress.stage and progress.stage.value not in ["completed", "failed", "not_started"]:
                # Show running state
                with progress_placeholder.container():
                    render_progress_panel(
                        is_running=True,
                        current_stage=progress.stage.value if progress.stage else "",
                        total_progress=progress.total_progress,
                    )
            else:
                with progress_placeholder.container():
                    render_progress_panel(is_running=False)

        # Handle start button click - this runs the evaluation with progress in the right panel
        if start_clicked:
            self._start_evaluation(config, progress_placeholder)

    def _render_history_view(self) -> None:
        """Render the history view with past evaluations."""
        viewing_task = st.session_state.get(self.STATE_VIEWING_TASK)

        if viewing_task:
            # Show report viewer
            render_report_viewer(
                task_id=viewing_task,
                on_back=self._on_back_from_report,
            )
        else:
            # Show history list
            render_history_panel(
                on_view=self._on_view_task,
                on_resume=self._on_resume_task,
                on_delete=self._on_delete_task,
                limit=15,
            )

    def _on_view_task(self, task_id: str) -> None:
        """Handle view task button click."""
        st.session_state[self.STATE_VIEWING_TASK] = task_id
        st.rerun()

    def _on_resume_task(self, task_id: str) -> None:
        """Handle resume task button click."""
        history_manager = HistoryManager()
        details = history_manager.get_task_details(task_id)
        if details:
            output_dir = details.get("task_dir")
            st.session_state[self.STATE_OUTPUT_DIR] = output_dir
            st.info(t("arena.history.resuming", task_id=task_id))

            # Use PipelineRunner.resume to continue from checkpoint
            try:
                runner = PipelineRunner.resume(output_dir)
                result = run_async(runner.run())
                if result:
                    st.session_state[self.STATE_RESULT] = result
                    st.success(t("arena.history.resume_complete"))
            except Exception as e:
                st.error(t("arena.history.resume_failed", error=str(e)))

    def _on_delete_task(self, task_id: str) -> None:
        """Handle delete task button click."""
        history_manager = HistoryManager()
        if history_manager.delete_task(task_id):
            st.success(t("arena.history.task_deleted", task_id=task_id))
            st.rerun()
        else:
            st.error(t("arena.history.delete_failed"))

    def _on_back_from_report(self) -> None:
        """Handle back button from report viewer."""
        st.session_state[self.STATE_VIEWING_TASK] = None
        st.rerun()

    def _init_session_state(self) -> None:
        """Initialize session state variables."""
        if self.STATE_PROGRESS not in st.session_state:
            st.session_state[self.STATE_PROGRESS] = None
        if self.STATE_RESULT not in st.session_state:
            st.session_state[self.STATE_RESULT] = None
        if self.STATE_OUTPUT_DIR not in st.session_state:
            st.session_state[self.STATE_OUTPUT_DIR] = None
        if self.STATE_VIEWING_TASK not in st.session_state:
            st.session_state[self.STATE_VIEWING_TASK] = None

    def _validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not config.get("task_description"):
            return False, t("arena.validation.task_required")

        endpoints = config.get("target_endpoints", {})
        valid_endpoints = [ep for ep in endpoints.values() if ep.get("api_key") and ep.get("model")]
        if len(valid_endpoints) < 2:
            total = len(endpoints)
            configured = len(valid_endpoints)
            return (
                False,
                t("arena.validation.min_models", configured=configured, total=total),
            )

        if not config.get("judge_api_key"):
            return False, t("arena.validation.judge_api_required")

        return True, ""

    def _start_evaluation(  # pylint: disable=too-many-statements
        self, config: dict[str, Any], progress_placeholder: Any
    ) -> None:
        """Start the evaluation pipeline.

        Displays progress in the right panel using the provided placeholder.
        Calls pipeline.evaluate() directly to ensure checkpoint system works properly.

        Args:
            config: Configuration dictionary
            progress_placeholder: Streamlit empty placeholder for progress display
        """
        # Track evaluation success for auto-refresh
        evaluation_success = False

        # Generate output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path.home() / ".openjudge_studio" / "evaluations" / f"zs_{timestamp}")
        config["output_dir"] = output_dir

        # Initialize progress state
        progress = PipelineProgress()
        progress.output_dir = output_dir
        st.session_state[self.STATE_PROGRESS] = progress
        st.session_state[self.STATE_OUTPUT_DIR] = output_dir
        st.session_state[self.STATE_RESULT] = None

        # Use st.status in the right panel for real-time progress display
        with progress_placeholder.container():
            with st.status(f"🔄 {t('arena.progress.running')}", expanded=True) as status:
                try:
                    # Build config and create pipeline
                    runner = PipelineRunner(config)

                    # Save UI config for resume capability
                    runner.save_config()

                    arena_config = runner._build_arena_config()  # pylint: disable=protected-access

                    from cookbooks.auto_arena.auto_arena_pipeline import (
                        AutoArenaPipeline,
                    )

                    status.update(label=f"🔄 {t('arena.progress.initializing')}")
                    st.write(f"**{t('arena.progress.init_desc')}**")
                    st.write(f"- {t('arena.progress.task')}: {config.get('task_description', '')[:50]}...")
                    st.write(f"- {t('arena.progress.target_models')}: {len(config.get('target_endpoints', {}))}")
                    st.write(f"- {t('arena.progress.queries_to_generate')}: {config.get('num_queries', 20)}")

                    # Create pipeline with resume support
                    pipeline = AutoArenaPipeline(config=arena_config, resume=True)

                    progress.update_stage(PipelineStage.QUERIES, 0.0, t("arena.progress.running"))
                    st.session_state[self.STATE_PROGRESS] = progress

                    status.update(label=f"🔄 {t('arena.progress.running_pipeline')}")
                    st.write("---")
                    st.write(f"**{t('arena.progress.running_pipeline')}** {t('arena.progress.running_desc')}")
                    st.write(f"{t('arena.progress.pipeline_steps')}")
                    st.write(f"1. {t('arena.progress.step1')}")
                    st.write(f"2. {t('arena.progress.step2')}")
                    st.write(f"3. {t('arena.progress.step3')}")
                    st.write(f"4. {t('arena.progress.step4')}")
                    st.write(f"5. {t('arena.progress.step5')}")

                    # Run the complete evaluation pipeline
                    result = run_async(pipeline.evaluate())

                    # Completed
                    progress.stage = PipelineStage.COMPLETED
                    progress.total_progress = 1.0
                    progress.result = result.model_dump()
                    st.session_state[self.STATE_PROGRESS] = progress
                    st.session_state[self.STATE_RESULT] = result.model_dump()

                    # Save results
                    pipeline.save_results(result)

                    status.update(label=f"✅ {t('arena.progress.complete')}", state="complete")
                    st.write("---")
                    st.write(f"✅ **{t('arena.progress.completed_success')}**")
                    st.write(f"🏆 **{t('arena.progress.best_model')}:** {result.best_pipeline}")
                    st.write(f"📊 **{t('arena.progress.total_queries')}:** {result.total_queries}")
                    st.write(f"⚖️ **{t('arena.progress.total_comparisons')}:** {result.total_comparisons}")

                    # Show rankings
                    st.write("---")
                    st.write(f"**{t('arena.progress.rankings')}:**")
                    for rank, (name, win_rate) in enumerate(result.rankings, 1):
                        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                        st.write(f"{medal} {name}: {win_rate:.1%}")

                    # Mark evaluation as completed successfully
                    evaluation_success = True

                except Exception as e:
                    progress.stage = PipelineStage.FAILED
                    progress.error = str(e)
                    st.session_state[self.STATE_PROGRESS] = progress
                    status.update(label=f"❌ {t('arena.progress.failed')}", state="error")
                    st.error(t("arena.progress.failed_msg", error=str(e)))
                    st.write("---")
                    st.write(f"💡 **{t('arena.progress.resume_tip')}**")
                    evaluation_success = False

        # Auto-refresh page after evaluation completes to show full result panel with chart
        if evaluation_success:
            st.rerun()

    def _render_quick_guide(self) -> None:
        """Render the quick start guide."""
        st.markdown(
            f"""
            <div class="feature-card">
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
                    {t("arena.help.title")}
                </div>
                <div class="guide-step">
                    <div class="guide-number">1</div>
                    <div class="guide-text">
                        <strong>{t("arena.help.step1_title")}:</strong> {t("arena.help.step1_desc")}
                    </div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">2</div>
                    <div class="guide-text">
                        <strong>{t("arena.help.step2_title")}:</strong> {t("arena.help.step2_desc")}
                    </div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">3</div>
                    <div class="guide-text">
                        <strong>{t("arena.help.step3_title")}:</strong> {t("arena.help.step3_desc")}
                    </div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">4</div>
                    <div class="guide-text">
                        <strong>{t("arena.help.step4_title")}:</strong> {t("arena.help.step4_desc")}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def on_mount(self) -> None:
        """Initialize Auto Arena feature state when mounted."""
        self._init_session_state()

    def on_unmount(self) -> None:
        """Cleanup when feature is unmounted."""
        # Could cancel running pipeline here if needed
