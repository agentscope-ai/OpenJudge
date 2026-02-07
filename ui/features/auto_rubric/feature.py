# -*- coding: utf-8 -*-
"""Auto Rubric feature implementation for OpenJudge Studio.

Phase 1: Simple Rubric mode (zero-shot from task description)
Phase 2: Iterative Rubric mode (data-driven from labeled data)
         + History management
"""

from typing import Any

import streamlit as st
from core.base_feature import BaseFeature
from features.auto_rubric.components.history_panel import (
    render_history_panel,
    render_task_detail,
)
from features.auto_rubric.components.iterative_config_panel import (
    render_iterative_config_panel,
    validate_iterative_config,
)
from features.auto_rubric.components.result_panel import render_result_panel
from features.auto_rubric.components.sidebar import render_rubric_sidebar
from features.auto_rubric.components.simple_config_panel import (
    render_simple_config_panel,
    validate_simple_config,
)
from features.auto_rubric.services.history_manager import HistoryManager
from features.auto_rubric.services.rubric_generator_service import (
    IterativeRubricConfig,
    RubricGeneratorService,
    SimpleRubricConfig,
)
from shared.i18n import t
from shared.utils.helpers import run_async

from openjudge.graders.schema import GraderMode
from openjudge.models.schema.prompt_template import LanguageEnum


class AutoRubricFeature(BaseFeature):
    """Auto Rubric feature.

    Provides UI for automatically generating evaluation rubrics without
    manual design.

    Modes:
    - Simple Rubric: Zero-shot generation from task description
    - Iterative Rubric: Data-driven generation from labeled data
    """

    feature_id = "auto_rubric"
    feature_name = "Grader Generator"  # Fallback name, actual name from i18n
    feature_icon = "🔧"
    feature_description = "Automatically generate evaluation graders"
    order = 3

    # Session state keys
    STATE_RESULT = "rubric_result"
    STATE_CONFIG = "rubric_config"
    STATE_GRADER = "rubric_grader"
    STATE_MODE = "rubric_generation_mode"
    STATE_VIEWING_TASK = "rubric_viewing_task"

    @property
    def display_label(self) -> str:
        """Get the display label for navigation with i18n support."""
        return f"{self.feature_icon} {t('rubric.name')}"

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
                    <span>{t('rubric.name')}</span>
                </h1>
                <p style="color: #94A3B8; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                    {t('rubric.description')}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    def render_sidebar(self) -> dict[str, Any]:
        """Render the Auto Rubric sidebar configuration."""
        return render_rubric_sidebar()

    def render_main_content(self, sidebar_config: dict[str, Any]) -> None:
        """Render the main content area for Auto Rubric."""
        self._init_session_state()

        # Tab navigation
        tab_new, tab_history, tab_help = st.tabs(
            [
                f"🆕 {t('rubric.tabs.new')}",
                f"📜 {t('rubric.tabs.history')}",
                f"❓ {t('rubric.tabs.help')}",
            ]
        )

        with tab_new:
            self._render_new_rubric_view(sidebar_config)

        with tab_history:
            self._render_history_view()

        with tab_help:
            self._render_help_view()

    def _render_new_rubric_view(self, sidebar_config: dict[str, Any]) -> None:
        """Render the new rubric generation view."""
        # Mode selection
        selected_mode = self._render_mode_selector()

        st.markdown(
            '<div class="custom-divider" style="margin: 1rem 0;"></div>',
            unsafe_allow_html=True,
        )

        # Two-column layout
        col_config, col_result = st.columns([1, 1], gap="large")

        with col_config:
            if selected_mode == "simple":
                config = render_simple_config_panel(sidebar_config)
                is_valid, validation_msg = validate_simple_config(config)
            else:
                config = render_iterative_config_panel(sidebar_config)
                is_valid, validation_msg = validate_iterative_config(config)

            st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

            if not is_valid:
                st.warning(validation_msg)

            generate_clicked = st.button(
                f"🚀 {t('rubric.config.generate')}",
                type="primary",
                use_container_width=True,
                disabled=not is_valid,
            )

        with col_result:
            result_placeholder = st.empty()

            result = st.session_state.get(self.STATE_RESULT)
            stored_config = st.session_state.get(self.STATE_CONFIG)
            stored_grader = st.session_state.get(self.STATE_GRADER)

            with result_placeholder.container():
                render_result_panel(
                    result=result,
                    config=stored_config,
                    grader=stored_grader,
                )

        if generate_clicked:
            if selected_mode == "simple":
                self._start_simple_generation(config, result_placeholder)
            else:
                self._start_iterative_generation(config, result_placeholder)

    def _render_mode_selector(self) -> str:
        """Render the generation mode selector and return selected mode."""
        st.markdown(
            f"""
            <div style="
                font-weight: 600;
                color: #F1F5F9;
                margin-bottom: 0.75rem;
            ">{t('rubric.mode.title')}</div>
            """,
            unsafe_allow_html=True,
        )

        # Get current mode from session state
        current_mode = st.session_state.get(self.STATE_MODE, "simple")

        col1, col2 = st.columns(2)

        with col1:
            simple_selected = current_mode == "simple"
            if st.button(
                f"⚡ {t('rubric.mode.simple')}",
                key="mode_simple",
                use_container_width=True,
                type="primary" if simple_selected else "secondary",
            ):
                st.session_state[self.STATE_MODE] = "simple"
                st.rerun()

            st.markdown(
                f"<div style='font-size: 0.8rem; color: #94A3B8; text-align: center;'>"
                f"{t('rubric.mode.simple_desc')}</div>",
                unsafe_allow_html=True,
            )

        with col2:
            iterative_selected = current_mode == "iterative"
            if st.button(
                f"📊 {t('rubric.mode.iterative')}",
                key="mode_iterative",
                use_container_width=True,
                type="primary" if iterative_selected else "secondary",
            ):
                st.session_state[self.STATE_MODE] = "iterative"
                st.rerun()

            st.markdown(
                f"<div style='font-size: 0.8rem; color: #94A3B8; text-align: center;'>"
                f"{t('rubric.mode.iterative_desc')}</div>",
                unsafe_allow_html=True,
            )

        return st.session_state.get(self.STATE_MODE, "simple")

    def _clear_test_state(self) -> None:
        """Clear all test-related session state.

        This ensures a clean test panel when generating a new grader,
        preventing stale inputs/results from previous sessions.
        """
        # Clear test results and running state
        st.session_state["rubric_test_result"] = None
        st.session_state["rubric_test_running"] = False

        # Clear listwise test inputs
        response_count = st.session_state.get("rubric_test_response_count", 0)
        for i in range(response_count):
            key = f"rubric_test_response_{i}"
            if key in st.session_state:
                del st.session_state[key]
        if "rubric_test_response_count" in st.session_state:
            del st.session_state["rubric_test_response_count"]

        # Clear compact test inputs (used in expander)
        keys_to_clear = [
            "rubric_test_query_compact",
            "rubric_test_response_compact",
            "rubric_test_response_compact_1",
            "rubric_test_response_compact_2",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    def _start_simple_generation(self, config: dict[str, Any], result_placeholder: Any) -> None:
        """Start Simple Rubric generation."""
        st.session_state[self.STATE_RESULT] = None
        st.session_state[self.STATE_CONFIG] = None
        st.session_state[self.STATE_GRADER] = None
        # Clear all test state to avoid showing stale data from previous grader
        self._clear_test_state()

        generation_success = False

        with result_placeholder.container():
            with st.status(f"🔄 {t('rubric.result.generating')}", expanded=True) as status:
                try:
                    st.write(f"**{t('rubric.result.init_model')}**")

                    grader_mode = GraderMode.POINTWISE if config["grader_mode"] == "pointwise" else GraderMode.LISTWISE
                    language = LanguageEnum.ZH if config["language"] == "ZH" else LanguageEnum.EN

                    service_config = SimpleRubricConfig(
                        grader_name=config["grader_name"],
                        task_description=config["task_description"],
                        scenario=config.get("scenario"),
                        sample_queries=config.get("sample_queries"),
                        grader_mode=grader_mode,
                        language=language,
                        min_score=config.get("min_score", 0),
                        max_score=config.get("max_score", 5),
                        max_retries=config.get("max_retries", 3),
                        api_endpoint=config["api_endpoint"],
                        api_key=config["api_key"],
                        model_name=config["model_name"],
                    )

                    st.write(f"**{t('rubric.result.calling_api')}**")

                    service = RubricGeneratorService()
                    result = run_async(service.generate_simple(service_config))

                    st.write(f"**{t('rubric.result.processing')}**")

                    if result.success:
                        st.session_state[self.STATE_RESULT] = {
                            "success": True,
                            "rubrics": result.rubrics,
                            "grader_config": result.grader_config,
                        }
                        st.session_state[self.STATE_CONFIG] = result.grader_config
                        st.session_state[self.STATE_GRADER] = result.grader

                        # Save to history
                        self._save_to_history(config, result.rubrics, result.grader_config, "simple")

                        status.update(label=f"✅ {t('rubric.result.success')}", state="complete")
                        generation_success = True
                    else:
                        st.session_state[self.STATE_RESULT] = {
                            "success": False,
                            "error": result.error,
                        }
                        status.update(label=f"❌ {t('rubric.result.failed')}", state="error")
                        st.error(t("rubric.result.error", error=result.error))

                except Exception as e:
                    st.session_state[self.STATE_RESULT] = {
                        "success": False,
                        "error": str(e),
                    }
                    status.update(label=f"❌ {t('rubric.result.failed')}", state="error")
                    st.error(t("rubric.result.error", error=str(e)))

        if generation_success:
            st.rerun()

    def _start_iterative_generation(self, config: dict[str, Any], result_placeholder: Any) -> None:
        """Start Iterative Rubric generation."""
        st.session_state[self.STATE_RESULT] = None
        st.session_state[self.STATE_CONFIG] = None
        st.session_state[self.STATE_GRADER] = None
        # Clear all test state to avoid showing stale data from previous grader
        self._clear_test_state()

        generation_success = False

        with result_placeholder.container():
            with st.status(f"🔄 {t('rubric.result.generating')}", expanded=True) as status:
                try:
                    st.write(f"**{t('rubric.result.init_model')}**")

                    grader_mode = GraderMode.POINTWISE if config["grader_mode"] == "pointwise" else GraderMode.LISTWISE
                    language = LanguageEnum.ZH if config["language"] == "ZH" else LanguageEnum.EN

                    service_config = IterativeRubricConfig(
                        grader_name=config["grader_name"],
                        dataset=config["dataset"],
                        grader_mode=grader_mode,
                        task_description=config.get("task_description"),
                        language=language,
                        min_score=config.get("min_score", 0),
                        max_score=config.get("max_score", 5),
                        enable_categorization=config.get("enable_categorization", True),
                        categories_number=config.get("categories_number", 5),
                        query_specific_generate_number=config.get("query_specific_generate_number", 2),
                        max_retries=config.get("max_retries", 3),
                        api_endpoint=config["api_endpoint"],
                        api_key=config["api_key"],
                        model_name=config["model_name"],
                    )

                    data_count = config.get("data_count", 0)

                    # Estimate time based on data count
                    # Rough estimate: ~3 seconds per sample for generation
                    estimated_minutes = max(1, (data_count * 3) // 60)

                    # Show detailed progress info with spinner
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(99, 102, 241, 0.1);
                            border: 1px solid rgba(99, 102, 241, 0.3);
                            border-radius: 8px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                        ">
                            <div style="color: #A5B4FC; font-weight: 600; margin-bottom: 0.5rem;">
                                📊 {t('rubric.iterative.generating_rubrics')}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.9rem;">
                                • {t('rubric.iterative.data_count')}: \
<strong style="color: #F1F5F9;">{data_count}</strong>
                            </div>
                            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.5rem;">
                                💡 {t('rubric.iterative.time_estimate_mins', minutes=estimated_minutes)}
                            </div>
                            <div style="color: #F59E0B; font-size: 0.8rem; margin-top: 0.75rem; \
padding-top: 0.5rem; border-top: 1px solid rgba(99, 102, 241, 0.2);">
                                ⚠️ {t('rubric.iterative.no_cancel_warning')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    service = RubricGeneratorService()

                    # Use spinner with dynamic text for better UX
                    import time

                    start_time = time.time()
                    current_stage = {"name": t("rubric.result.init_model"), "pct": 0}

                    def on_progress(stage: str, pct: float) -> None:
                        stage_names = {
                            "init": t("rubric.result.init_model"),
                            "generating": t("rubric.iterative.generating_rubrics"),
                            "processing": t("rubric.result.processing"),
                            "complete": t("rubric.result.success"),
                        }
                        current_stage["name"] = stage_names.get(stage, stage)
                        current_stage["pct"] = pct

                    # Show animated spinner during generation
                    with st.spinner(
                        f"🔄 {t('rubric.iterative.generating_rubrics')} {t('rubric.iterative.please_wait')}"
                    ):
                        result = run_async(service.generate_iterative(service_config, on_progress))

                    # Show elapsed time after completion
                    elapsed = time.time() - start_time
                    elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                    st.caption(f"⏱️ {t('rubric.iterative.total_time')}: {elapsed_str}")

                    if result.success:
                        st.session_state[self.STATE_RESULT] = {
                            "success": True,
                            "rubrics": result.rubrics,
                            "grader_config": result.grader_config,
                        }
                        st.session_state[self.STATE_CONFIG] = result.grader_config
                        st.session_state[self.STATE_GRADER] = result.grader

                        # Save to history
                        self._save_to_history(
                            config,
                            result.rubrics,
                            result.grader_config,
                            "iterative",
                            data_count=config.get("data_count"),
                        )

                        status.update(label=f"✅ {t('rubric.result.success')}", state="complete")
                        generation_success = True
                    else:
                        st.session_state[self.STATE_RESULT] = {
                            "success": False,
                            "error": result.error,
                        }
                        status.update(label=f"❌ {t('rubric.result.failed')}", state="error")
                        st.error(t("rubric.result.error", error=result.error))

                except Exception as e:
                    st.session_state[self.STATE_RESULT] = {
                        "success": False,
                        "error": str(e),
                    }
                    status.update(label=f"❌ {t('rubric.result.failed')}", state="error")
                    st.error(t("rubric.result.error", error=str(e)))

        if generation_success:
            st.rerun()

    def _save_to_history(
        self,
        config: dict[str, Any],
        rubrics: str,
        grader_config: dict[str, Any],
        mode: str,
        data_count: int | None = None,
    ) -> None:
        """Save generated grader to history."""
        try:
            history_manager = HistoryManager()
            task_id = history_manager.generate_task_id()
            history_manager.save_grader(
                task_id=task_id,
                config=config,
                rubrics=rubrics,
                grader_config=grader_config,
                mode=mode,
                data_count=data_count,
            )
        except Exception as e:
            # Don't fail the generation if history save fails
            st.warning(f"Failed to save to history: {e}")

    def _render_history_view(self) -> None:
        """Render the history view."""
        viewing_task = st.session_state.get(self.STATE_VIEWING_TASK)

        if viewing_task:
            render_task_detail(
                task_id=viewing_task,
                on_back=self._on_back_from_detail,
            )
        else:
            render_history_panel(
                on_view=self._on_view_task,
                on_delete=self._on_delete_task,
                limit=20,
            )

    def _on_view_task(self, task_id: str) -> None:
        """Handle view task button click."""
        st.session_state[self.STATE_VIEWING_TASK] = task_id
        st.rerun()

    def _on_delete_task(self, task_id: str) -> None:
        """Handle delete task button click."""
        history_manager = HistoryManager()
        if history_manager.delete_task(task_id):
            st.success(t("rubric.history.deleted", task_id=task_id))
            st.rerun()
        else:
            st.error(t("rubric.history.delete_failed"))

    def _on_back_from_detail(self) -> None:
        """Handle back button from task detail."""
        st.session_state[self.STATE_VIEWING_TASK] = None
        st.rerun()

    def _render_help_view(self) -> None:
        """Render the help view with quick start guide."""
        st.markdown(
            f"""
            <div class="feature-card">
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
                    {t("rubric.help.title")}
                </div>
                <div style="color: #94A3B8; margin-bottom: 1rem;">
                    {t("rubric.help.overview")}
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Simple Rubric steps
        st.markdown(
            f"""
                <div style="color: #A5B4FC; font-weight: 500; margin-bottom: 0.5rem;">
                    ⚡ {t("rubric.help.simple_title")}
                </div>
                <div class="guide-step">
                    <div class="guide-number">1</div>
                    <div class="guide-text">{t("rubric.help.simple_step1")}</div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">2</div>
                    <div class="guide-text">{t("rubric.help.simple_step2")}</div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">3</div>
                    <div class="guide-text">{t("rubric.help.simple_step3")}</div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">4</div>
                    <div class="guide-text">{t("rubric.help.simple_step4")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Iterative Rubric steps
        st.markdown(
            f"""
            <div class="feature-card" style="margin-top: 1rem;">
                <div style="color: #34D399; font-weight: 500; margin-bottom: 0.5rem;">
                    📊 {t("rubric.help.iterative_title")}
                </div>
                <div class="guide-step">
                    <div class="guide-number">1</div>
                    <div class="guide-text">{t("rubric.help.iterative_step1")}</div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">2</div>
                    <div class="guide-text">{t("rubric.help.iterative_step2")}</div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">3</div>
                    <div class="guide-text">{t("rubric.help.iterative_step3")}</div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">4</div>
                    <div class="guide-text">{t("rubric.help.iterative_step4")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tips
        st.markdown(
            f"""
            <div class="feature-card" style="margin-top: 1rem;">
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
                    💡 {t("rubric.help.tips_title")}
                </div>
                <ul style="color: #94A3B8; margin: 0; padding-left: 1.25rem;">
                    <li>{t("rubric.help.tip1")}</li>
                    <li>{t("rubric.help.tip2")}</li>
                    <li>{t("rubric.help.tip3")}</li>
                    <li>{t("rubric.help.tip4")}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _init_session_state(self) -> None:
        """Initialize session state variables."""
        defaults = {
            self.STATE_RESULT: None,
            self.STATE_CONFIG: None,
            self.STATE_GRADER: None,
            self.STATE_MODE: "simple",
            self.STATE_VIEWING_TASK: None,
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    def on_mount(self) -> None:
        """Initialize Auto Rubric feature state when mounted."""
        self._init_session_state()

    def on_unmount(self) -> None:
        """Cleanup when feature is unmounted."""
