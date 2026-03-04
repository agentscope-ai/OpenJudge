# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""Paper Review feature implementation for OpenJudge Studio.

Supports PDF paper review and bibliography verification.
"""

import uuid
from typing import Any, Optional

import streamlit as st
from core.base_feature import BaseFeature
from features.paper_review.components.batch_panel import (
    render_batch_progress,
    render_batch_results,
)
from features.paper_review.components.history_panel import (
    render_history_detail,
    render_history_stats,
)
from features.paper_review.components.progress_panel import (
    render_compact_progress,
    render_progress_panel,
)
from features.paper_review.components.result_panel import render_result_panel
from features.paper_review.services.batch_runner import (
    BatchPaperItem,
    BatchProgress,
    BatchResult,
    BatchRunner,
    generate_batch_csv_report,
)
from features.paper_review.services.history_service import (
    HistoryEntry,
    get_history_service,
)
from features.paper_review.services.pipeline_runner import (
    PipelineRunner,
    ReviewTaskResult,
    create_task_config_from_sidebar,
)
from shared.i18n import t

from cookbooks.paper_review import ReviewProgress
from cookbooks.paper_review.processors import BibChecker
from cookbooks.paper_review.schema import VerificationStatus


class PaperReviewFeature(BaseFeature):
    """Paper Review feature.

    Provides UI for AI-powered academic paper review including:
    - PDF paper review (safety, correctness, review, criticality)
    - Bibliography verification
    - Batch paper review
    - Review history management
    """

    feature_id = "paper_review"
    feature_name = "Paper Review"  # Default/fallback name
    feature_icon = "📄"
    feature_description = "AI-powered academic paper review"  # Default/fallback
    order = 3

    # Session state keys
    STATE_PROGRESS = "paper_review_progress"
    STATE_RESULT = "paper_review_result"
    STATE_VIEWING_TASK = "paper_review_viewing_task"
    STATE_CURRENT_TAB = "paper_review_current_tab"
    STATE_BATCH_PROGRESS = "paper_review_batch_progress"
    STATE_BATCH_RESULTS = "paper_review_batch_results"
    STATE_BATCH_TASK_ID = "paper_review_batch_task_id"
    STATE_BIB_RESULT = "paper_review_bib_result"

    @property
    def display_label(self) -> str:
        """Get the display label for navigation with i18n support."""
        return f"{self.feature_icon} {t('paper_review.name')}"

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
                    <span>{t('paper_review.name')}</span>
                </h1>
                <p style="color: #94A3B8; margin: 0.25rem 0 0 0; font-size: 0.95rem;">
                    {t('paper_review.description')}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    def render_sidebar(self) -> dict[str, Any]:
        """Render the paper review sidebar configuration.

        Returns:
            Dictionary containing all sidebar configuration values
        """
        # TODO: Implement sidebar component
        config: dict[str, Any] = {}

        st.markdown(
            f'<div class="section-header">{t("paper_review.sidebar.api_settings")}</div>',
            unsafe_allow_html=True,
        )

        # API Endpoint (optional)
        api_endpoint = st.text_input(
            t("paper_review.sidebar.custom_endpoint"),
            placeholder="https://api.openai.com/v1",
            help=t("paper_review.sidebar.endpoint_help"),
        )
        config["api_endpoint"] = api_endpoint if api_endpoint else None

        # API Key
        api_key = st.text_input(
            t("paper_review.sidebar.api_key"),
            type="password",
            placeholder=t("paper_review.sidebar.api_key_placeholder"),
        )
        config["api_key"] = api_key

        if api_key:
            st.success(t("paper_review.sidebar.api_configured"))
        else:
            st.warning(t("paper_review.sidebar.api_required"))

        # Model settings
        st.markdown(
            f'<div class="section-header">{t("paper_review.sidebar.model_settings")}</div>',
            unsafe_allow_html=True,
        )

        # Model selection with custom option
        preset_models = ["gpt-5.2", "gpt-5.1", "gemini-3-pro-preview", "Custom..."]
        selected_model = st.selectbox(
            t("paper_review.sidebar.model"),
            options=preset_models,
            index=0,
        )

        # If custom selected, show text input
        if selected_model == "Custom...":
            model_name = st.text_input(
                t("paper_review.sidebar.custom_model"),
                placeholder="e.g., claude-3-5-sonnet, qwen-max",
                help=t("paper_review.sidebar.custom_model_help"),
            )
        else:
            model_name = selected_model
        config["model_name"] = model_name

        # Temperature
        temperature = st.slider(
            t("paper_review.sidebar.temperature"),
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help=t("paper_review.sidebar.temperature_help"),
        )
        config["temperature"] = temperature

        # Review options
        st.markdown(
            f'<div class="section-header">{t("paper_review.sidebar.review_options")}</div>',
            unsafe_allow_html=True,
        )

        config["enable_safety_checks"] = st.checkbox(
            t("paper_review.sidebar.enable_safety"),
            value=True,
            help=t("paper_review.sidebar.enable_safety_help"),
        )

        config["enable_correctness"] = st.checkbox(
            t("paper_review.sidebar.enable_correctness"),
            value=True,
            help=t("paper_review.sidebar.enable_correctness_help"),
        )

        config["enable_review"] = st.checkbox(
            t("paper_review.sidebar.enable_review"),
            value=True,
            help=t("paper_review.sidebar.enable_review_help"),
        )

        config["enable_criticality"] = st.checkbox(
            t("paper_review.sidebar.enable_criticality"),
            value=True,
            help=t("paper_review.sidebar.enable_criticality_help"),
        )

        config["enable_bib_verification"] = st.checkbox(
            t("paper_review.sidebar.enable_bib"),
            value=True,
            help=t("paper_review.sidebar.enable_bib_help"),
        )

        # CrossRef email (optional)
        with st.expander(t("paper_review.sidebar.advanced")):
            crossref_email = st.text_input(
                t("paper_review.sidebar.crossref_email"),
                placeholder="your@email.com",
                help=t("paper_review.sidebar.crossref_email_help"),
            )
            config["crossref_mailto"] = crossref_email if crossref_email else None

        return config

    def render_main_content(self, sidebar_config: dict[str, Any]) -> None:
        """Render the main content area for paper review.

        Args:
            sidebar_config: Configuration from the sidebar
        """
        # Initialize session state
        self._init_session_state()

        # Tab navigation
        tab_single, tab_bib, tab_batch, tab_history, tab_help = st.tabs(
            [
                f"📄 {t('paper_review.tabs.single')}",
                f"📚 {t('paper_review.tabs.bib_check')}",
                f"📦 {t('paper_review.tabs.batch')}",
                f"📜 {t('paper_review.tabs.history')}",
                f"❓ {t('paper_review.tabs.help')}",
            ]
        )

        with tab_single:
            if st.session_state.get(self.STATE_CURRENT_TAB) != 0:
                st.session_state[self.STATE_CURRENT_TAB] = 0
            self._render_single_review(sidebar_config)

        with tab_bib:
            if st.session_state.get(self.STATE_CURRENT_TAB) != 1:
                st.session_state[self.STATE_CURRENT_TAB] = 1
            self._render_bib_only_check(sidebar_config)

        with tab_batch:
            if st.session_state.get(self.STATE_CURRENT_TAB) != 2:
                st.session_state[self.STATE_CURRENT_TAB] = 2
            self._render_batch_review(sidebar_config)

        with tab_history:
            self._render_history_view()

        with tab_help:
            self._render_help()

    def _render_single_review(self, sidebar_config: dict[str, Any]) -> None:
        """Render single paper review view."""
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown(
                f"### {t('paper_review.single.upload_title')}",
            )

            # PDF upload
            uploaded_pdf = st.file_uploader(
                t("paper_review.single.upload_pdf"),
                type=["pdf"],
                help=t("paper_review.single.upload_pdf_help"),
                key="paper_review_pdf_uploader",
            )

            # BIB upload (optional)
            uploaded_bib = st.file_uploader(
                t("paper_review.single.upload_bib"),
                type=["bib"],
                help=t("paper_review.single.upload_bib_help"),
                key="paper_review_bib_uploader",
            )

            # Paper name
            paper_name = st.text_input(
                t("paper_review.single.paper_name"),
                placeholder=t("paper_review.single.paper_name_placeholder"),
                key="paper_review_paper_name",
            )

            # Validation
            can_start = uploaded_pdf is not None and sidebar_config.get("api_key")

            # Show validation messages
            if not can_start:
                if not uploaded_pdf:
                    st.caption(f"⚠️ {t('paper_review.single.missing_pdf')}")
                if not sidebar_config.get("api_key"):
                    st.caption(f"⚠️ {t('paper_review.single.missing_api_key')}")

            # Start button
            start_clicked = st.button(
                f"🚀 {t('paper_review.single.start')}",
                type="primary",
                use_container_width=True,
                disabled=not can_start,
                key="paper_review_start_btn",
            )

            # Clear result button (if there's a result)
            if st.session_state.get(self.STATE_RESULT):
                if st.button(
                    "🗑️ Clear Result",
                    use_container_width=True,
                    key="paper_review_clear_btn",
                ):
                    st.session_state[self.STATE_RESULT] = None
                    st.session_state[self.STATE_PROGRESS] = None
                    st.rerun()

        with col_result:
            st.markdown(
                f"### {t('paper_review.single.result_title')}",
            )

            # Get current state
            task_result: Optional[ReviewTaskResult] = st.session_state.get(self.STATE_RESULT)
            progress: Optional[ReviewProgress] = st.session_state.get(self.STATE_PROGRESS)

            if task_result:
                # Show completed result
                render_result_panel(task_result)
            elif progress:
                # Show progress
                render_progress_panel(progress)
            else:
                # Empty state
                render_progress_panel(None)

        # Handle start button click
        if start_clicked and uploaded_pdf:
            self._start_single_review(
                sidebar_config,
                uploaded_pdf,
                uploaded_bib,
                paper_name or "Paper",
            )

    def _start_single_review(
        self,
        sidebar_config: dict[str, Any],
        uploaded_pdf: Any,
        uploaded_bib: Any,
        paper_name: str,
    ) -> None:
        """Start single paper review.

        Args:
            sidebar_config: Configuration from sidebar
            uploaded_pdf: Uploaded PDF file object
            uploaded_bib: Optional uploaded BIB file object
            paper_name: Name of the paper for the report
        """
        # Clear previous results
        st.session_state[self.STATE_RESULT] = None
        st.session_state[self.STATE_PROGRESS] = None

        # Read file contents
        pdf_bytes = uploaded_pdf.read()
        bib_content = uploaded_bib.read().decode("utf-8") if uploaded_bib else None

        # Create task config
        task_config = create_task_config_from_sidebar(sidebar_config)
        task_config.paper_name = paper_name

        # Progress callback to update session state
        def on_progress(progress: ReviewProgress) -> None:
            st.session_state[self.STATE_PROGRESS] = progress

        # Create runner
        runner = PipelineRunner(task_config, progress_callback=on_progress)

        # Run with status display
        with st.status(
            f"🔄 {t('paper_review.single.in_progress')}",
            expanded=True,
        ) as status:
            try:
                # Show initial info
                st.write(f"**{t('paper_review.single.paper_name')}:** {paper_name}")
                st.write(f"**{t('paper_review.sidebar.model')}:** {task_config.model_name}")
                st.write("---")

                # Create progress display placeholder
                progress_text = st.empty()
                progress_bar = st.empty()

                # Update progress display
                def update_progress_display(progress: ReviewProgress) -> None:
                    pct = progress.progress_percent / 100
                    progress_text.write(render_compact_progress(progress))
                    progress_bar.progress(pct)

                # Override callback to also update display
                original_callback = on_progress

                def combined_callback(progress: ReviewProgress) -> None:
                    original_callback(progress)
                    update_progress_display(progress)

                runner._progress_callback = combined_callback  # pylint: disable=protected-access

                # Run the pipeline
                task_result = runner.run(pdf_bytes, bib_content)

                # Clear progress display
                progress_text.empty()
                progress_bar.empty()

                # Store result
                st.session_state[self.STATE_RESULT] = task_result

                # Save to history if successful
                if task_result.success:
                    try:
                        history_service = get_history_service()
                        history_service.save(task_result)
                    except Exception:
                        pass  # Don't fail review for history errors

                # Update status
                if task_result.success:
                    status.update(
                        label=f"✅ {t('paper_review.single.completed')} ({task_result.elapsed_time_display})",
                        state="complete",
                    )
                    st.write("---")
                    st.write(f"✅ **{t('paper_review.single.completed')}**")

                    # Show summary
                    if task_result.result:
                        result = task_result.result
                        if result.review:
                            st.write(f"📝 **{t('paper_review.result.review_score')}:** {result.review.score}/6")
                        if result.correctness:
                            display_score = 4 - result.correctness.score
                            st.write(f"🔍 **{t('paper_review.result.correctness_score')}:** {display_score}/3")
                else:
                    status.update(
                        label=f"❌ {t('paper_review.progress.failed')}",
                        state="error",
                    )
                    st.error(task_result.error or "Unknown error")

            except Exception as e:
                status.update(
                    label=f"❌ {t('paper_review.progress.failed')}",
                    state="error",
                )
                st.error(t("paper_review.error.pipeline_failed", error=str(e)))

                # Store error result
                st.session_state[self.STATE_RESULT] = ReviewTaskResult(
                    success=False,
                    error=str(e),
                    paper_name=paper_name,
                )

        # Rerun to show results
        st.rerun()

    def _render_bib_only_check(self, sidebar_config: dict[str, Any]) -> None:
        """Render standalone BIB verification view."""
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown(f"### {t('paper_review.bib_check.upload_title')}")

            # BIB upload
            uploaded_bib = st.file_uploader(
                t("paper_review.bib_check.upload_bib"),
                type=["bib"],
                help=t("paper_review.bib_check.upload_bib_help"),
                key="paper_review_bib_only_uploader",
            )

            # CrossRef email (optional, from sidebar)
            crossref_email = sidebar_config.get("crossref_mailto")

            # Validation
            can_start = uploaded_bib is not None

            if not can_start:
                st.caption(f"⚠️ {t('paper_review.bib_check.missing_bib')}")

            # Start button
            start_clicked = st.button(
                f"🔍 {t('paper_review.bib_check.start')}",
                type="primary",
                use_container_width=True,
                disabled=not can_start,
                key="paper_review_bib_only_start_btn",
            )

            # Clear result button
            if st.session_state.get(self.STATE_BIB_RESULT):
                if st.button(
                    "🗑️ Clear Result",
                    use_container_width=True,
                    key="paper_review_bib_only_clear_btn",
                ):
                    st.session_state[self.STATE_BIB_RESULT] = None
                    st.rerun()

        with col_result:
            st.markdown(f"### {t('paper_review.bib_check.result_title')}")

            bib_result = st.session_state.get(self.STATE_BIB_RESULT)

            if bib_result:
                self._render_bib_result(bib_result)
            else:
                st.markdown(
                    f"""<div style="
                        text-align: center;
                        padding: 3rem;
                        color: #64748B;
                        border: 2px dashed #334155;
                        border-radius: 8px;
                    ">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">📚</div>
                        <div>{t('paper_review.bib_check.empty_state')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Handle start button click
        if start_clicked and uploaded_bib:
            self._start_bib_check(uploaded_bib, crossref_email)

    def _start_bib_check(self, uploaded_bib: Any, crossref_email: Optional[str]) -> None:
        """Start standalone BIB verification.

        Args:
            uploaded_bib: Uploaded BIB file object
            crossref_email: Optional CrossRef email for better rate limits
        """
        bib_content = uploaded_bib.read().decode("utf-8")

        with st.status(
            f"🔍 {t('paper_review.bib_check.in_progress')}",
            expanded=True,
        ) as status:
            try:
                st.write(f"**{t('paper_review.bib_check.parsing')}**")

                checker = BibChecker(mailto=crossref_email)
                references = checker.parse_bib_file(bib_content)

                st.write(f"Found **{len(references)}** references")
                st.write("---")

                # Progress
                progress_bar = st.progress(0)
                progress_text = st.empty()

                results = []
                for i, ref in enumerate(references):
                    progress_text.write(f"Checking: {ref.title[:50]}...")
                    result = checker.verify_reference(ref)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(references))

                progress_bar.empty()
                progress_text.empty()

                # Calculate summary
                verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
                suspect = sum(1 for r in results if r.status == VerificationStatus.SUSPECT)
                errors = sum(1 for r in results if r.status == VerificationStatus.ERROR)

                bib_result = {
                    "total": len(references),
                    "verified": verified,
                    "suspect": suspect,
                    "errors": errors,
                    "verification_rate": verified / len(references) if references else 0,
                    "results": results,
                }

                st.session_state[self.STATE_BIB_RESULT] = bib_result

                status.update(
                    label=f"✅ {t('paper_review.bib_check.completed')} ({verified}/{len(references)})",
                    state="complete",
                )

                checker.close()

            except Exception as e:
                status.update(
                    label=f"❌ {t('paper_review.progress.failed')}",
                    state="error",
                )
                st.error(str(e))

        st.rerun()

    def _render_bib_result(self, bib_result: dict) -> None:
        """Render BIB verification results.

        Args:
            bib_result: Dictionary containing verification results
        """
        total = bib_result["total"]
        verified = bib_result["verified"]
        suspect = bib_result["suspect"]
        errors = bib_result["errors"]
        rate = bib_result["verification_rate"]

        # Summary header
        rate_color = "#22C55E" if rate >= 0.8 else ("#F59E0B" if rate >= 0.5 else "#EF4444")

        st.markdown(
            f"""<div style="
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.25rem;
                margin-bottom: 1.5rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-weight: 600; color: #F1F5F9; font-size: 1.1rem;">
                            {t('paper_review.bib_check.summary')}
                        </div>
                        <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 0.25rem;">
                            {total} {t('paper_review.result.total_refs')}
                        </div>
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: {rate_color};">
                        {rate:.0%}
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"✅ {t('paper_review.result.verified')}", verified)
        with col2:
            st.metric(f"⚠️ {t('paper_review.result.suspect')}", suspect)
        with col3:
            st.metric(f"❌ {t('paper_review.result.errors')}", errors)

        # Results list
        st.markdown("---")
        results = bib_result.get("results", [])

        # Filter tabs
        filter_tab = st.radio(
            "Filter",
            ["All", "Verified", "Suspect", "Errors"],
            horizontal=True,
            key="bib_result_filter",
        )

        for result in results:
            if filter_tab == "Verified" and result.status != VerificationStatus.VERIFIED:
                continue
            if filter_tab == "Suspect" and result.status != VerificationStatus.SUSPECT:
                continue
            if filter_tab == "Errors" and result.status != VerificationStatus.ERROR:
                continue

            self._render_bib_ref_item(result)

    def _render_bib_ref_item(self, result: Any) -> None:
        """Render a single reference verification result."""
        ref = result.reference
        status = result.status

        if status == VerificationStatus.VERIFIED:
            icon = "✅"
            color = "#22C55E"
            bg = "rgba(34, 197, 94, 0.05)"
        elif status == VerificationStatus.SUSPECT:
            icon = "⚠️"
            color = "#F59E0B"
            bg = "rgba(245, 158, 11, 0.05)"
        else:
            icon = "❌"
            color = "#EF4444"
            bg = "rgba(239, 68, 68, 0.05)"

        title = ref.title[:80] + "..." if len(ref.title) > 80 else ref.title
        authors = ref.authors[:50] + "..." if ref.authors and len(ref.authors) > 50 else (ref.authors or "")

        with st.container():
            st.markdown(
                f"""<div style="
                    background: {bg};
                    border-left: 3px solid {color};
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem;
                    border-radius: 0 4px 4px 0;
                ">
                    <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
                        <span style="font-size: 1rem;">{icon}</span>
                        <div style="flex: 1;">
                            <div style="font-weight: 500; color: #F1F5F9; font-size: 0.9rem;">
                                {title}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.25rem;">
                                {authors} {f"({ref.year})" if ref.year else ""}
                            </div>
                            <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.25rem;">
                                {result.message}
                            </div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    def _render_batch_review(self, sidebar_config: dict[str, Any]) -> None:
        """Render batch paper review view."""
        col_upload, col_progress = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown(f"### {t('paper_review.batch.upload_title')}")

            # Multiple PDF upload
            uploaded_files = st.file_uploader(
                t("paper_review.batch.upload_help"),
                type=["pdf"],
                accept_multiple_files=True,
                key="paper_review_batch_uploader",
            )

            # Concurrency setting
            max_concurrency = st.slider(
                t("paper_review.batch.max_concurrency"),
                min_value=1,
                max_value=3,
                value=1,
                help=t("paper_review.batch.max_concurrency_help"),
                key="paper_review_batch_concurrency",
            )

            # Show uploaded files
            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} files selected:**")
                for f in uploaded_files[:5]:
                    st.caption(f"📄 {f.name}")
                if len(uploaded_files) > 5:
                    st.caption(f"... and {len(uploaded_files) - 5} more")

            # Validation
            can_start = len(uploaded_files) > 0 and sidebar_config.get("api_key")

            if not can_start:
                if not uploaded_files:
                    st.caption(f"⚠️ {t('paper_review.single.missing_pdf')}")
                if not sidebar_config.get("api_key"):
                    st.caption(f"⚠️ {t('paper_review.single.missing_api_key')}")

            # Start button
            start_clicked = st.button(
                f"🚀 {t('paper_review.batch.start')}",
                type="primary",
                use_container_width=True,
                disabled=not can_start,
                key="paper_review_batch_start_btn",
            )

            # Clear button
            batch_result: Optional[BatchResult] = st.session_state.get(self.STATE_BATCH_RESULTS)
            if batch_result:
                if st.button(
                    "🗑️ Clear Results",
                    use_container_width=True,
                    key="paper_review_batch_clear_btn",
                ):
                    st.session_state[self.STATE_BATCH_RESULTS] = None
                    st.session_state[self.STATE_BATCH_PROGRESS] = None
                    st.rerun()

        with col_progress:
            st.markdown(f"### {t('paper_review.batch.progress_title')}")

            # Get current state
            batch_progress: Optional[BatchProgress] = st.session_state.get(self.STATE_BATCH_PROGRESS)
            batch_result = st.session_state.get(self.STATE_BATCH_RESULTS)

            if batch_result:
                render_batch_results(batch_result)

                # Export buttons
                st.markdown("---")
                col1, _ = st.columns(2)
                with col1:
                    # Download all reports as zip would require zipfile, for now just CSV
                    csv_data = generate_batch_csv_report(batch_result.results)
                    st.download_button(
                        label=f"📊 {t('paper_review.batch.export_csv')}",
                        data=csv_data,
                        file_name="batch_review_summary.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="paper_review_batch_export_csv",
                    )
            elif batch_progress:
                render_batch_progress(batch_progress)
            else:
                render_batch_progress(None)

        # Handle start button click
        if start_clicked and uploaded_files:
            self._start_batch_review(
                sidebar_config,
                uploaded_files,
                max_concurrency,
            )

    def _start_batch_review(
        self,
        sidebar_config: dict[str, Any],
        uploaded_files: list[Any],
        max_concurrency: int,
    ) -> None:
        """Start batch paper review.

        Args:
            sidebar_config: Configuration from sidebar
            uploaded_files: List of uploaded PDF files
            max_concurrency: Maximum concurrent reviews
        """
        # Clear previous results
        st.session_state[self.STATE_BATCH_RESULTS] = None
        st.session_state[self.STATE_BATCH_PROGRESS] = None

        # Prepare paper items
        papers: list[BatchPaperItem] = []
        for f in uploaded_files:
            paper_name = f.name.rsplit(".", 1)[0]  # Remove .pdf extension
            papers.append(
                BatchPaperItem(
                    paper_id=str(uuid.uuid4()),
                    paper_name=paper_name,
                    pdf_bytes=f.read(),
                )
            )

        # Create task config
        task_config = create_task_config_from_sidebar(sidebar_config)

        # Progress callback
        def on_batch_progress(progress: BatchProgress) -> None:
            st.session_state[self.STATE_BATCH_PROGRESS] = progress

        # Create runner
        runner = BatchRunner(task_config, progress_callback=on_batch_progress)

        # Run with status display
        with st.status(
            f"🔄 {t('paper_review.single.in_progress')}",
            expanded=True,
        ) as status:
            try:
                st.write(f"**Total Papers:** {len(papers)}")
                st.write(f"**Concurrency:** {max_concurrency}")
                st.write("---")

                # Progress display
                progress_text = st.empty()
                progress_bar = st.empty()

                def update_display(progress: BatchProgress) -> None:
                    pct = progress.progress_percent / 100
                    current = progress.current_paper or "-"
                    progress_text.write(f"📄 {current} ({progress.completed}/{progress.total})")
                    progress_bar.progress(pct)

                # Combined callback
                original_callback = on_batch_progress

                def combined_callback(progress: BatchProgress) -> None:
                    original_callback(progress)
                    update_display(progress)

                runner._progress_callback = combined_callback  # pylint: disable=protected-access

                # Run batch
                batch_result = runner.run(papers, max_concurrency)

                # Clear display
                progress_text.empty()
                progress_bar.empty()

                # Store result
                st.session_state[self.STATE_BATCH_RESULTS] = batch_result

                # Save successful results to history
                history_service = get_history_service()
                for result in batch_result.results:
                    if result.success:
                        try:
                            history_service.save(result)
                        except Exception:
                            pass  # Don't fail batch for history errors

                # Update status
                if batch_result.success:
                    status.update(
                        label=f"✅ Batch Complete ({batch_result.completed}/{batch_result.total})",
                        state="complete",
                    )
                else:
                    status.update(
                        label="⚠️ Batch Completed with Errors",
                        state="complete",
                    )

            except Exception as e:
                status.update(
                    label="❌ Batch Failed",
                    state="error",
                )
                st.error(str(e))

        st.rerun()

    def _render_history_view(self) -> None:
        """Render history view."""
        history_service = get_history_service()

        # Check if viewing a specific entry
        viewing_task_id = st.session_state.get(self.STATE_VIEWING_TASK)

        if viewing_task_id:
            self._render_history_detail_view(viewing_task_id)
            return

        # Header with stats
        st.markdown(f"### {t('paper_review.history.title')}")

        stats = history_service.get_summary_stats()
        render_history_stats(stats)

        st.markdown("---")

        # History list
        entries = history_service.list_all(limit=50)

        if not entries:
            st.markdown(
                f"""<div style="
                    text-align: center;
                    padding: 3rem;
                    color: #64748B;
                    border: 2px dashed #334155;
                    border-radius: 8px;
                ">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📜</div>
                    <div>{t('paper_review.history.no_history')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            return

        # Render list with actions
        for entry in entries:
            self._render_history_entry_row(entry)

    def _render_history_entry_row(self, entry: HistoryEntry) -> None:
        """Render a single history entry row with actions."""
        success = entry.success
        icon = "✅" if success else "❌"

        scores = []
        if entry.review_score is not None:
            scores.append(f"📝 {entry.review_score}/6")
        if entry.correctness_score is not None:
            scores.append(f"🔍 {entry.correctness_score}/3")
        score_text = " • ".join(scores) if scores else ""

        col1, col2, col3, col4 = st.columns([0.4, 3, 1.5, 1.5])

        with col1:
            st.markdown(f"<span style='font-size: 1.25rem;'>{icon}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{entry.paper_name}**")
            st.caption(f"{entry.created_at_display} • {score_text} • ⏱️ {entry.elapsed_time_display}")

        with col3:
            if st.button(
                f"👁️ {t('paper_review.history.view')}",
                key=f"view_{entry.task_id}",
                use_container_width=True,
            ):
                st.session_state[self.STATE_VIEWING_TASK] = entry.task_id
                st.rerun()

        with col4:
            if st.button(
                f"🗑️ {t('paper_review.history.delete')}",
                key=f"delete_{entry.task_id}",
                use_container_width=True,
            ):
                history_service = get_history_service()
                history_service.delete(entry.task_id)
                st.toast(t("paper_review.history.deleted"))
                st.rerun()

    def _render_history_detail_view(self, task_id: str) -> None:
        """Render detailed view of a history entry.

        Args:
            task_id: The task ID to display
        """
        history_service = get_history_service()
        entry = history_service.get(task_id)

        if not entry:
            st.error("History entry not found")
            st.session_state[self.STATE_VIEWING_TASK] = None
            st.rerun()
            return

        # Back button
        if st.button("← Back to History", key="history_back_btn"):
            st.session_state[self.STATE_VIEWING_TASK] = None
            st.rerun()

        st.markdown("---")

        # Render detail
        render_history_detail(entry)

    def _render_help(self) -> None:
        """Render help view."""
        st.markdown(
            f"""
            <div class="feature-card">
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
                    {t("paper_review.help.title")}
                </div>
                <div class="guide-step">
                    <div class="guide-number">1</div>
                    <div class="guide-text">
                        <strong>{t("paper_review.help.step1_title")}:</strong>
                        {t("paper_review.help.step1_desc")}
                    </div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">2</div>
                    <div class="guide-text">
                        <strong>{t("paper_review.help.step2_title")}:</strong>
                        {t("paper_review.help.step2_desc")}
                    </div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">3</div>
                    <div class="guide-text">
                        <strong>{t("paper_review.help.step3_title")}:</strong>
                        {t("paper_review.help.step3_desc")}
                    </div>
                </div>
                <div class="guide-step">
                    <div class="guide-number">4</div>
                    <div class="guide-text">
                        <strong>{t("paper_review.help.step4_title")}:</strong>
                        {t("paper_review.help.step4_desc")}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Review stages explanation
        st.markdown(
            f"""
            <div class="feature-card" style="margin-top: 1rem;">
                <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 0.75rem;">
                    {t("paper_review.help.stages_title")}
                </div>
                <div style="color: #94A3B8; font-size: 0.9rem;">
                    <p><strong>🛡️ {t("paper_review.help.stage_safety")}:</strong>
                    {t("paper_review.help.stage_safety_desc")}</p>
                    <p><strong>🔍 {t("paper_review.help.stage_correctness")}:</strong>
                    {t("paper_review.help.stage_correctness_desc")}</p>
                    <p><strong>📝 {t("paper_review.help.stage_review")}:</strong>
                    {t("paper_review.help.stage_review_desc")}</p>
                    <p><strong>⚠️ {t("paper_review.help.stage_criticality")}:</strong>
                    {t("paper_review.help.stage_criticality_desc")}</p>
                    <p><strong>📚 {t("paper_review.help.stage_bib")}:</strong>
                    {t("paper_review.help.stage_bib_desc")}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _init_session_state(self) -> None:
        """Initialize session state variables."""
        if self.STATE_PROGRESS not in st.session_state:
            st.session_state[self.STATE_PROGRESS] = None
        if self.STATE_RESULT not in st.session_state:
            st.session_state[self.STATE_RESULT] = None
        if self.STATE_VIEWING_TASK not in st.session_state:
            st.session_state[self.STATE_VIEWING_TASK] = None
        if self.STATE_CURRENT_TAB not in st.session_state:
            st.session_state[self.STATE_CURRENT_TAB] = 0
        if self.STATE_BATCH_PROGRESS not in st.session_state:
            st.session_state[self.STATE_BATCH_PROGRESS] = None
        if self.STATE_BATCH_RESULTS not in st.session_state:
            st.session_state[self.STATE_BATCH_RESULTS] = None
        if self.STATE_BATCH_TASK_ID not in st.session_state:
            st.session_state[self.STATE_BATCH_TASK_ID] = None
        if self.STATE_BIB_RESULT not in st.session_state:
            st.session_state[self.STATE_BIB_RESULT] = None

    def on_mount(self) -> None:
        """Initialize paper review feature state when mounted."""
        self._init_session_state()

    def on_unmount(self) -> None:
        """Cleanup when feature is unmounted."""
        # Could cancel running pipeline here if needed
