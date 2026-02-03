# -*- coding: utf-8 -*-
"""Data upload panel for Iterative Rubric mode.

Provides UI for uploading and previewing labeled training data.
"""

import html
import random
from typing import Any

import streamlit as st
from features.auto_rubric.services.data_parser import DataParser
from shared.i18n import t


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text)) if text else ""


def render_data_upload_panel(mode: str = "pointwise", required_fields: str = "") -> dict[str, Any]:
    """Render the data upload panel for Iterative Rubric mode.

    Args:
        mode: Evaluation mode ("pointwise" or "listwise").

    Returns:
        Dictionary containing:
        - is_valid: Whether valid data has been uploaded
        - data: Parsed data list (if valid), sampled if user selected fewer records
        - count: Number of records to use
        - total_count: Total number of records in file
        - min_score: Minimum score from data (pointwise mode only)
        - max_score: Maximum score from data (pointwise mode only)
    """
    result: dict[str, Any] = {
        "is_valid": False,
        "data": None,
        "count": 0,
        "total_count": 0,
        "min_score": None,
        "max_score": None,
    }

    # Show required fields hint if provided
    if required_fields:
        st.caption(f"📋 {t('rubric.upload.required_fields')}: `{required_fields}`")

    # File uploader
    uploaded_file = st.file_uploader(
        t("rubric.upload.label"),
        type=["json", "jsonl", "csv"],
        help=t("rubric.upload.help"),
        key="rubric_data_upload",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Parse the file
        parser = DataParser()
        content = uploaded_file.read()

        with st.spinner(t("rubric.upload.parsing")):
            parse_result = parser.parse_file(
                file_content=content,
                filename=uploaded_file.name,
                mode=mode,
            )

        if parse_result.success and parse_result.data:
            total_count = parse_result.total_count
            all_data = parse_result.data

            # Build info items
            info_items = [f"📄 {uploaded_file.name}", f"📊 {total_count} {t('rubric.upload.records')}"]

            # Add score range for pointwise mode
            if mode == "pointwise" and parse_result.min_score is not None and parse_result.max_score is not None:
                info_items.append(
                    f"🎯 {t('rubric.upload.score_range_label')}: {parse_result.min_score} - {parse_result.max_score}"
                )

            # Show success message with file info
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    border-radius: 8px;
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.75rem;
                ">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.25rem;">✅</span>
                        <span style="color: #22C55E; font-weight: 600; font-size: 0.9rem;">
                            {t('rubric.upload.success_short')}
                        </span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 1rem; color: #94A3B8; font-size: 0.85rem;">
                        {"".join(f'<span>{item}</span>' for item in info_items)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Data sampling selector - only show if data > 100
            if total_count > 100:
                # Initialize session state for sample count if needed
                sample_key = "rubric_sample_count_value"
                if sample_key not in st.session_state:
                    st.session_state[sample_key] = min(500, total_count)

                # Number input for custom value
                selected_count = st.number_input(
                    t("rubric.upload.sample_title"),
                    min_value=10,
                    max_value=total_count,
                    value=st.session_state[sample_key],
                    step=10,
                    key="rubric_sample_input",
                    help=f"{t('rubric.upload.sample_help')} (max: {total_count})",
                )
                st.session_state[sample_key] = selected_count

                # Quick select buttons (3 options max to avoid overflow)
                quick_options = [100, 500, total_count]
                quick_options = [opt for opt in quick_options if opt <= total_count]
                quick_options = sorted(set(quick_options))

                btn_cols = st.columns(len(quick_options))
                for i, opt in enumerate(quick_options):
                    with btn_cols[i]:
                        label = f"全部 ({opt})" if opt == total_count else str(opt)
                        is_selected = selected_count == opt
                        if st.button(
                            label,
                            key=f"quick_btn_{opt}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                        ):
                            st.session_state[sample_key] = opt
                            st.rerun()

            else:
                # For small datasets, use all data
                selected_count = total_count

            # Sample data if needed
            if selected_count < total_count:
                # Use random sampling with a fixed seed for reproducibility within session
                if "rubric_sample_seed" not in st.session_state:
                    st.session_state["rubric_sample_seed"] = random.randint(0, 10000)

                rng = random.Random(st.session_state["rubric_sample_seed"])
                sampled_data = rng.sample(all_data, selected_count)

                # Show sampling info
                percentage = round(selected_count / total_count * 100, 1)
                st.caption(
                    f"🎲 {t('rubric.upload.sampled_info', selected=selected_count, total=total_count)} ({percentage}%)"
                )
            else:
                sampled_data = all_data

            result["is_valid"] = True
            result["data"] = sampled_data
            result["count"] = len(sampled_data)
            result["total_count"] = total_count
            result["min_score"] = parse_result.min_score
            result["max_score"] = parse_result.max_score

            # Show warnings if any
            if parse_result.warnings:
                with st.expander(
                    f"⚠️ {t('rubric.upload.warnings')} ({len(parse_result.warnings)})",
                    expanded=False,
                ):
                    for warning in parse_result.warnings[:10]:
                        st.caption(f"• {_escape_html(warning)}")
                    if len(parse_result.warnings) > 10:
                        st.caption(f"... {len(parse_result.warnings) - 10} more")

            # Show data preview (collapsed by default)
            with st.expander(
                f"👁️ {t('rubric.upload.preview')}",
                expanded=False,
            ):
                preview = parser.get_preview(sampled_data, max_items=3)
                for i, item in enumerate(preview, 1):
                    st.markdown(
                        f"<div style='color: #A5B4FC; font-weight: 500; margin-top: 0.5rem;'>Record {i}</div>",
                        unsafe_allow_html=True,
                    )
                    for key, value in item.items():
                        display_value = _escape_html(str(value))
                        if len(display_value) > 150:
                            display_value = display_value[:150] + "..."
                        st.markdown(
                            f"<div style='font-size: 0.85rem;'>"
                            f"<span style='color: #94A3B8;'>{_escape_html(key)}:</span> "
                            f"<span style='color: #E2E8F0;'>{display_value}</span></div>",
                            unsafe_allow_html=True,
                        )
        else:
            # Show error
            error_msg = parse_result.error or "Unknown error"
            st.error(f"{t('rubric.upload.error')}: {_escape_html(error_msg)}")

            if parse_result.warnings:
                with st.expander(t("rubric.upload.details"), expanded=False):
                    for warning in parse_result.warnings[:10]:
                        st.caption(f"• {_escape_html(warning)}")

    return result
