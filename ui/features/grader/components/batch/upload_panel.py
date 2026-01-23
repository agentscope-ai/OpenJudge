# -*- coding: utf-8 -*-
"""Upload panel component for batch evaluation.

Provides:
- File upload (JSON/CSV)
- Data preview
- Format instructions and sample download
- Validation feedback
"""

import hashlib
from typing import Any

import streamlit as st
from features.grader.services.file_parser import (
    MAX_BATCH_SIZE,
    generate_sample_data,
    get_optional_fields_for_grader,
    get_required_fields_for_grader,
    is_grader_batch_supported,
    parse_file,
    validate_data_for_grader,
)
from shared.components.common import render_section_header


def _render_format_instructions(grader_config: dict[str, Any]) -> None:
    """Render format instructions based on selected grader."""
    required_fields = get_required_fields_for_grader(grader_config)
    optional_fields = get_optional_fields_for_grader(grader_config)
    category = grader_config.get("category", "common")

    st.markdown(
        """<div style="
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="font-weight: 600; color: #A5B4FC; margin-bottom: 0.5rem;">
                📋 Data Format Requirements / 数据格式要求
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Required fields
    required_display = ", ".join(required_fields) if required_fields else "response"
    st.markdown(
        f"""<div style="color: #E2E8F0; font-size: 0.85rem; margin-bottom: 0.5rem;">
            <strong>Required fields / 必需字段:</strong>
            <code style="background: rgba(30, 41, 59, 0.8); padding: 0.15rem 0.4rem;
                border-radius: 4px; margin-left: 0.5rem;">{required_display}</code>
        </div>""",
        unsafe_allow_html=True,
    )

    # Optional fields
    if optional_fields:
        optional_display = ", ".join(optional_fields)
        st.markdown(
            f"""<div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                <strong>Optional fields / 可选字段:</strong>
                <code style="background: rgba(30, 41, 59, 0.8); padding: 0.15rem 0.4rem;
                    border-radius: 4px; margin-left: 0.5rem;">{optional_display}</code>
            </div>""",
            unsafe_allow_html=True,
        )

    # Format note for agent graders
    if category == "agent":
        st.markdown(
            """<div style="color: #FCD34D; font-size: 0.8rem; margin-top: 0.5rem;">
                ⚠️ Agent graders require JSON format. CSV is not supported for complex fields.
                <br/>Agent 评估器需要 JSON 格式，CSV 不支持复杂字段。
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_sample_download(grader_config: dict[str, Any]) -> None:
    """Render sample file download buttons."""
    col1, col2 = st.columns(2)

    with col1:
        json_sample = generate_sample_data(grader_config, "json")
        st.download_button(
            label="📥 Download JSON Sample",
            data=json_sample,
            file_name="sample_data.json",
            mime="application/json",
            use_container_width=True,
        )

    category = grader_config.get("category", "common")
    with col2:
        if category != "agent":
            csv_sample = generate_sample_data(grader_config, "csv")
            st.download_button(
                label="📥 Download CSV Sample",
                data=csv_sample,
                file_name="sample_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button(
                "CSV not supported",
                disabled=True,
                use_container_width=True,
                help="Agent graders require JSON format",
            )


def _render_data_preview(data: list[dict[str, Any]], max_rows: int = 5) -> None:
    """Render data preview table."""
    if not data:
        return

    st.markdown(
        f"""<div style="
            font-weight: 500;
            color: #94A3B8;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        ">
            Data Preview / 数据预览 (showing {min(len(data), max_rows)} of {len(data)} records)
        </div>""",
        unsafe_allow_html=True,
    )

    # Prepare preview data
    preview_data = data[:max_rows]

    # Truncate long values for display
    display_data = []
    for record in preview_data:
        display_record = {}
        for key, value in record.items():
            if value is None:
                display_record[key] = ""
            elif isinstance(value, str):
                display_record[key] = value[:100] + "..." if len(value) > 100 else value
            elif isinstance(value, (dict, list)):
                str_value = str(value)
                display_record[key] = str_value[:100] + "..." if len(str_value) > 100 else str_value
            else:
                display_record[key] = str(value)
        display_data.append(display_record)

    # Use dataframe for display
    st.dataframe(
        display_data,
        use_container_width=True,
        height=min(200, 50 + len(display_data) * 35),
    )


def _render_validation_results(
    validation_result: Any,
    record_count: int,
) -> None:
    """Render validation results."""
    if validation_result.valid:
        st.success(
            f"✅ Data validated successfully! / 数据验证通过！\n\n"
            f"**{record_count}** records ready for evaluation / **{record_count}** 条数据准备就绪"
        )
    else:
        for error in validation_result.errors:
            st.error(f"❌ {error}")

    for warning in validation_result.warnings:
        st.warning(f"⚠️ {warning}")


def render_upload_panel(sidebar_config: dict[str, Any]) -> dict[str, Any]:
    """Render the file upload panel.

    Args:
        sidebar_config: Configuration from sidebar

    Returns:
        Dictionary containing:
        - parsed_data: List of parsed records (or empty list)
        - is_valid: Whether data is valid for evaluation
        - record_count: Number of records
        - validation_errors: List of validation errors
    """
    result = {
        "parsed_data": [],
        "is_valid": False,
        "record_count": 0,
        "validation_errors": [],
    }

    grader_config = sidebar_config.get("grader_config")
    grader_name = sidebar_config.get("grader_name")

    render_section_header("Upload Data / 上传数据")

    # Check if grader is selected
    if not grader_config or not grader_name:
        st.warning("Please select a grader from the sidebar / 请先在侧边栏选择评估器")
        return result

    # Check if grader supports batch evaluation
    is_supported, reason = is_grader_batch_supported(grader_config)
    if not is_supported:
        st.error(f"❌ {reason}")
        return result

    # Format instructions
    _render_format_instructions(grader_config)

    # Sample download
    with st.expander("📥 Download Sample Files / 下载示例文件", expanded=False):
        _render_sample_download(grader_config)

    # File uploader
    st.markdown(
        f"""<div style="
            color: #94A3B8;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        ">
            Supported formats: JSON, CSV | Max records: {MAX_BATCH_SIZE:,}
        </div>""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload evaluation data",
        type=["json", "csv"],
        key="batch_file_uploader",
        label_visibility="collapsed",
        help=f"Upload JSON or CSV file with up to {MAX_BATCH_SIZE:,} records",
    )

    # Process uploaded file
    if uploaded_file is not None:
        # Read file content for hash calculation
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer for later use

        # Use content hash to detect file changes (more reliable than name+size)
        content_hash = hashlib.md5(file_content).hexdigest()
        file_key = f"{uploaded_file.name}_{content_hash}"
        prev_file_key = st.session_state.get("batch_prev_file_key")

        if file_key != prev_file_key:
            # New file or content changed - parse it
            st.session_state.batch_prev_file_key = file_key

            with st.spinner("Parsing file... / 正在解析文件..."):
                # Create a BytesIO object from content since we already read the file
                import io

                file_obj = io.BytesIO(file_content)
                parse_result = parse_file(file_obj, uploaded_file.name)

            if not parse_result.success:
                for error in parse_result.errors:
                    st.error(f"❌ {error}")
                st.session_state.batch_parsed_data = None
                st.session_state.batch_parse_result = None
                return result

            # Store parsed data in session state
            st.session_state.batch_parsed_data = parse_result.data
            st.session_state.batch_parse_result = parse_result

            # Show parse warnings
            for warning in parse_result.warnings:
                st.warning(f"⚠️ {warning}")

        # Retrieve from session state
        parsed_data = st.session_state.get("batch_parsed_data")
        parse_result = st.session_state.get("batch_parse_result")

        if parsed_data:
            # Show data preview
            _render_data_preview(parsed_data)

            # Validate data
            validation_result = validate_data_for_grader(parsed_data, grader_config)
            _render_validation_results(validation_result, len(parsed_data))

            # Update result
            result["parsed_data"] = parsed_data
            result["record_count"] = len(parsed_data)
            result["is_valid"] = validation_result.valid
            result["validation_errors"] = validation_result.errors

    else:
        # Clear session state when no file
        if "batch_parsed_data" in st.session_state:
            del st.session_state.batch_parsed_data
        if "batch_parse_result" in st.session_state:
            del st.session_state.batch_parse_result
        if "batch_prev_file_key" in st.session_state:
            del st.session_state.batch_prev_file_key

        # Empty state
        st.markdown(
            """<div style="
                text-align: center;
                padding: 2rem;
                color: #64748B;
                background: rgba(30, 41, 59, 0.3);
                border: 2px dashed #334155;
                border-radius: 8px;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-size: 0.9rem;">
                    Drag and drop a file here or click to browse<br/>
                    拖放文件到此处或点击选择文件
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    return result
