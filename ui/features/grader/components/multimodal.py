# -*- coding: utf-8 -*-
"""Multimodal input components for Grader feature."""

from typing import Any, Optional

import streamlit as st
from shared.utils.helpers import encode_image_to_base64, get_image_format


def render_image_uploader(
    key: str = "image_upload",
    label: str = "Upload Image",
    help_text: str = "Supported formats: JPG, PNG, GIF, WEBP",
) -> Optional[dict[str, Any]]:
    """Render an image uploader component.

    Args:
        key: Unique key for the uploader
        label: Label text
        help_text: Help text to display

    Returns:
        Dictionary with image data or None if no image uploaded
    """
    uploaded_file = st.file_uploader(
        label,
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key=key,
        help=help_text,
    )

    if uploaded_file is not None:
        # Read image bytes
        image_bytes = uploaded_file.read()

        # Show preview
        st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)

        # Get format and encode
        image_format = get_image_format(uploaded_file.name)
        base64_data = encode_image_to_base64(image_bytes)

        return {
            "filename": uploaded_file.name,
            "format": image_format,
            "base64": base64_data,
            "size": len(image_bytes),
        }

    return None


def render_image_url_input(
    key: str = "image_url",
    label: str = "Image URL",
    placeholder: str = "https://example.com/image.jpg",
) -> Optional[str]:
    """Render an image URL input component.

    Args:
        key: Unique key for the input
        label: Label text
        placeholder: Placeholder text

    Returns:
        Image URL or None if empty
    """
    url = st.text_input(
        label,
        key=key,
        placeholder=placeholder,
        help="Enter a publicly accessible image URL",
    )

    if url:
        try:
            st.image(url, caption="Image Preview", use_container_width=True)
            return url
        except Exception as e:
            st.warning(f"Could not load image preview. Error: {e}")
            return url

    return None


def render_multimodal_input(
    key_prefix: str = "multimodal",
) -> tuple[list[Any], Optional[str]]:
    """Render a multimodal input section for text + images.

    Args:
        key_prefix: Prefix for component keys

    Returns:
        Tuple of (content_list for evaluation, context_above text)
    """
    from openjudge.graders.multimodal._internal import MLLMImage

    st.markdown(
        """
        <div class="info-card">
            <div style="font-size: 0.85rem; color: #94A3B8;">
                For multimodal evaluation, provide text context and images.
                Images will be evaluated for coherence with surrounding text.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Text context above image
    context_above = st.text_area(
        "Context Above Image",
        key=f"{key_prefix}_context_above",
        height=100,
        placeholder="Enter text that appears before the image...",
        help="Text content that comes before the image",
    )

    # Image input method selector
    input_method = st.radio(
        "Image Input Method",
        options=["Upload", "URL"],
        horizontal=True,
        key=f"{key_prefix}_method",
    )

    image_data = None
    if input_method == "Upload":
        image_data = render_image_uploader(key=f"{key_prefix}_upload")
    else:
        image_url = render_image_url_input(key=f"{key_prefix}_url")
        if image_url:
            image_data = {"url": image_url}

    # Text context below image
    context_below = st.text_area(
        "Context Below Image",
        key=f"{key_prefix}_context_below",
        height=100,
        placeholder="Enter text that appears after the image...",
        help="Text content that comes after the image",
    )

    # Build response content list
    content_list: list[Any] = []

    if context_above:
        content_list.append(context_above)

    if image_data:
        if "url" in image_data:
            content_list.append(MLLMImage(url=image_data["url"]))
        elif "base64" in image_data:
            content_list.append(
                MLLMImage(
                    base64=image_data["base64"],
                    format=image_data.get("format", "jpeg"),
                )
            )

    if context_below:
        content_list.append(context_below)

    return content_list, context_above


def render_text_to_image_input(
    key_prefix: str = "t2i",
) -> tuple[str, Optional[Any]]:
    """Render input for text-to-image evaluation.

    Args:
        key_prefix: Prefix for component keys

    Returns:
        Tuple of (text_prompt, MLLMImage or None)
    """
    from openjudge.graders.multimodal._internal import MLLMImage

    st.markdown(
        """
        <div class="info-card">
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Evaluate AI-generated images based on text prompts.
                Assesses both semantic consistency and visual quality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Text prompt
    text_prompt = st.text_area(
        "Text Prompt",
        key=f"{key_prefix}_prompt",
        height=100,
        placeholder="A fluffy orange cat sitting on a blue velvet sofa...",
        help="The text prompt used to generate the image",
    )

    # Generated image input
    st.markdown("**Generated Image**")

    input_method = st.radio(
        "Input Method",
        options=["Upload", "URL"],
        horizontal=True,
        key=f"{key_prefix}_method",
    )

    image = None
    if input_method == "Upload":
        image_data = render_image_uploader(key=f"{key_prefix}_upload")
        if image_data and "base64" in image_data:
            image = MLLMImage(
                base64=image_data["base64"],
                format=image_data.get("format", "jpeg"),
            )
    else:
        image_url = render_image_url_input(key=f"{key_prefix}_url")
        if image_url:
            image = MLLMImage(url=image_url)

    return text_prompt, image
