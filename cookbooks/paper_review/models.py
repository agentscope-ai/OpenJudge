# -*- coding: utf-8 -*-
"""LiteLLM-based model wrapper for PDF support."""

import base64
import hashlib
import os
import tempfile
import threading
from typing import Any, List, Optional

import litellm
from loguru import logger

litellm.drop_params = True
os.environ.setdefault("LITELLM_LOG", "ERROR")


def _pdf_base64_to_text(data_url: str) -> str:
    """Extract text from a base64-encoded PDF data URL using PyMuPDF."""
    try:
        import pymupdf  # PyMuPDF

        if data_url.startswith("data:application/pdf;base64,"):
            b64 = data_url[len("data:application/pdf;base64,") :]
        else:
            b64 = data_url
        pdf_bytes = base64.b64decode(b64)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(pages_text)
    except Exception as e:
        return f"[PDF text extraction failed: {e}]"


def _transform_messages_for_text_api(messages: List[dict]) -> List[dict]:
    """Convert any 'file' content blocks (PDF) to plain text blocks."""
    transformed = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "file":
                    file_data = block.get("file", {}).get("file_data", "")
                    text = _pdf_base64_to_text(file_data)
                    new_content.append({"type": "text", "text": text})
                else:
                    new_content.append(block)
            transformed.append({**msg, "content": new_content})
        else:
            transformed.append(msg)
    return transformed


class LiteLLMModel:
    """LiteLLM-based model with native PDF support.

    When using DashScope (dashscope.aliyuncs.com) with the qwen-long model,
    PDFs are uploaded via the Files API and referenced using fileid:// in the
    system message, as described in the DashScope documentation:
    https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

    For all other providers / models, the code first attempts to pass the PDF
    inline as a ``type: "file"`` content block (OpenAI native format).  If the
    API rejects that, it falls back to local text extraction via PyMuPDF.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 1500,
        use_vision_for_pdf: bool = False,
        vision_max_pages: Optional[int] = 30,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.use_vision_for_pdf = use_vision_for_pdf
        self.vision_max_pages = vision_max_pages
        # Cache pdf_data_hash -> DashScope file_id to avoid re-uploading the
        # same PDF across multiple grader calls within one pipeline run.
        self._file_id_cache: dict = {}
        # Cache pdf_data_hash -> list[base64_png] to avoid re-rendering pages.
        self._page_images_cache: dict = {}
        # Lock to prevent duplicate uploads when parallel graders race on the
        # same PDF hash (achat runs each call in a thread via asyncio.to_thread).
        self._upload_lock = threading.Lock()

    # ------------------------------------------------------------------
    # DashScope helpers
    # ------------------------------------------------------------------

    def _is_dashscope(self) -> bool:
        """Return True when base_url points to a DashScope endpoint, or the model
        uses the ``dashscope/`` provider prefix (e.g. ``dashscope/qwen3.5-plus``)."""
        if self.base_url and "dashscope" in self.base_url.lower():
            return True
        return self.model.startswith("dashscope/")

    def _supports_fileid(self) -> bool:
        """Return True for models that support DashScope fileid:// document analysis.

        Per the official docs, only qwen-long (and its snapshot variants)
        currently support document understanding via the Files API.
        """
        return "qwen-long" in self.model.lower()

    def _dashscope_rejects_file_block(self) -> bool:
        """Return True for DashScope models known not to accept type:'file' blocks.

        DashScope's OpenAI-compatible endpoint only accepts 'text', 'image_url',
        'video_url' and 'video' as content block types.  Sending a 'file' block
        causes an immediate 400 BadRequestError.  For models in this list we
        skip the inline attempt entirely and go straight to local text extraction,
        avoiding one unnecessary round-trip per grader call.
        """
        if not self._is_dashscope():
            return False
        # qwen-long uses fileid:// instead; all other Qwen models reject file blocks
        return not self._supports_fileid()

    def _pdf_to_page_images(self, pdf_data: str, max_pages: Optional[int] = None) -> list:
        """Render each PDF page to a base64-encoded JPEG using pypdfium2.

        Returns a list of ``data:image/jpeg;base64,...`` strings, one per page.
        ``max_pages`` overrides ``self.vision_max_pages`` for this call only.
        Cache key is ``(data_hash, effective_max_pages)`` so different page
        limits each get their own cached result.
        """
        import io

        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ImportError(
                "pypdfium2 is required for vision-based PDF processing. " "Install it with:  pip install pypdfium2"
            ) from exc

        effective_max = max_pages if max_pages is not None else self.vision_max_pages
        data_hash = hashlib.md5(pdf_data.encode()).hexdigest()
        cache_key = (data_hash, effective_max)
        if cache_key in self._page_images_cache:
            logger.debug(f"Reusing cached page images for hash {data_hash[:8]} " f"(max_pages={effective_max})")
            return self._page_images_cache[cache_key]

        b64 = pdf_data.removeprefix("data:application/pdf;base64,")
        pdf_bytes = base64.b64decode(b64)

        doc = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(doc)
        if effective_max is not None and total_pages > effective_max:
            logger.info(f"PDF has {total_pages} pages; rendering only the first " f"{effective_max} (max_pages limit)")
            page_count = effective_max
        else:
            page_count = total_pages

        images = []
        for i in range(page_count):
            page = doc[i]
            # Scale 1.5 → ~108 DPI; good balance between readability and token cost
            bitmap = page.render(scale=1.5)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            images.append(f"data:image/jpeg;base64,{img_b64}")
        doc.close()

        self._page_images_cache[cache_key] = images
        logger.info(f"Rendered {len(images)}/{total_pages} page(s) for vision model " f"(max_pages={effective_max})")
        return images

    def warmup_vision_cache(self, pdf_data: str, max_pages: Optional[int] = None) -> None:
        """Pre-render PDF pages into the cache (must be called from the main thread).

        All subsequent grader threads will find the cache populated and skip
        rendering, avoiding concurrent pypdfium2 access which is unsafe.
        ``max_pages`` overrides the instance default for this warm-up pass.
        """
        self._pdf_to_page_images(pdf_data, max_pages=max_pages)

    def _transform_messages_for_vision_pdf(
        self, messages: List[dict], pdf_data: str, max_pages: Optional[int] = None
    ) -> List[dict]:
        """Replace each ``type:'file'`` block with one ``image_url`` block per page."""
        page_images = self._pdf_to_page_images(pdf_data, max_pages=max_pages)
        image_blocks = [{"type": "image_url", "image_url": {"url": img}} for img in page_images]

        transformed: List[dict] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content: list = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "file":
                        new_content.extend(image_blocks)
                    else:
                        new_content.append(block)
                transformed.append({**msg, "content": new_content})
            else:
                transformed.append(msg)
        return transformed

    def _get_openai_client(self):
        """Return an openai.OpenAI client pointed at self.base_url (or the default
        DashScope endpoint when the model uses the ``dashscope/`` prefix)."""
        from openai import OpenAI

        base_url = self.base_url
        if base_url is None and self.model.startswith("dashscope/"):
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return OpenAI(api_key=self.api_key, base_url=base_url)

    def _extract_file_data_from_messages(self, messages: List[dict]) -> Optional[str]:
        """Return the first PDF base64 data URL found in a 'file' content block."""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "file":
                        return block.get("file", {}).get("file_data")
        return None

    def _upload_pdf_to_dashscope(self, pdf_data: str) -> Optional[str]:
        """Upload a base64-encoded PDF to the DashScope Files API.

        Returns the file_id string on success, or None if the upload fails
        (the caller should then fall back to local text extraction).

        Results are cached by a hash of the PDF content so the same paper is
        uploaded at most once per model instance lifetime.  A threading.Lock
        prevents parallel grader threads from racing and uploading the same
        PDF multiple times simultaneously.
        """
        data_hash = hashlib.md5(pdf_data.encode()).hexdigest()
        # Fast path: check cache without lock first
        if data_hash in self._file_id_cache:
            logger.debug(f"Reusing cached DashScope file_id for hash {data_hash[:8]}")
            return self._file_id_cache[data_hash]

        with self._upload_lock:
            # Re-check inside lock: another thread may have uploaded while we waited
            if data_hash in self._file_id_cache:
                logger.debug(f"Reusing cached DashScope file_id for hash {data_hash[:8]} (post-lock)")
                return self._file_id_cache[data_hash]

            try:
                b64 = pdf_data.removeprefix("data:application/pdf;base64,")
                pdf_bytes = base64.b64decode(b64)

                client = self._get_openai_client()

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                try:
                    with open(tmp_path, "rb") as f:
                        file_object = client.files.create(file=f, purpose="file-extract")
                    logger.info(f"Uploaded PDF to DashScope, file_id={file_object.id}")
                    self._file_id_cache[data_hash] = file_object.id
                    return file_object.id
                finally:
                    os.unlink(tmp_path)

            except Exception as e:
                logger.warning(f"DashScope PDF upload failed ({e}), falling back to text extraction")
                return None

    def _transform_messages_for_fileid(self, messages: List[dict], file_id: str) -> List[dict]:
        """Rewrite messages to use DashScope fileid:// instead of inline file blocks.

        The ``type: "file"`` content block is removed from the user message.
        A new system message ``fileid://<file_id>`` is inserted immediately
        before the first user message that contained a file block, matching
        the format shown in the DashScope documentation.
        """
        transformed: List[dict] = []
        fileid_inserted = False

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = [
                    block for block in content if not (isinstance(block, dict) and block.get("type") == "file")
                ]
                had_file_block = len(new_content) < len(content)

                if had_file_block and not fileid_inserted:
                    transformed.append({"role": "system", "content": f"fileid://{file_id}"})
                    fileid_inserted = True

                # Simplify single-text-block array to a plain string so that
                # qwen-long (which expects string content) handles it correctly.
                if len(new_content) == 1 and new_content[0].get("type") == "text":
                    transformed.append({**msg, "content": new_content[0]["text"]})
                elif new_content:
                    transformed.append({**msg, "content": new_content})
                # Drop the message entirely if it contained only a file block.
            else:
                transformed.append(msg)

        return transformed

    def cleanup_files(self) -> None:
        """Delete all files previously uploaded to DashScope and clear the cache.

        Call this after a pipeline run to avoid leaving orphaned files on the
        DashScope Files service.
        """
        if not self._file_id_cache:
            return
        try:
            client = self._get_openai_client()
            for file_id in list(self._file_id_cache.values()):
                try:
                    client.files.delete(file_id)
                    logger.debug(f"Deleted DashScope file {file_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete DashScope file {file_id}: {e}")
            self._file_id_cache.clear()
        except Exception as e:
            logger.warning(f"DashScope file cleanup failed: {e}")

    # ------------------------------------------------------------------
    # Core chat methods
    # ------------------------------------------------------------------

    async def achat(self, messages: List[dict], **kwargs) -> Any:
        """Async chat completion with PDF support."""
        import asyncio

        return await asyncio.to_thread(self._chat_sync, messages, **kwargs)

    def _chat_sync(self, messages: List[dict], **kwargs) -> Any:
        """Sync chat completion.

        Accepts a private ``_vision_max_pages`` kwarg (int | None) that
        overrides ``self.vision_max_pages`` for this call only.  The kwarg is
        stripped before forwarding to litellm.
        """
        vision_max_pages_override: Optional[int] = kwargs.pop("_vision_max_pages", None)
        completion_kwargs = {
            "model": self._get_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            **kwargs,
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        # When the user specifies a dashscope/ model prefix but no explicit
        # base_url, automatically wire up the DashScope OpenAI-compatible endpoint
        # so litellm (routed as openai/) can reach the correct API.
        effective_base_url = self.base_url
        if effective_base_url is None and self.model.startswith("dashscope/"):
            effective_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if effective_base_url:
            completion_kwargs["api_base"] = effective_base_url

        # ------------------------------------------------------------------
        # DashScope + qwen-long: upload PDF and use fileid:// in system msg
        # ------------------------------------------------------------------
        if self._is_dashscope() and self._supports_fileid():
            pdf_data = self._extract_file_data_from_messages(messages)
            if pdf_data:
                file_id = self._upload_pdf_to_dashscope(pdf_data)
                if file_id:
                    completion_kwargs["messages"] = self._transform_messages_for_fileid(messages, file_id)
                else:
                    # Upload failed — skip the inline attempt and go straight
                    # to local text extraction (qwen-long doesn't accept
                    # type:"file" blocks either).
                    completion_kwargs["messages"] = _transform_messages_for_text_api(messages)

        # ------------------------------------------------------------------
        # General path: try inline file block; fall back to vision or text
        # ------------------------------------------------------------------
        # For DashScope models other than qwen-long we know upfront that
        # type:'file' blocks are rejected.  Skip the doomed first attempt.
        if self._dashscope_rejects_file_block():
            pdf_data = self._extract_file_data_from_messages(completion_kwargs["messages"])
            if pdf_data is not None:
                if self.use_vision_for_pdf:
                    logger.debug("DashScope vision model: rendering PDF pages as images")
                    completion_kwargs["messages"] = self._transform_messages_for_vision_pdf(
                        completion_kwargs["messages"],
                        pdf_data,
                        max_pages=vision_max_pages_override,
                    )
                else:
                    logger.debug(
                        "DashScope model does not support type:'file' blocks — " "using local text extraction directly"
                    )
                    completion_kwargs["messages"] = _transform_messages_for_text_api(completion_kwargs["messages"])

        try:
            response = litellm.completion(**completion_kwargs)
        except litellm.BadRequestError as e:
            if "file" in str(e).lower() or "invalid value" in str(e).lower():
                # API does not support 'file' type — convert PDF to text and retry
                completion_kwargs["messages"] = _transform_messages_for_text_api(messages)
                response = litellm.completion(**completion_kwargs)
            else:
                raise
        return _LiteLLMResponse(response.choices[0].message.content)

    def _get_model_name(self) -> str:
        """Get model name with provider prefix if needed.

        litellm does not have a built-in ``dashscope/`` provider prefix.
        DashScope is accessed via an OpenAI-compatible endpoint, so the model
        name must be routed with the ``openai/`` prefix when going through
        litellm.  We strip the ``dashscope/`` prefix here and let the
        ``api_base`` / ``api_key`` kwargs direct the request to DashScope.
        """
        model = self.model

        # Strip the convenience ``dashscope/`` prefix that users (and SKILL.md)
        # use to express intent — litellm has no such provider and it must be
        # replaced with ``openai/`` to use the OpenAI-compatible route.
        if model.startswith("dashscope/"):
            model = "openai/" + model[len("dashscope/") :]
            return model

        # Add provider prefix for litellm routing
        if "gemini" in model.lower() and not model.startswith("gemini/"):
            model = f"gemini/{model}"
        elif "claude" in model.lower() and not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        elif self.base_url and not any(
            model.startswith(p) for p in ("openai/", "anthropic/", "gemini/", "azure/", "bedrock/")
        ):
            # When using a custom base_url with an OpenAI-compatible API (e.g. DashScope, vLLM),
            # LiteLLM requires the "openai/" prefix to route correctly.
            model = f"openai/{model}"
        return model


class _LiteLLMResponse:
    """Simple response wrapper."""

    def __init__(self, content: str):
        self.content = content
