# -*- coding: utf-8 -*-
"""LiteLLM-based model wrapper for PDF support."""

import base64
import hashlib
import os
import tempfile
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
            b64 = data_url[len("data:application/pdf;base64,"):]
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
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        # Cache pdf_data_hash -> DashScope file_id to avoid re-uploading the
        # same PDF across multiple grader calls within one pipeline run.
        self._file_id_cache: dict = {}

    # ------------------------------------------------------------------
    # DashScope helpers
    # ------------------------------------------------------------------

    def _is_dashscope(self) -> bool:
        """Return True when base_url points to a DashScope endpoint."""
        return bool(self.base_url and "dashscope" in self.base_url.lower())

    def _supports_fileid(self) -> bool:
        """Return True for models that support DashScope fileid:// document analysis.

        Per the official docs, only qwen-long (and its snapshot variants)
        currently support document understanding via the Files API.
        """
        return "qwen-long" in self.model.lower()

    def _get_openai_client(self):
        """Return an openai.OpenAI client pointed at self.base_url."""
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

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
        uploaded at most once per model instance lifetime.
        """
        data_hash = hashlib.md5(pdf_data.encode()).hexdigest()
        if data_hash in self._file_id_cache:
            logger.debug(f"Reusing cached DashScope file_id for hash {data_hash[:8]}")
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
                    block for block in content
                    if not (isinstance(block, dict) and block.get("type") == "file")
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
        """Sync chat completion."""
        completion_kwargs = {
            "model": self._get_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            **kwargs,
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.base_url:
            completion_kwargs["api_base"] = self.base_url

        # ------------------------------------------------------------------
        # DashScope + qwen-long: upload PDF and use fileid:// in system msg
        # ------------------------------------------------------------------
        if self._is_dashscope() and self._supports_fileid():
            pdf_data = self._extract_file_data_from_messages(messages)
            if pdf_data:
                file_id = self._upload_pdf_to_dashscope(pdf_data)
                if file_id:
                    completion_kwargs["messages"] = self._transform_messages_for_fileid(
                        messages, file_id
                    )
                else:
                    # Upload failed — skip the inline attempt and go straight
                    # to local text extraction (qwen-long doesn't accept
                    # type:"file" blocks either).
                    completion_kwargs["messages"] = _transform_messages_for_text_api(messages)

        # ------------------------------------------------------------------
        # General path: try inline file block; fall back to text extraction
        # ------------------------------------------------------------------
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
        """Get model name with provider prefix if needed."""
        model = self.model
        # Add provider prefix for litellm routing
        if "gemini" in model.lower() and not model.startswith("gemini/"):
            model = f"gemini/{model}"
        elif "claude" in model.lower() and not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        elif self.base_url and not any(model.startswith(p) for p in ("openai/", "anthropic/", "gemini/", "azure/", "bedrock/")):
            # When using a custom base_url with an OpenAI-compatible API (e.g. DashScope, vLLM),
            # LiteLLM requires the "openai/" prefix to route correctly.
            model = f"openai/{model}"
        return model


class _LiteLLMResponse:
    """Simple response wrapper."""

    def __init__(self, content: str):
        self.content = content
