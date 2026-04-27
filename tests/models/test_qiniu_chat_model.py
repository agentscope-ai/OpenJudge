#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for QiniuChatModel."""

import pytest

from openjudge.models import QiniuChatModel
from openjudge.models.qiniu_chat_model import QINIU_MODELS


@pytest.mark.unit
class TestQiniuChatModelInit:
    """Verify constructor defaults and argument handling."""

    def test_default_base_url(self):
        model = QiniuChatModel(api_key="test-key")
        assert model.client.base_url is not None
        assert "qnaigc.com" in str(model.client.base_url)

    def test_default_model(self):
        model = QiniuChatModel(api_key="test-key")
        assert model.model == "deepseek-v3"

    def test_qiniu_models_contains_deepseek_v3(self):
        assert "deepseek-v3" in QINIU_MODELS

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("QINIU_API_KEY", "env-key-123")
        model = QiniuChatModel()
        assert model.client.api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("QINIU_API_KEY", "env-key")
        model = QiniuChatModel(api_key="explicit-key")
        assert model.client.api_key == "explicit-key"

    def test_custom_base_url(self):
        model = QiniuChatModel(api_key="k", base_url="https://custom.qnaigc.com/v1")
        assert "custom.qnaigc.com" in str(model.client.base_url)


@pytest.mark.unit
class TestQiniuChatModelInheritance:
    """Verify class hierarchy and exported symbols."""

    def test_inherits_from_openai_chat_model(self):
        from openjudge.models.openai_chat_model import OpenAIChatModel

        assert issubclass(QiniuChatModel, OpenAIChatModel)

    def test_exported_from_models_package(self):
        from openjudge.models import QiniuChatModel as Imported

        assert Imported is QiniuChatModel
