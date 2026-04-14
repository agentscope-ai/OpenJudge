#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CodeSecurityGrader.

Tests code security evaluation with mocked LLM responses.

Example:
    Run all tests:
    ```bash
    pytest tests/graders/code/test_code_security.py -v
    ```
"""

from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.code.code_security import (
    DEFAULT_CODE_SECURITY_TEMPLATE,
    CodeSecurityGrader,
)
from openjudge.models.schema.prompt_template import LanguageEnum


@pytest.mark.unit
class TestCodeSecurityGraderUnit:
    """Unit tests for CodeSecurityGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = CodeSecurityGrader(model=mock_model)
        assert grader.name == "code_security"
        assert grader.threshold == 4
        assert grader.model == mock_model

    def test_initialization_with_custom_threshold(self):
        """Test initialization with custom threshold"""
        mock_model = AsyncMock()
        grader = CodeSecurityGrader(model=mock_model, threshold=3)
        assert grader.threshold == 3

    def test_initialization_invalid_threshold(self):
        """Test initialization with invalid threshold raises ValueError"""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            CodeSecurityGrader(model=mock_model, threshold=0)

        with pytest.raises(ValueError, match="threshold must be in range"):
            CodeSecurityGrader(model=mock_model, threshold=6)

    def test_initialization_with_language(self):
        """Test initialization with different languages"""
        mock_model = AsyncMock()
        grader_zh = CodeSecurityGrader(model=mock_model, language=LanguageEnum.ZH)
        assert grader_zh.language == LanguageEnum.ZH

    def test_default_template_exists(self):
        """Test that default template is properly defined"""
        assert DEFAULT_CODE_SECURITY_TEMPLATE is not None
        assert LanguageEnum.EN in DEFAULT_CODE_SECURITY_TEMPLATE.messages
        assert LanguageEnum.ZH in DEFAULT_CODE_SECURITY_TEMPLATE.messages

    @pytest.mark.asyncio
    async def test_secure_code_high_score(self):
        """Test evaluation of secure code"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "No security issues detected. The code follows secure coding practices throughout.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a login function that checks user credentials in a database.",
                response="""import sqlite3, hashlib
def login(username, password, conn):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cursor = conn.execute(
        "SELECT id FROM users WHERE username = ? AND password_hash = ?",
        (username, hashed),
    )
    return cursor.fetchone()""",
            )

            assert result.score == 5
            assert "No security issues" in result.reason
            assert result.metadata["threshold"] == 4

    @pytest.mark.asyncio
    async def test_insecure_code_low_score(self):
        """Test evaluation of insecure code with SQL injection and hardcoded credentials"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "SQL injection: string concatenation in query. Hardcoded credential: DB_PASSWORD.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a login function that checks user credentials in a database.",
                response="""import sqlite3
DB_PASSWORD = "admin123"
def login(username, password):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return conn.execute(query).fetchone()""",
            )

            assert result.score == 1
            assert "SQL injection" in result.reason

    @pytest.mark.asyncio
    async def test_moderate_security_issues(self):
        """Test evaluation of code with moderate security issues"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": "Missing input validation on an internal-only endpoint. Moderate security issue.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Build a file download endpoint.",
                response="def download(path): return open(path).read()",
            )

            assert result.score == 3
            assert "Missing input validation" in result.reason

    @pytest.mark.asyncio
    async def test_path_traversal_vulnerability(self):
        """Test evaluation of code with path traversal vulnerability"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Path traversal vulnerability: user controls `path` without sanitization.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Build a file download endpoint.",
                response="def download(path): return open(path).read()",
            )

            assert result.score == 2
            assert "Path traversal" in result.reason

    @pytest.mark.asyncio
    async def test_metadata_contains_threshold(self):
        """Test that metadata contains the threshold value"""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 4, "reason": "Minor concern."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function.",
                response="def func(): pass",
            )

            assert result.metadata["threshold"] == 3

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM fails"""
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = Exception("API Error")

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function.",
                response="def func(): pass",
            )

            assert "Evaluation error: API Error" in result.error

    @pytest.mark.asyncio
    async def test_pure_math_function_high_score(self):
        """Test that a pure math function with no security surface gets high score"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "No security issues. Pure math function with no security surface area.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response

            mock_model = AsyncMock()
            grader = CodeSecurityGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                query="Write a function to compute factorial.",
                response="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            )

            assert result.score == 5
