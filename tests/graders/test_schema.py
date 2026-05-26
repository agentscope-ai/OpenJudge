#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for schema types: EvalSuggestion, EvalFeedback,
and the eval_feedback field on GraderScore and GraderScoreCallback.
"""

import pytest

from openjudge.graders.schema import (
    EvalFeedback,
    EvalSuggestion,
    GraderScore,
    GraderScoreCallback,
)


@pytest.mark.unit
class TestEvalSuggestion:
    def test_create_with_reason_only(self):
        s = EvalSuggestion(reason="No assertion checks correctness")
        assert s.reason == "No assertion checks correctness"
        assert s.assertion is None

    def test_create_with_assertion(self):
        s = EvalSuggestion(
            assertion="The output includes 'John Smith'",
            reason="A hallucinated document would also pass",
        )
        assert s.assertion == "The output includes 'John Smith'"

    def test_serialization_roundtrip(self):
        s = EvalSuggestion(reason="Improve coverage")
        data = s.model_dump()
        s2 = EvalSuggestion(**data)
        assert s2 == s


@pytest.mark.unit
class TestEvalFeedback:
    def test_create_defaults(self):
        f = EvalFeedback()
        assert f.suggestions == []
        assert f.overall == "No suggestions, evals look solid"

    def test_create_with_suggestions(self):
        f = EvalFeedback(
            suggestions=[EvalSuggestion(reason="Add check")],
            overall="Assertions check presence but not correctness",
        )
        assert len(f.suggestions) == 1
        assert f.suggestions[0].reason == "Add check"
        assert f.overall == "Assertions check presence but not correctness"

    def test_from_dict(self):
        f = EvalFeedback(
            **{
                "suggestions": [{"reason": "Weak assertion"}],
                "overall": "Needs improvement",
            }
        )
        assert len(f.suggestions) == 1
        assert isinstance(f.suggestions[0], EvalSuggestion)

    def test_serialization_roundtrip(self):
        f = EvalFeedback(
            suggestions=[EvalSuggestion(assertion="A", reason="B")],
            overall="Fair",
        )
        data = f.model_dump()
        f2 = EvalFeedback(**data)
        assert f2.overall == f.overall
        assert len(f2.suggestions) == len(f.suggestions)


@pytest.mark.unit
class TestGraderScoreEvalFeedback:
    def test_score_defaults_none_feedback(self):
        s = GraderScore(name="test", score=3.0, reason="OK")
        assert s.eval_feedback is None

    def test_score_with_eval_feedback(self):
        fb = EvalFeedback(
            suggestions=[EvalSuggestion(reason="Add check")],
            overall="Needs work",
        )
        s = GraderScore(name="test", score=2.0, reason="Weak", eval_feedback=fb)
        assert s.eval_feedback is not None
        assert s.eval_feedback.overall == "Needs work"

    def test_score_serialization_includes_feedback(self):
        s = GraderScore(
            name="test",
            score=5.0,
            reason="Perfect",
            eval_feedback=EvalFeedback(overall="Solid"),
        )
        data = s.model_dump()
        assert "eval_feedback" in data
        assert data["eval_feedback"]["overall"] == "Solid"

    def test_score_deserialization_with_feedback(self):
        data = {
            "name": "test",
            "score": 3.0,
            "reason": "Mid",
            "eval_feedback": {
                "suggestions": [{"assertion": "A", "reason": "R"}],
                "overall": "OK",
            },
        }
        s = GraderScore(**data)
        assert s.eval_feedback is not None
        assert isinstance(s.eval_feedback, EvalFeedback)
        assert len(s.eval_feedback.suggestions) == 1
        assert isinstance(s.eval_feedback.suggestions[0], EvalSuggestion)


@pytest.mark.unit
class TestGraderScoreCallbackExcludesEvalFeedback:
    def test_callback_has_no_eval_feedback_field(self):
        schema = GraderScoreCallback.model_json_schema()
        assert "eval_feedback" not in schema.get("properties", {})

    def test_callback_only_has_reason_score_metadata(self):
        schema = GraderScoreCallback.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        assert props == {"reason", "score", "metadata"}
