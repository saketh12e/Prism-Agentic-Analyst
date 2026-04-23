"""Tests for graph/utils.py — no external dependencies needed."""
from __future__ import annotations

import pytest
from graph.utils import extract_text, last_ai_text, parse_json_from_text


class TestExtractText:
    def test_plain_string(self):
        assert extract_text("hello") == "hello"

    def test_list_of_strings(self):
        assert extract_text(["hello", "world"]) == "hello world"

    def test_list_of_dicts_with_text_key(self):
        result = extract_text([{"text": "foo"}, {"text": "bar"}])
        assert result == "foo bar"

    def test_list_of_dicts_missing_text_key(self):
        result = extract_text([{"other": "ignored"}])
        assert result == ""

    def test_empty_string(self):
        assert extract_text("") == ""

    def test_none_returns_empty(self):
        assert extract_text(None) == ""

    def test_integer_converted(self):
        assert extract_text(42) == "42"

    def test_mixed_list(self):
        result = extract_text(["a", {"text": "b"}, {"no_text": "c"}])
        assert "a" in result
        assert "b" in result


class TestParseJsonFromText:
    def test_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_from_text(text)
        assert result == {"key": "value"}

    def test_plain_code_block(self):
        text = '```\n{"key": 42}\n```'
        result = parse_json_from_text(text)
        assert result == {"key": 42}

    def test_raw_json(self):
        text = 'Here is the result: {"a": 1, "b": 2}'
        result = parse_json_from_text(text)
        assert result == {"a": 1, "b": 2}

    def test_no_json_returns_none(self):
        assert parse_json_from_text("no json here at all") is None

    def test_invalid_json_returns_none(self):
        assert parse_json_from_text("```json\n{bad json\n```") is None

    def test_nested_json(self):
        text = '```json\n{"outer": {"inner": [1, 2, 3]}}\n```'
        result = parse_json_from_text(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_prefers_json_block_over_raw(self):
        text = '{"raw": true}\n```json\n{"block": true}\n```'
        result = parse_json_from_text(text)
        assert result == {"block": True}

    def test_empty_string_returns_none(self):
        assert parse_json_from_text("") is None


class FakeMessage:
    """Minimal stand-in for a LangChain message with .content."""
    def __init__(self, content):
        self.content = content


class TestLastAiText:
    def test_returns_last_non_empty(self):
        msgs = [FakeMessage("first"), FakeMessage(""), FakeMessage("last")]
        assert last_ai_text(msgs) == "last"

    def test_reverses_order(self):
        msgs = [FakeMessage("early"), FakeMessage("recent")]
        assert last_ai_text(msgs) == "recent"

    def test_empty_list_returns_empty(self):
        assert last_ai_text([]) == ""

    def test_all_empty_returns_empty(self):
        assert last_ai_text([FakeMessage(""), FakeMessage("")]) == ""

    def test_list_content(self):
        msgs = [FakeMessage(["hello", "world"])]
        result = last_ai_text(msgs)
        assert "hello" in result

    def test_none_content_skipped(self):
        msgs = [FakeMessage("valid"), FakeMessage(None)]
        assert last_ai_text(msgs) == "valid"
