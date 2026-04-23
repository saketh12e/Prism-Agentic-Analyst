"""Tests for tools/insight_tools.py — subprocess execution, no LLM calls."""
from __future__ import annotations

import pytest
from tools.insight_tools import test_hypothesis


class TestTestHypothesis:
    def test_successful_execution(self, clean_csv_path):
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": "Salary varies by city",
            "code": "output = df.groupby('city')['salary'].mean()",
        })
        assert result["success"] is True
        assert len(result["result"]) > 0
        assert result["error"] == ""

    def test_hypothesis_echoed_back(self, clean_csv_path):
        hyp = "Test hypothesis text"
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": hyp,
            "code": "output = df.shape[0]",
        })
        assert result["hypothesis"] == hyp

    def test_missing_output_variable(self, clean_csv_path):
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": "Bad code test",
            "code": "x = 42",  # never assigns to `output`
        })
        assert result["success"] is False

    def test_syntax_error_returns_failure(self, clean_csv_path):
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": "Syntax error test",
            "code": "output = df[",
        })
        assert result["success"] is False
        assert result["error"] != ""

    def test_scalar_output(self, clean_csv_path):
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": "Mean salary is above 50k",
            "code": "output = df['salary'].mean()",
        })
        assert result["success"] is True
        # Result should be numeric string
        assert float(result["result"]) > 0

    def test_result_capped_at_2000_chars(self, clean_csv_path):
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": "Large output test",
            "code": "output = '\\n'.join([str(i) for i in range(10000)])",
        })
        assert result["success"] is True
        assert len(result["result"]) <= 2000

    def test_runtime_exception_returns_failure(self, clean_csv_path):
        result = test_hypothesis.invoke({
            "df_path": clean_csv_path,
            "hypothesis": "Division by zero",
            "code": "output = 1 / 0",
        })
        assert result["success"] is False
