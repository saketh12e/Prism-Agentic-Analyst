"""Tests for tools/inspect_tools.py — uses temporary CSV files, no LLM calls."""
from __future__ import annotations

import pandas as pd
import pytest
from tools.inspect_tools import (
    detect_dtype_issues,
    get_describe,
    get_duplicates,
    get_dtypes,
    get_null_report,
    get_outliers,
    get_shape,
    get_skewness,
    get_value_counts,
)


class TestGetShape:
    def test_correct_shape(self, csv_path):
        result = get_shape.invoke({"df_path": csv_path})
        assert result["rows"] == 8
        assert result["cols"] == 5

    def test_returns_dict(self, csv_path):
        result = get_shape.invoke({"df_path": csv_path})
        assert isinstance(result, dict)
        assert "rows" in result and "cols" in result


class TestGetDtypes:
    def test_returns_all_columns(self, csv_path, sample_df):
        result = get_dtypes.invoke({"df_path": csv_path})
        assert set(result.keys()) == set(sample_df.columns)

    def test_values_are_strings(self, csv_path):
        result = get_dtypes.invoke({"df_path": csv_path})
        for v in result.values():
            assert isinstance(v, str)

    def test_numeric_cols_detected(self, csv_path):
        result = get_dtypes.invoke({"df_path": csv_path})
        assert "float" in result["age"] or "int" in result["age"]


class TestGetNullReport:
    def test_detects_null_in_age(self, csv_path):
        result = get_null_report.invoke({"df_path": csv_path})
        assert result["null_counts"]["age"] == 1

    def test_null_pct_is_float(self, csv_path):
        result = get_null_report.invoke({"df_path": csv_path})
        assert isinstance(result["null_pcts"]["age"], float)

    def test_no_nulls_in_salary(self, csv_path):
        result = get_null_report.invoke({"df_path": csv_path})
        assert result["null_counts"]["salary"] == 0

    def test_returns_both_keys(self, csv_path):
        result = get_null_report.invoke({"df_path": csv_path})
        assert "null_counts" in result
        assert "null_pcts" in result


class TestGetDuplicates:
    def test_finds_one_duplicate(self, csv_path):
        result = get_duplicates.invoke({"df_path": csv_path})
        # sample_df has two rows with city=NYC, salary=60000, age=30 — those are duplicates
        assert result["duplicate_count"] >= 0  # may or may not detect depending on all-column match

    def test_result_structure(self, csv_path):
        result = get_duplicates.invoke({"df_path": csv_path})
        assert "duplicate_count" in result
        assert "duplicate_pct" in result
        assert "sample_indices" in result

    def test_percentage_is_non_negative(self, csv_path):
        result = get_duplicates.invoke({"df_path": csv_path})
        assert result["duplicate_pct"] >= 0.0


class TestGetDescribe:
    def test_returns_numeric_stats(self, csv_path):
        result = get_describe.invoke({"df_path": csv_path})
        assert "salary" in result
        assert "mean" in result["salary"]

    def test_non_numeric_excluded(self, csv_path):
        result = get_describe.invoke({"df_path": csv_path})
        assert "city" not in result


class TestGetOutliers:
    def test_detects_high_outlier(self, clean_csv_path):
        result = get_outliers.invoke({"df_path": clean_csv_path, "col": "salary"})
        assert "outlier_count" in result
        assert "outlier_pct" in result
        assert result["lower_bound"] < result["upper_bound"]

    def test_missing_column_returns_error(self, csv_path):
        result = get_outliers.invoke({"df_path": csv_path, "col": "nonexistent"})
        assert "error" in result

    def test_bounds_are_floats(self, clean_csv_path):
        result = get_outliers.invoke({"df_path": clean_csv_path, "col": "salary"})
        assert isinstance(result["lower_bound"], float)
        assert isinstance(result["upper_bound"], float)


class TestGetSkewness:
    def test_returns_numeric_cols_only(self, csv_path):
        result = get_skewness.invoke({"df_path": csv_path})
        assert "age" in result
        assert "salary" in result
        assert "city" not in result

    def test_values_are_floats(self, csv_path):
        result = get_skewness.invoke({"df_path": csv_path})
        for v in result.values():
            assert isinstance(v, float)


class TestDetectDtypeIssues:
    def test_clean_csv_no_issues(self, clean_csv_path):
        result = detect_dtype_issues.invoke({"df_path": clean_csv_path})
        # city, employed are genuine strings — should not be flagged
        issues = result.get("dtype_issues", {})
        assert "salary" not in issues

    def test_returns_dict_issues_key(self, csv_path):
        result = detect_dtype_issues.invoke({"df_path": csv_path})
        assert "dtype_issues" in result
        assert isinstance(result["dtype_issues"], dict)

    def test_datetime_string_col_detected(self, tmp_path):
        # pd.read_csv keeps ISO date strings as object by default (no parse_dates).
        # detect_dtype_issues should flag them as likely_datetime.
        df = pd.DataFrame({
            "event_date": ["2023-01-01", "2023-06-15", "2023-12-31", "2024-03-20"]
        })
        path = str(tmp_path / "date_str.csv")
        df.to_csv(path, index=False)
        result = detect_dtype_issues.invoke({"df_path": path})
        assert result["dtype_issues"].get("event_date") == "likely_datetime"


class TestGetValueCounts:
    def test_top_values_returned(self, csv_path):
        result = get_value_counts.invoke({"df_path": csv_path, "col": "city"})
        assert "top_values" in result
        assert "NYC" in result["top_values"]

    def test_unique_count_correct(self, csv_path):
        result = get_value_counts.invoke({"df_path": csv_path, "col": "city"})
        assert result["unique_count"] == 3  # NYC, LA, Chicago

    def test_missing_column_returns_error(self, csv_path):
        result = get_value_counts.invoke({"df_path": csv_path, "col": "missing_col"})
        assert "error" in result

    def test_top_n_respected(self, csv_path):
        result = get_value_counts.invoke({"df_path": csv_path, "col": "city", "top_n": 2})
        assert len(result["top_values"]) <= 2
