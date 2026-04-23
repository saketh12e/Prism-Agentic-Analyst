"""Tests for tools/stat_tools.py — uses temporary CSV files, no LLM calls."""
from __future__ import annotations

import pandas as pd
import pytest
from tools.stat_tools import (
    run_anova,
    run_chi_square,
    run_correlation_matrix,
    run_pearson,
    run_segment_compare,
    run_ttest,
)


class TestRunPearson:
    def test_strong_positive_correlation(self, correlated_csv):
        result = run_pearson.invoke({"df_path": correlated_csv, "col_a": "x", "col_b": "y"})
        assert result["r"] > 0.9
        assert result["significant"] is True
        assert result["direction"] == "positive"

    def test_result_structure(self, correlated_csv):
        result = run_pearson.invoke({"df_path": correlated_csv, "col_a": "x", "col_b": "y"})
        for key in ("test", "col_a", "col_b", "r", "p_value", "significant", "direction"):
            assert key in result

    def test_r_is_bounded(self, correlated_csv):
        result = run_pearson.invoke({"df_path": correlated_csv, "col_a": "x", "col_b": "y"})
        assert -1.0 <= result["r"] <= 1.0

    def test_p_value_is_float(self, correlated_csv):
        result = run_pearson.invoke({"df_path": correlated_csv, "col_a": "x", "col_b": "y"})
        assert isinstance(result["p_value"], float)


class TestRunChiSquare:
    def test_independent_columns(self, tmp_path):
        import numpy as np
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "cat_a": rng.choice(["X", "Y"], 200),
            "cat_b": rng.choice(["P", "Q"], 200),
        })
        path = str(tmp_path / "cat.csv")
        df.to_csv(path, index=False)
        result = run_chi_square.invoke({"df_path": path, "col_a": "cat_a", "col_b": "cat_b"})
        assert "chi2" in result
        assert "p_value" in result
        assert "significant" in result

    def test_dependent_columns_are_significant(self, tmp_path):
        df = pd.DataFrame({
            "cat_a": ["A"] * 50 + ["B"] * 50,
            "cat_b": ["X"] * 50 + ["Y"] * 50,
        })
        path = str(tmp_path / "dep.csv")
        df.to_csv(path, index=False)
        result = run_chi_square.invoke({"df_path": path, "col_a": "cat_a", "col_b": "cat_b"})
        assert result["significant"] is True

    def test_dof_is_non_negative(self, tmp_path):
        df = pd.DataFrame({
            "a": ["X", "Y", "X", "Y"],
            "b": ["P", "P", "Q", "Q"],
        })
        path = str(tmp_path / "dof.csv")
        df.to_csv(path, index=False)
        result = run_chi_square.invoke({"df_path": path, "col_a": "a", "col_b": "b"})
        assert result["dof"] >= 0


class TestRunTtest:
    def test_significant_difference(self, two_group_csv):
        result = run_ttest.invoke({
            "df_path": two_group_csv, "num_col": "value", "group_col": "group"
        })
        assert result["significant"] is True
        assert abs(result["effect_size_cohens_d"]) > 0.5

    def test_result_structure(self, two_group_csv):
        result = run_ttest.invoke({
            "df_path": two_group_csv, "num_col": "value", "group_col": "group"
        })
        for key in ("test", "t", "p_value", "significant", "effect_size_cohens_d",
                    "groups", "group_means"):
            assert key in result

    def test_p_value_very_small_for_clear_separation(self, two_group_csv):
        result = run_ttest.invoke({
            "df_path": two_group_csv, "num_col": "value", "group_col": "group"
        })
        assert result["p_value"] < 0.001

    def test_not_enough_groups_returns_error(self, tmp_path):
        df = pd.DataFrame({"group": ["A"] * 10, "value": range(10)})
        path = str(tmp_path / "single_group.csv")
        df.to_csv(path, index=False)
        result = run_ttest.invoke({"df_path": path, "num_col": "value", "group_col": "group"})
        assert "error" in result


class TestRunAnova:
    def test_three_group_anova(self, tmp_path):
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "val":   [1.0] * 20 + [5.0] * 20 + [10.0] * 20,
        })
        path = str(tmp_path / "anova.csv")
        df.to_csv(path, index=False)
        result = run_anova.invoke({"df_path": path, "num_col": "val", "group_col": "group"})
        assert result["significant"] is True
        assert "f_stat" in result
        assert "group_means" in result

    def test_one_group_returns_error(self, tmp_path):
        df = pd.DataFrame({"group": ["A"] * 10, "val": range(10)})
        path = str(tmp_path / "one_group.csv")
        df.to_csv(path, index=False)
        result = run_anova.invoke({"df_path": path, "num_col": "val", "group_col": "group"})
        assert "error" in result


class TestRunCorrelationMatrix:
    def test_top_pairs_returned(self, correlated_csv):
        result = run_correlation_matrix.invoke({"df_path": correlated_csv})
        assert "top_pairs" in result
        assert len(result["top_pairs"]) >= 1

    def test_top_pair_is_x_y(self, correlated_csv):
        result = run_correlation_matrix.invoke({"df_path": correlated_csv})
        top = result["top_pairs"][0]
        cols = {top["col_a"], top["col_b"]}
        assert cols == {"x", "y"}

    def test_matrix_is_dict(self, correlated_csv):
        result = run_correlation_matrix.invoke({"df_path": correlated_csv})
        assert isinstance(result["matrix"], dict)


class TestRunSegmentCompare:
    def test_returns_segment_stats(self, csv_path):
        result = run_segment_compare.invoke({
            "df_path": csv_path, "segment_col": "city", "value_col": "salary"
        })
        assert "segment_stats" in result
        seg = result["segment_stats"]
        assert "mean" in seg
        assert "NYC" in seg["mean"]

    def test_all_aggregations_present(self, csv_path):
        result = run_segment_compare.invoke({
            "df_path": csv_path, "segment_col": "city", "value_col": "salary"
        })
        for agg in ("mean", "median", "count", "std"):
            assert agg in result["segment_stats"]
