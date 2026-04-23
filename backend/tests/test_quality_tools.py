"""Tests for tools/quality_tools.py — no external API calls needed."""
from __future__ import annotations

import pytest
from graph.state import ProfileReport
from tools.quality_tools import compute_quality_score


def _make_profile(**overrides) -> ProfileReport:
    """Return a ProfileReport with sensible defaults, applying any overrides."""
    defaults = dict(
        shape=(100, 5),
        dtypes={"a": "float64", "b": "float64", "c": "object"},
        null_counts={"a": 0, "b": 0, "c": 0},
        null_pcts={"a": 0.0, "b": 0.0, "c": 0.0},
        duplicate_count=0,
        duplicate_pct=0.0,
        numeric_cols=["a", "b"],
        categorical_cols=["c"],
        date_cols=[],
        describe_stats={},
        outlier_flags={"a": {"count": 0, "pct": 0.0}, "b": {"count": 0, "pct": 0.0}},
        skewness={"a": 0.1, "b": -0.2},
        dtype_issues={},
        clean_csv_path="/tmp/clean.csv",
    )
    defaults.update(overrides)
    return ProfileReport(**defaults)


class TestComputeQualityScore:
    def test_perfect_data_returns_high_score(self):
        profile = _make_profile()
        result = compute_quality_score(profile)
        assert result["overall"] >= 95.0
        assert result["grade"] == "A"

    def test_high_null_rate_lowers_completeness(self):
        profile = _make_profile(
            null_pcts={"a": 50.0, "b": 50.0, "c": 50.0},
        )
        result = compute_quality_score(profile)
        assert result["completeness"] == pytest.approx(50.0, abs=1.0)
        # overall = 50*0.35 + 100*0.25 + 100*0.20 + 100*0.20 = 82.5
        assert result["overall"] < 90.0

    def test_duplicates_lower_uniqueness(self):
        profile = _make_profile(duplicate_pct=40.0)
        result = compute_quality_score(profile)
        assert result["uniqueness"] == pytest.approx(60.0, abs=1.0)

    def test_dtype_issues_lower_validity(self):
        profile = _make_profile(
            dtype_issues={"a": "likely_datetime", "b": "likely_numeric"},
            dtypes={"a": "object", "b": "object", "c": "object"},
        )
        result = compute_quality_score(profile)
        # 2 issues out of 3 cols = 33% validity penalty
        assert result["validity"] < 70.0

    def test_outliers_lower_consistency(self):
        profile = _make_profile(
            outlier_flags={"a": {"count": 20, "pct": 20.0}, "b": {"count": 15, "pct": 15.0}},
        )
        result = compute_quality_score(profile)
        # avg_outlier_pct = 17.5 → penalty = 35 → consistency = 65
        assert result["consistency"] == pytest.approx(65.0, abs=1.0)

    def test_grade_a_above_90(self):
        assert compute_quality_score(_make_profile())["grade"] == "A"

    def test_grade_b_75_to_90(self):
        # null=40%→completeness=60, dup=20%→uniqueness=80, no other issues
        # overall = 60*0.35 + 80*0.25 + 100*0.20 + 100*0.20 = 21+20+20+20 = 81 → Grade B
        profile = _make_profile(
            null_pcts={"a": 40.0, "b": 40.0, "c": 40.0},
            duplicate_pct=20.0,
        )
        result = compute_quality_score(profile)
        assert result["grade"] == "B"
        assert 75.0 <= result["overall"] < 90.0

    def test_grade_d_below_60(self):
        profile = _make_profile(
            null_pcts={"a": 80.0, "b": 80.0, "c": 80.0},
            duplicate_pct=50.0,
            outlier_flags={"a": {"count": 30, "pct": 30.0}, "b": {"count": 25, "pct": 25.0}},
            dtype_issues={"a": "likely_datetime"},
            dtypes={"a": "object", "b": "object", "c": "object"},
        )
        result = compute_quality_score(profile)
        assert result["grade"] == "D"

    def test_scores_are_bounded_0_to_100(self):
        profile = _make_profile(
            null_pcts={"a": 200.0},  # absurdly large – should still clamp to 0
        )
        result = compute_quality_score(profile)
        for key in ("overall", "completeness", "uniqueness", "validity", "consistency"):
            assert 0.0 <= result[key] <= 100.0

    def test_verdict_present_and_non_empty(self):
        result = compute_quality_score(_make_profile())
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 5

    def test_no_outlier_flags_gives_full_consistency(self):
        profile = _make_profile(outlier_flags={})
        result = compute_quality_score(profile)
        assert result["consistency"] == pytest.approx(100.0)

    def test_no_dtype_issues_gives_full_validity(self):
        profile = _make_profile(dtype_issues={})
        result = compute_quality_score(profile)
        assert result["validity"] == pytest.approx(100.0)

    def test_weights_sum_correctly(self):
        """Weighted sum: 0.35 + 0.25 + 0.20 + 0.20 = 1.00."""
        profile = _make_profile()
        result = compute_quality_score(profile)
        expected = (
            result["completeness"] * 0.35
            + result["uniqueness"] * 0.25
            + result["validity"] * 0.20
            + result["consistency"] * 0.20
        )
        assert result["overall"] == pytest.approx(expected, abs=0.1)
