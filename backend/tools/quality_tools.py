"""
PRISM — Data Quality Scoring
Computes a 0-100 quality score with dimensional breakdown from a ProfileReport.
"""
from __future__ import annotations

from graph.state import ProfileReport


def compute_quality_score(profile: ProfileReport) -> dict:
    """
    Compute a 0–100 data quality score across four dimensions:
      - Completeness: low missing-value rates across all columns
      - Uniqueness:   low duplicate-row percentage
      - Validity:     no dtype-mismatch / mixed-type issues
      - Consistency:  low outlier rates in numeric columns

    Weights: Completeness 35%, Uniqueness 25%, Validity 20%, Consistency 20%.
    Returns overall score, per-dimension scores, letter grade, and a verdict string.
    """
    # Completeness — penalise by average null % across all columns
    null_pcts = profile.null_pcts or {}
    avg_null_pct = sum(null_pcts.values()) / len(null_pcts) if null_pcts else 0.0
    completeness = max(0.0, 100.0 - avg_null_pct)

    # Uniqueness — penalise by duplicate row %
    uniqueness = max(0.0, 100.0 - profile.duplicate_pct)

    # Validity — penalise for columns flagged with dtype mismatches
    total_cols = len(profile.dtypes) or 1
    issue_count = len(profile.dtype_issues or {})
    validity = max(0.0, 100.0 - (issue_count / total_cols) * 100.0)

    # Consistency — penalise for outlier-heavy numeric columns (capped at 50%)
    outlier_pcts = [
        (v.get("pct", 0.0) if isinstance(v, dict) else 0.0)
        for v in (profile.outlier_flags or {}).values()
    ]
    avg_outlier_pct = sum(outlier_pcts) / len(outlier_pcts) if outlier_pcts else 0.0
    consistency = max(0.0, 100.0 - avg_outlier_pct * 2.0)

    overall = (
        completeness * 0.35
        + uniqueness  * 0.25
        + validity    * 0.20
        + consistency * 0.20
    )

    grade = (
        "A" if overall >= 90 else
        "B" if overall >= 75 else
        "C" if overall >= 60 else
        "D"
    )
    verdict = {
        "A": "Excellent — data is analysis-ready with minimal issues.",
        "B": "Good — minor quality concerns; review flagged columns.",
        "C": "Fair — notable quality issues that may affect results.",
        "D": "Poor — significant data quality problems detected.",
    }[grade]

    return {
        "overall":      round(overall, 1),
        "completeness": round(completeness, 1),
        "uniqueness":   round(uniqueness, 1),
        "validity":     round(validity, 1),
        "consistency":  round(consistency, 1),
        "grade":        grade,
        "verdict":      verdict,
    }
