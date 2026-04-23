"""
PRISM — Profile Agent / Phase 1: Inspection Tools
All tools read the CSV from disk; none mutate it.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool


@tool
def get_shape(df_path: str) -> dict:
    """Return the number of rows and columns in the CSV."""
    df = pd.read_csv(df_path)
    return {"rows": df.shape[0], "cols": df.shape[1]}


@tool
def get_dtypes(df_path: str) -> dict:
    """Return a mapping of column name → pandas dtype string."""
    df = pd.read_csv(df_path)
    return df.dtypes.astype(str).to_dict()


@tool
def get_null_report(df_path: str) -> dict:
    """Return null counts and null percentages for every column."""
    df = pd.read_csv(df_path)
    counts = df.isnull().sum().to_dict()
    pcts = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    return {"null_counts": counts, "null_pcts": pcts}


@tool
def get_duplicates(df_path: str) -> dict:
    """Return the count and percentage of fully-duplicate rows."""
    df = pd.read_csv(df_path)
    mask = df.duplicated()
    return {
        "duplicate_count": int(mask.sum()),
        "duplicate_pct": round(float(mask.mean()) * 100, 2),
        "sample_indices": mask[mask].index.tolist()[:5],
    }


@tool
def get_describe(df_path: str) -> dict:
    """Return pandas describe() for all numeric columns (count/mean/std/min/max etc.)."""
    df = pd.read_csv(df_path)
    return df.describe().round(3).to_dict()


@tool
def get_outliers(df_path: str, col: str) -> dict:
    """
    Detect outliers in a single numeric column using the IQR method.
    Returns count, percentage, and bounds.
    """
    df = pd.read_csv(df_path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    series = df[col].dropna()
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    return {
        "col": col,
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(float(mask.mean()) * 100, 2),
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
    }


@tool
def get_skewness(df_path: str) -> dict:
    """Return skewness for all numeric columns. |skew| > 1 = highly skewed."""
    df = pd.read_csv(df_path)
    return df.select_dtypes("number").skew().round(3).to_dict()


@tool
def detect_dtype_issues(df_path: str) -> dict:
    """
    Inspect all object columns and flag any that are actually numeric or datetime.
    Returns a dict of col -> 'likely_datetime' | 'likely_numeric'.
    """
    df = pd.read_csv(df_path)
    issues: dict[str, str] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            pd.to_datetime(df[col], errors="raise", format="mixed")
            issues[col] = "likely_datetime"
            continue
        except Exception:
            pass
        try:
            pd.to_numeric(df[col], errors="raise")
            issues[col] = "likely_numeric"
        except Exception:
            pass
    return {"dtype_issues": issues}


@tool
def get_value_counts(df_path: str, col: str, top_n: int = 10) -> dict:
    """
    Return the top-N most frequent values for a categorical column.
    Useful for understanding cardinality before plotting.
    """
    df = pd.read_csv(df_path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    vc = df[col].value_counts().head(top_n)
    return {
        "col": col,
        "unique_count": int(df[col].nunique()),
        "top_values": vc.to_dict(),
    }
