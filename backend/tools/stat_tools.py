"""
PRISM — Stat Agent Tools
Statistical tests and time-series analysis on the cleaned CSV.
"""

from __future__ import annotations

import pandas as pd
import scipy.stats
from langchain_core.tools import tool


@tool
def run_pearson(df_path: str, col_a: str, col_b: str) -> dict:
    """
    Pearson correlation between two numeric columns.
    Returns r, p-value, significance flag, and direction.
    """
    df = pd.read_csv(df_path)
    series_a = df[col_a].dropna()
    series_b = df[col_b].dropna()
    common = series_a.index.intersection(series_b.index)
    r, p = scipy.stats.pearsonr(series_a[common], series_b[common])
    return {
        "test": "pearson",
        "col_a": col_a,
        "col_b": col_b,
        "r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "direction": "positive" if r > 0 else "negative",
    }


@tool
def run_chi_square(df_path: str, col_a: str, col_b: str) -> dict:
    """
    Chi-square test of independence between two categorical columns.
    Returns chi2 statistic, p-value, degrees of freedom, and significance flag.
    """
    df = pd.read_csv(df_path)
    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = scipy.stats.chi2_contingency(ct)
    return {
        "test": "chi_square",
        "col_a": col_a,
        "col_b": col_b,
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p), 6),
        "dof": int(dof),
        "significant": bool(p < 0.05),
    }


@tool
def run_ttest(df_path: str, num_col: str, group_col: str) -> dict:
    """
    Independent-samples t-test comparing a numeric column across the first two
    groups of a binary categorical column.
    Returns t, p-value, Cohen's d effect size, group means, and significance flag.
    """
    df = pd.read_csv(df_path)
    groups = df[group_col].dropna().unique()[:2]
    if len(groups) < 2:
        return {"error": f"Need exactly 2 groups in '{group_col}', found {len(groups)}"}
    g1 = df[df[group_col] == groups[0]][num_col].dropna()
    g2 = df[df[group_col] == groups[1]][num_col].dropna()
    t, p = scipy.stats.ttest_ind(g1, g2)
    pooled_std = (g1.std() + g2.std()) / 2
    cohen_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std != 0 else 0.0
    return {
        "test": "ttest",
        "num_col": num_col,
        "group_col": group_col,
        "t": round(float(t), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "effect_size_cohens_d": round(float(cohen_d), 3),
        "groups": [str(g) for g in groups],
        "group_means": [round(float(g1.mean()), 2), round(float(g2.mean()), 2)],
    }


@tool
def run_anova(df_path: str, num_col: str, group_col: str) -> dict:
    """
    One-way ANOVA comparing a numeric column across all groups of a categorical column.
    Returns F-statistic, p-value, per-group means, and significance flag.
    """
    df = pd.read_csv(df_path)
    group_arrays = [g[num_col].dropna().values for _, g in df.groupby(group_col)]
    if len(group_arrays) < 2:
        return {"error": "ANOVA requires at least 2 groups"}
    f_stat, p = scipy.stats.f_oneway(*group_arrays)
    group_means = df.groupby(group_col)[num_col].mean().round(2).to_dict()
    return {
        "test": "anova",
        "num_col": num_col,
        "group_col": group_col,
        "f_stat": round(float(f_stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "group_means": {str(k): float(v) for k, v in group_means.items()},
    }


@tool
def run_correlation_matrix(df_path: str) -> dict:
    """
    Full Pearson correlation matrix for all numeric columns.
    Returns the matrix as a dict-of-dicts and the top-5 strongest pairs.
    """
    df = pd.read_csv(df_path)
    num_df = df.select_dtypes("number")
    corr = num_df.corr().round(3)
    cols = list(corr.columns)
    pairs = [
        {"col_a": cols[i], "col_b": cols[j], "r": float(corr.iloc[i, j])}
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {
        "matrix": corr.to_dict(),
        "top_pairs": pairs[:5],
    }


@tool
def detect_date_columns(df_path: str) -> dict:
    """
    Scan all columns and identify those that parse cleanly as dates.
    Returns a list of {"col": str, "sample": str}.
    """
    df = pd.read_csv(df_path)
    date_cols = []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            date_cols.append({"col": col, "sample": str(parsed.iloc[0])})
        except Exception:
            pass
    return {"date_columns": date_cols}


@tool
def run_time_trend(df_path: str, date_col: str, value_col: str, freq: str = "M") -> dict:
    """
    Resample a numeric column by a date column and compute trend statistics.

    freq: 'D'=daily | 'W'=weekly | 'ME'=monthly | 'QE'=quarterly | 'YE'=yearly
    Returns resampled trend, growth rates, peak period, and overall direction.
    """
    df = pd.read_csv(df_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    trend = df[value_col].resample(freq).mean().dropna()
    if trend.empty:
        return {"error": "No data after resampling — check date_col and freq"}
    growth = trend.pct_change().mul(100).round(2).dropna()
    return {
        "date_col": date_col,
        "value_col": value_col,
        "freq": freq,
        "trend": {str(k): round(float(v), 2) for k, v in trend.items()},
        "growth_rates": {str(k): float(v) for k, v in growth.items()},
        "peak_period": str(trend.idxmax()),
        "lowest_period": str(trend.idxmin()),
        "overall_trend": "up" if trend.iloc[-1] > trend.iloc[0] else "down",
        "start_value": round(float(trend.iloc[0]), 2),
        "end_value": round(float(trend.iloc[-1]), 2),
    }


@tool
def run_segment_compare(df_path: str, segment_col: str, value_col: str) -> dict:
    """
    Compare a numeric column across all segments of a categorical column.
    Returns mean, median, count, and std for each segment.
    """
    df = pd.read_csv(df_path)
    stats = (
        df.groupby(segment_col)[value_col]
        .agg(["mean", "median", "count", "std"])
        .round(2)
    )
    return {
        "segment_col": segment_col,
        "value_col": value_col,
        "segment_stats": stats.to_dict(),
    }
