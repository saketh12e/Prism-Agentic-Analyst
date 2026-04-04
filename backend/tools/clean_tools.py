"""
PRISM — Profile Agent / Phase 2: Cleaning Tools
All tools read the current CSV, apply one transformation, and write the result
to output_path (which may equal df_path to overwrite in-place).
"""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool


@tool
def fix_nulls(df_path: str, col: str, strategy: str, output_path: str) -> dict:
    """
    Fill or remove null values in a single column.

    strategy options:
      'mean'         — fill with column mean (numeric only)
      'median'       — fill with column median (numeric only)
      'mode'         — fill with most frequent value (any dtype)
      'zero'         — fill with 0
      'forward_fill' — propagate last valid observation forward
      'drop'         — drop rows where this column is null
    """
    df = pd.read_csv(df_path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    before = int(df[col].isnull().sum())

    if strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    elif strategy == "zero":
        df[col] = df[col].fillna(0)
    elif strategy == "forward_fill":
        df[col] = df[col].ffill()
    elif strategy == "drop":
        df = df.dropna(subset=[col])
    else:
        return {"error": f"Unknown strategy '{strategy}'"}

    df.to_csv(output_path, index=False)
    return {
        "col": col,
        "strategy": strategy,
        "nulls_before": before,
        "nulls_after": int(df[col].isnull().sum()),
        "output_path": output_path,
    }


@tool
def remove_duplicates(df_path: str, output_path: str) -> dict:
    """Remove fully-duplicate rows and write the deduplicated CSV."""
    df = pd.read_csv(df_path)
    before = len(df)
    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)
    return {
        "rows_before": before,
        "rows_after": len(df),
        "removed": before - len(df),
        "output_path": output_path,
    }


@tool
def fix_dtype(df_path: str, col: str, target_type: str, output_path: str) -> dict:
    """
    Convert a column to the correct dtype.

    target_type options:
      'datetime' — pd.to_datetime (coerce invalid → NaT)
      'numeric'  — pd.to_numeric  (coerce invalid → NaN)
      'int'      — nullable integer (Int64)
      'float'    — float64
      'string'   — object/string
    """
    df = pd.read_csv(df_path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    original = str(df[col].dtype)

    if target_type == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
    elif target_type == "numeric":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif target_type == "int":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    elif target_type == "float":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    elif target_type == "string":
        df[col] = df[col].astype(str)
    else:
        return {"error": f"Unknown target_type '{target_type}'"}

    df.to_csv(output_path, index=False)
    return {
        "col": col,
        "dtype_from": original,
        "dtype_to": target_type,
        "output_path": output_path,
    }


@tool
def cap_outliers(df_path: str, col: str, output_path: str) -> dict:
    """
    Winsorise outliers in a numeric column by capping values at the IQR bounds
    (Q1 - 1.5*IQR  and  Q3 + 1.5*IQR).  Preserves row count.
    """
    df = pd.read_csv(df_path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    capped = df[col].clip(lower=lower, upper=upper)
    capped_count = int((df[col] != capped).sum())
    df[col] = capped
    df.to_csv(output_path, index=False)
    return {
        "col": col,
        "capped_count": capped_count,
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
        "output_path": output_path,
    }


@tool
def strip_whitespace(df_path: str, col: str, output_path: str) -> dict:
    """Strip leading/trailing whitespace from all string values in a column."""
    df = pd.read_csv(df_path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    df[col] = df[col].astype(str).str.strip()
    df.to_csv(output_path, index=False)
    return {"col": col, "action": "whitespace_stripped", "output_path": output_path}
