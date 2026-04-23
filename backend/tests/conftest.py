"""
Shared pytest fixtures for PRISM backend tests.
Sets up sys.path so graph.* and tools.* imports work without installing the package.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Must happen before any local import so graph/ and tools/ resolve correctly.
_BACKEND = Path(__file__).parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ── Shared CSV fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small, well-structured DataFrame used in most tool tests."""
    return pd.DataFrame({
        "age":      [25, 30, 35, 40, 45, 30, None, 50],
        "salary":   [50000, 60000, 70000, 80000, 90000, 60000, 55000, 100000],
        "city":     ["NYC", "LA", "NYC", "LA", "Chicago", "NYC", "LA", "Chicago"],
        "score":    [8.5, 7.0, 9.0, 6.5, 8.0, 7.5, 6.0, 9.5],
        "employed": ["yes", "no", "yes", "yes", "no", "yes", "no", "yes"],
    })


@pytest.fixture
def csv_path(sample_df, tmp_path) -> str:
    """Write sample_df to a temporary CSV and return the file path."""
    path = str(tmp_path / "test_data.csv")
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def clean_csv_path(sample_df, tmp_path) -> str:
    """CSV with no nulls (for tools that expect already-cleaned data)."""
    df = sample_df.copy()
    df["age"] = df["age"].fillna(df["age"].mean())
    path = str(tmp_path / "clean_data.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def two_group_csv(tmp_path) -> str:
    """CSV designed for t-test / ANOVA tests (clear group separation)."""
    df = pd.DataFrame({
        "group":  ["A"] * 30 + ["B"] * 30,
        "value":  [10.0 + i * 0.1 for i in range(30)] + [20.0 + i * 0.1 for i in range(30)],
    })
    path = str(tmp_path / "two_group.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def correlated_csv(tmp_path) -> str:
    """CSV with a strong positive correlation between x and y."""
    import numpy as np
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 50)
    y = 2 * x + rng.normal(0, 5, 50)
    df = pd.DataFrame({"x": x, "y": y})
    path = str(tmp_path / "correlated.csv")
    df.to_csv(path, index=False)
    return path
