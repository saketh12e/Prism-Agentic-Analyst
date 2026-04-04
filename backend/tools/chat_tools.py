"""
PRISM — Chat Agent Tools
On-demand pandas code execution and profile querying for the chat interface.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap

import pandas as pd
import plotly.express as px
from langchain_core.tools import tool

_ACCENT = "#6366f1"
_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0"),
    title_font=dict(size=15),
)


@tool
def execute_pandas(df_path: str, code: str) -> str:
    """
    Execute arbitrary pandas code against the cleaned CSV in an isolated subprocess.

    Rules:
    - The DataFrame is pre-loaded as `df`.
    - Your code MUST assign the final result to a variable named `output`.
    - Execution timeout is 20 seconds.
    - Returns a string representation of `output`, or an error message.

    Example code:
        output = df.groupby('category')['sales'].sum().sort_values(ascending=False).head(5)
    """
    script = textwrap.dedent(f"""
import pandas as pd
import numpy as np

df = pd.read_csv('{df_path}')

{code}

if 'output' not in dir():
    print("ERROR: code did not set an 'output' variable")
elif hasattr(output, 'to_string'):
    print(output.to_string())
else:
    print(output)
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname],
            capture_output=True,
            text=True,
            timeout=20,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0 or not stdout:
            return f"ERROR: {stderr or 'empty output'}"
        return stdout
    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out (20s limit)"
    finally:
        os.unlink(fname)


@tool
def auto_chart_from_query(df_path: str, code: str, title: str, chart_hint: str = "auto") -> str:
    """
    Run pandas code, inspect the result shape, and auto-generate a Plotly chart.

    code must assign a pandas Series or DataFrame to `output`.
    chart_hint: 'bar' | 'line' | 'scatter' | 'auto'
    Returns a Plotly JSON string.
    """
    try:
        df_full = pd.read_csv(df_path)
        local_ns: dict = {"df": df_full, "pd": pd}
        exec(code, local_ns)  # noqa: S102
        output = local_ns.get("output")
        if output is None:
            return json.dumps({"error": "code did not set 'output'"})

        # Normalise to a 2-column DataFrame
        if isinstance(output, pd.Series):
            plot_df = output.reset_index()
            plot_df.columns = ["x", "y"]
        elif isinstance(output, pd.DataFrame):
            plot_df = output.reset_index(drop=True)
            cols = list(plot_df.columns)
            plot_df = plot_df[[cols[0], cols[1]]].rename(columns={cols[0]: "x", cols[1]: "y"})
        else:
            return json.dumps({"error": f"output type not supported: {type(output)}"})

        # Pick chart type
        hint = chart_hint.lower()
        if hint == "line" or (hint == "auto" and pd.api.types.is_datetime64_any_dtype(plot_df["x"])):
            fig = px.line(plot_df, x="x", y="y", title=title, markers=True)
            fig.update_traces(line_color=_ACCENT)
        elif hint == "scatter":
            fig = px.scatter(plot_df, x="x", y="y", title=title, opacity=0.6, trendline="ols")
        else:
            fig = px.bar(plot_df, x="x", y="y", title=title,
                         color="y", color_continuous_scale="Viridis")

        fig.update_layout(**_DARK)
        return fig.to_json()
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def query_profile(profile_json: str, question: str) -> str:
    """
    Answer a structural question about the dataset using the stored ProfileReport JSON.
    This avoids reloading the CSV for questions like 'how many columns?'.

    question examples:
    - 'how many rows?'
    - 'what are the numeric columns?'
    - 'which column has the most nulls?'
    """
    try:
        p = json.loads(profile_json)
    except Exception:
        return "Could not parse profile JSON."

    q = question.lower()
    if "row" in q:
        return f"The dataset has {p.get('shape', [None])[0]:,} rows."
    if "column" in q and "numeric" in q:
        return f"Numeric columns: {', '.join(p.get('numeric_cols', []))}"
    if "column" in q and "categor" in q:
        return f"Categorical columns: {', '.join(p.get('categorical_cols', []))}"
    if "null" in q or "missing" in q:
        nulls = p.get("null_pcts", {})
        worst = max(nulls, key=nulls.get) if nulls else "none"
        return f"Highest null rate: '{worst}' at {nulls.get(worst, 0):.1f}%"
    if "duplicate" in q:
        return f"{p.get('duplicate_count', 0)} duplicate rows ({p.get('duplicate_pct', 0):.1f}%)"
    if "skew" in q:
        skew = p.get("skewness", {})
        high = {k: v for k, v in skew.items() if abs(v) > 1}
        return f"Highly skewed columns (|skew|>1): {high}" if high else "No highly skewed columns."

    return "Could not answer that from the profile alone. Try execute_pandas for custom queries."
