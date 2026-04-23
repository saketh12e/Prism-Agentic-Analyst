"""
PRISM — Insight Agent Tools
Executes hypothesis-testing pandas code against the cleaned CSV in an
isolated subprocess and returns a structured result the agent can evaluate.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap

from langchain_core.tools import tool


@tool
def test_hypothesis(df_path: str, hypothesis: str, code: str) -> dict:
    """
    Test a data hypothesis by executing pandas code against the cleaned CSV.

    Parameters
    ----------
    df_path    : absolute path to the cleaned CSV
    hypothesis : a one-sentence description of what you are testing
                 (e.g. "Revenue is higher in Q4 than in other quarters")
    code       : pandas code that loads nothing — `df` is pre-loaded.
                 Must assign the key result to a variable named `output`.
                 `output` may be a scalar, Series, or DataFrame.

    Returns a dict with:
      - hypothesis  : echoed back
      - result      : string representation of `output`
      - success     : True if code ran without errors
      - error       : error message if success is False
    """
    script = textwrap.dedent(f"""
import pandas as pd
import numpy as np

df = pd.read_csv('{df_path}')

{code}

if 'output' not in dir():
    print("ERROR: code did not assign to 'output'")
elif hasattr(output, 'to_string'):
    print(output.to_string())
else:
    print(output)
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name

    try:
        proc = subprocess.run(
            ["python", fname],
            capture_output=True,
            text=True,
            timeout=20,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0 or stdout.startswith("ERROR"):
            return {
                "hypothesis": hypothesis,
                "result": "",
                "success": False,
                "error": stderr or stdout or "unknown error",
            }
        return {
            "hypothesis": hypothesis,
            "result": stdout[:2000],  # cap to avoid LLM context overflow
            "success": True,
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "hypothesis": hypothesis,
            "result": "",
            "success": False,
            "error": "Execution timed out (20 s limit)",
        }
    finally:
        os.unlink(fname)
