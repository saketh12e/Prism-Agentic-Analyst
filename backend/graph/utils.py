"""
PRISM — Shared agent utilities
"""

from __future__ import annotations

import json
import re


def extract_text(content) -> str:
    """
    Safely extract a plain string from an LLM message content field.

    Gemini (and other providers) sometimes return content as:
      - str                  → return as-is
      - list of str          → join
      - list of dicts        → extract "text" key from each part
      - anything else        → str()
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return " ".join(p for p in parts if p)
    return str(content) if content else ""


def parse_json_from_text(text: str) -> dict | None:
    """
    Extract the first JSON block from a free-form LLM response.
    Tries ```json ... ```, ``` ... ```, then raw { ... }.
    """
    for pattern in [r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    # Last resort: find first { and parse from there
    try:
        start = text.index("{")
        return json.loads(text[start:])
    except (ValueError, json.JSONDecodeError):
        return None


def last_ai_text(messages: list) -> str:
    """
    Return the text content of the last AI/assistant message in a list.
    Handles both str and list content via extract_text().
    """
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content is None:
            continue
        text = extract_text(content)
        if text.strip():
            return text
    return ""
