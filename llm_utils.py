"""Gemini API + data profiling utilities for the Chat with Your CSV app.

Public interface retained from the OpenAI version:
    MODEL_PRICING, AVAILABLE_MODELS, build_data_context, ask_llm_stream
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# USD per million tokens, Gemini Developer API standard paid tier.
# Prices may change; free-tier requests are not charged when eligible.
MODEL_PRICING = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
}
AVAILABLE_MODELS = list(MODEL_PRICING)

MAX_RESPONSE_TOKENS = 700
TEMPERATURE = 0.2
HISTORY_TURNS_TO_KEEP = 6

SYSTEM_PROMPT_TEMPLATE = """You are a data analyst assistant. The user has uploaded a CSV file \
and will ask you questions about it in plain English.

Answer their questions using ONLY the data summary below. Be precise, concise, and data-driven. \
If the summary does not contain enough information to answer a question (for example, the user \
asks for a specific value you cannot compute from the summary), say so clearly and suggest what \
the user could check in the raw data.

When relevant, reference specific column names and values from the summary. Keep answers to \
2 to 4 short paragraphs unless the user explicitly asks for more detail.

=== DATA SUMMARY ===
{data_context}
===================="""


def get_gemini_client() -> genai.Client:
    """Load GEMINI_API_KEY from Streamlit secrets or the environment."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            'Gemini API key not found.\n\n'
            '**Local:** create `.streamlit/secrets.toml` with '
            '`GEMINI_API_KEY = "your-api-key"`.\n\n'
            '**Cloud:** add `GEMINI_API_KEY` to your Streamlit app Settings → Secrets.'
        )
        st.stop()
    return genai.Client(api_key=api_key)


def build_data_context(df: pd.DataFrame) -> str:
    """Build a compact CSV profile; behavior preserved from the original file."""
    parts: list[str] = []
    parts.append(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    parts.append("\nColumns and data types:")
    for col, dtype in df.dtypes.items():
        parts.append(f"  - {col}: {dtype}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        parts.append("\nNumeric column statistics (from df.describe):")
        parts.append(df[numeric_cols].describe().round(2).to_string())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        parts.append("\nCategorical / text columns (top 5 values each):")
        for col in cat_cols[:5]:
            counts = df[col].value_counts(dropna=False).head(5)
            parts.append(f"\n  {col}:")
            for val, count in counts.items():
                parts.append(f"    {val}: {count:,}")

    parts.append("\nFirst 5 rows (as a sample of what the raw data looks like):")
    parts.append(df.head(5).to_string(index=False))

    null_counts = df.isna().sum()
    has_nulls = null_counts[null_counts > 0]
    if not has_nulls.empty:
        parts.append("\nColumns with missing values (count):")
        for col, n in has_nulls.items():
            parts.append(f"  - {col}: {n:,}")

    return "\n".join(parts)


def ask_llm_stream(
    user_message: str,
    data_context: str,
    history: list[dict],
    model: str = "gemini-3.5-flash-lite",
    usage_tracker: Optional[dict] = None,
) -> Iterator[str]:
    """Stream Gemini text to st.write_stream and update token usage in-place.

    `history` uses Streamlit/OpenAI-style roles: user and assistant.
    The public signature and tracker keys are unchanged from the original.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(data_context=data_context)
    contents: list[types.Content] = []
    for msg in history[-HISTORY_TURNS_TO_KEEP:]:
        role = msg.get("role")
        if role not in ("user", "assistant", "model"):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        contents.append(types.Content(
            role="model" if role in ("assistant", "model") else "user",
            parts=[types.Part.from_text(text=content)],
        ))
    contents.append(types.Content(
        role="user", parts=[types.Part.from_text(text=user_message)]
    ))

    if usage_tracker is not None:
        usage_tracker["input_tokens"] = 0
        usage_tracker["output_tokens"] = 0

    try:
        client = get_gemini_client()
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_RESPONSE_TOKENS,
            ),
        )
        for chunk in stream:
            # Usage metadata may be present on intermediate or final chunks.
            usage = getattr(chunk, "usage_metadata", None)
            if usage is not None and usage_tracker is not None:
                usage_tracker["input_tokens"] = int(
                    getattr(usage, "prompt_token_count", 0) or 0
                )
                # Include thinking tokens when present (part of billable output).
                usage_tracker["output_tokens"] = (
                    int(getattr(usage, "candidates_token_count", 0) or 0)
                    + int(getattr(usage, "thoughts_token_count", 0) or 0)
                )
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        yield f"\n\nError calling Gemini: {exc}"
